# SPDX-License-Identifier: GPL-3.0-or-later
"""
Turn Blender objects into a voxel grid for the optimizer.

This module *does* touch bpy/mathutils (it reads scene geometry), but it
returns plain numpy arrays so the optimizer core stays Blender-agnostic.

The inside/outside test is a per-point BVH ray-cast in Python, which is the
heaviest main-thread step of a run. To keep the viewport responsive it is
exposed as a *generator* (``build_grid_steps``) that yields progress every few
hundred points, so the modal operator can spread it over several timer ticks
instead of freezing the UI while a finer level is voxelized. ``build_grid`` is a
thin wrapper that just runs the generator to completion (used by headless code).

Node/element ordering matches core.fea (element ex + nx*(ey + ny*ez);
node ix + (nx+1)*(iy + (ny+1)*iz); DOFs 3n, 3n+1, 3n+2).
"""

import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

from . import inside_worker

try:
    import bpy  # noqa: F401
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    _HAS_BPY = True
except Exception:  # allow import in headless tests
    _HAS_BPY = False


class Grid:
    def __init__(self, dims, origin, vsize):
        self.nx, self.ny, self.nz = dims
        self.origin = np.asarray(origin, dtype=float)   # world min corner
        self.vsize = float(vsize)
        self.active = None
        self.fixed_dofs = None
        self.force = None

    @property
    def ndof(self):
        return 3 * (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    def voxel_centers(self):
        """(nelem, 3) world-space centers, element-index order."""
        xs = (np.arange(self.nx) + 0.5) * self.vsize + self.origin[0]
        ys = (np.arange(self.ny) + 0.5) * self.vsize + self.origin[1]
        zs = (np.arange(self.nz) + 0.5) * self.vsize + self.origin[2]
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    def node_coords(self):
        """((nx+1)*(ny+1)*(nz+1), 3) world node positions, ordered so the array
        index EQUALS the FEA global node id used in core.fea
        (id = ix + (nx+1)*(iy + (ny+1)*iz), i.e. x fastest). This alignment is
        what makes bearings/loads land on the correct nodes."""
        nxp, nyp, nzp = self.nx + 1, self.ny + 1, self.nz + 1
        n = np.arange(nxp * nyp * nzp)
        ix = n % nxp
        iy = (n // nxp) % nyp
        iz = n // (nxp * nyp)
        coords = np.stack([ix, iy, iz], axis=1).astype(float) * self.vsize
        return coords + self.origin

    def node_id(self, ix, iy, iz):
        return ix + (self.nx + 1) * (iy + (self.ny + 1) * iz)


def _bvh_from_object(obj, depsgraph):
    """World-space BVHTree of an evaluated mesh object."""
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    mw = obj.matrix_world
    verts = [mw @ v.co for v in mesh.vertices]
    polys = [tuple(p.vertices) for p in mesh.polygons]
    tree = BVHTree.FromPolygons(verts, polys, all_triangles=False, epsilon=0.0)
    eval_obj.to_mesh_clear()
    return tree


# Slightly tilted, non-axis-aligned ray. With an axis-aligned build mesh and an
# axis-aligned voxel grid, a pure +X ray grazes faces/edges exactly edge-on and
# the crossing-parity test misfires, punching scattered holes in the mask. A
# tilted direction can never lie in a face plane, so parity is robust.
_rd = np.array([1.0, 0.0073301, 0.0031337])
_RAY_DIR_T = tuple(_rd / np.linalg.norm(_rd))   # normalized, no mathutils needed


def _point_inside(tree, p, span, d, eps=1e-5):
    """Robust parity ray-cast for a single point: inside a closed mesh?"""
    cur = Vector((float(p[0]), float(p[1]), float(p[2])))
    count = 0
    remaining = float(span)
    while remaining > 0.0:
        loc, nrm, idx, dist = tree.ray_cast(cur, d, remaining)
        if loc is None:
            break
        count += 1
        step = (loc - cur).length + eps
        cur = loc + d * eps
        remaining -= step
        if count > 512:
            break
    return (count % 2) == 1


# How many points to test between progress checkpoints. Small enough that a
# single chunk is a few milliseconds even on a fine grid, so the caller can
# honour a per-tick time budget and keep the UI fluid.
_CHUNK = 256


def _inside_mask_steps(tree, points, span):
    """Generator: fill an inside/outside bool mask, yielding every _CHUNK
    points. Returns the finished mask (via StopIteration.value)."""
    inside = np.zeros(len(points), dtype=bool)
    d = Vector(_RAY_DIR_T)
    for i in range(len(points)):
        inside[i] = _point_inside(tree, points[i], span, d)
        if (i + 1) % _CHUNK == 0:
            yield _CHUNK
    return inside


def _inside_mask(tree, points, span):
    """Non-incremental convenience wrapper (runs the generator to the end)."""
    gen = _inside_mask_steps(tree, points, span)
    try:
        while True:
            next(gen)
    except StopIteration as done:
        return done.value


def _grid_and_reach(settings, resolution=None):
    """Compute the empty Grid (dims/origin/vsize) and the ray ``reach`` from the
    build-space bounding box. Cheap: O(8 corners), no per-point arrays. This is
    all the main thread needs for the async path - the worker builds the big
    voxel-center / node arrays itself."""
    bs = settings.build_space
    if bs is None:
        raise ValueError("No build space set")
    if resolution is None:
        resolution = settings.resolution

    # Bounding box of build space in world space.
    corners = [bs.matrix_world @ Vector(c) for c in bs.bound_box]
    cmin = np.min([[c.x, c.y, c.z] for c in corners], axis=0)
    cmax = np.max([[c.x, c.y, c.z] for c in corners], axis=0)
    ext = cmax - cmin
    longest = float(np.max(ext))
    vsize = longest / resolution
    dims = tuple(max(1, int(np.ceil(e / vsize))) for e in ext)

    # Centre the grid on the AABB so the voxel lattice is symmetric about the
    # build space (anchoring at the min corner caused a visible drift).
    total_ext = np.array(dims, dtype=float) * vsize
    origin = cmin - 0.5 * (total_ext - ext)
    grid = Grid(dims, origin, vsize)

    reach = float(np.sqrt((total_ext ** 2).sum())) + 2.0 * vsize
    return grid, reach


def _build_grid_shell(settings, resolution=None):
    """As ``_grid_and_reach`` but also materializes the voxel-center / node
    coordinate arrays. Used by the in-process generator (and tests)."""
    grid, reach = _grid_and_reach(settings, resolution)
    return grid, reach, grid.voxel_centers(), grid.node_coords()


def build_grid_steps(settings, depsgraph, resolution=None):
    """Generator that builds a Grid incrementally.

    Yields ``('progress', done_points, total_points)`` while the (heavy) inside
    tests run, then finally yields ``('grid', grid)``. The modal operator drains
    this under a per-tick time budget so voxelizing a finer level no longer
    freezes the viewport.

    This is the in-process BVH path (still used headless/in tests and as a
    fallback when no subprocess can be launched). The interactive operator
    prefers ``AsyncVoxelizer``, which runs the same inside tests in a separate
    process so Blender's main thread stays completely free.
    """
    grid, reach, centers, nodes = _build_grid_shell(settings, resolution)
    bs = settings.build_space
    depsgraph.update()

    # Pre-count total work so the caller can show a percentage.
    n_excl = sum(1 for it in settings.exclude if it.obj is not None)
    n_incl = sum(1 for it in getattr(settings, "include", []) if it.obj is not None)
    n_bear = sum(1 for b in settings.bearings if b.obj is not None)
    n_load = sum(1 for ld in settings.loads if ld.obj is not None)
    total = len(centers) * (1 + n_excl + n_incl) + len(nodes) * (n_bear + n_load)
    done = [0]

    def _run(inner):
        """Drive a mask generator, surfacing progress; return its mask."""
        while True:
            try:
                step = next(inner)
            except StopIteration as fin:
                return fin.value
            done[0] += step
            yield ('progress', done[0], total)

    # Build-space inside test on voxel centers.
    bs_tree = _bvh_from_object(bs, depsgraph)
    inside = yield from _run(_inside_mask_steps(bs_tree, centers, reach))

    # Exclusions force voxels empty.
    excl = np.zeros(len(centers), dtype=bool)
    for item in settings.exclude:
        if item.obj is None:
            continue
        t = _bvh_from_object(item.obj, depsgraph)
        excl = excl | (yield from _run(_inside_mask_steps(t, centers, reach)))

    # Inclusions force voxels part of the design domain (keep-out still wins
    # where the two overlap). Note: unlike the preferred subprocess solver
    # path, this in-process fallback does not force the *displayed* density
    # to 1 inside keep-in volumes - it only guarantees they're part of the
    # active/optimizable domain.
    keepin = np.zeros(len(centers), dtype=bool)
    for item in getattr(settings, "include", []):
        if item.obj is None:
            continue
        t = _bvh_from_object(item.obj, depsgraph)
        keepin = keepin | (yield from _run(_inside_mask_steps(t, centers, reach)))
    keepin = keepin & ~excl

    grid.active = (inside | keepin) & ~excl

    # Bearings -> fixed node DOFs.
    fixed = []
    for b in settings.bearings:
        if b.obj is None:
            continue
        t = _bvh_from_object(b.obj, depsgraph)
        mask = yield from _run(_inside_mask_steps(t, nodes, reach))
        for n in np.where(mask)[0]:
            if b.fix_x:
                fixed.append(3 * n)
            if b.fix_y:
                fixed.append(3 * n + 1)
            if b.fix_z:
                fixed.append(3 * n + 2)
    grid.fixed_dofs = (np.unique(np.asarray(fixed, dtype=np.int64))
                       if fixed else np.zeros(0, dtype=np.int64))

    # Loads -> distributed nodal force vector.
    force = np.zeros(grid.ndof)
    for ld in settings.loads:
        if ld.obj is None:
            continue
        t = _bvh_from_object(ld.obj, depsgraph)
        mask = yield from _run(_inside_mask_steps(t, nodes, reach))
        nin = np.where(mask)[0]
        if len(nin) == 0:
            continue
        fv = np.asarray(ld.force, dtype=float) / len(nin)
        for n in nin:
            force[3 * n:3 * n + 3] += fv
    grid.force = force

    yield ('grid', grid)


def build_grid(settings, depsgraph, resolution=None):
    """Construct a Grid in one go (drains build_grid_steps). For headless use;
    the interactive operator uses build_grid_steps to stay responsive."""
    gen = build_grid_steps(settings, depsgraph, resolution=resolution)
    for tag, *rest in gen:
        if tag == 'grid':
            return rest[0]
    raise RuntimeError("voxelization produced no grid")


# ---------------------------------------------------------------------------
# Out-of-process voxelization (keeps Blender's main thread free)
# ---------------------------------------------------------------------------
#
# The per-point inside tests are the heaviest part of building a level. The
# BVH path above runs them on the main thread (chunked so it does not freeze,
# but it still competes for the main thread). AsyncVoxelizer instead does only
# the cheap bpy work on the main thread - reading each object's triangles into
# numpy arrays - then hands those arrays to a separate Python process that runs
# the inside tests (core.inside_worker) and writes the boolean masks back. The
# operator polls it from the modal timer; between polls the main thread is
# genuinely idle, so the viewport stays fully interactive.
#
# No threads or multiprocessing-of-bpy are involved: the child is a plain
# numpy-only script, so it cannot touch (or crash) Blender state.


def _object_triangles(obj, depsgraph):
    """World-space triangulated geometry of ``obj`` as numpy arrays.

    Returns ``(verts (V,3) float64, faces (T,3) int64)``. Runs on the main
    thread (needs bpy) but is cheap - only mesh extraction, no per-point work.
    """
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        nv = len(mesh.vertices)
        co = np.empty(nv * 3, dtype=np.float64)
        mesh.vertices.foreach_get('co', co)
        co = co.reshape(nv, 3)
        # Apply the world matrix to all vertices at once.
        mw = np.array(obj.matrix_world, dtype=np.float64)   # 4x4
        co_h = np.empty((nv, 4), dtype=np.float64)
        co_h[:, :3] = co
        co_h[:, 3] = 1.0
        verts = (co_h @ mw.T)[:, :3]

        mesh.calc_loop_triangles()
        nt = len(mesh.loop_triangles)
        tri = np.empty(nt * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get('vertices', tri)
        faces = tri.reshape(nt, 3)
    finally:
        eval_obj.to_mesh_clear()
    return verts, faces


def _worker_script_path():
    return os.path.join(os.path.dirname(__file__), "inside_worker.py")


class AsyncVoxelizer:
    """Run a level's inside tests in a subprocess; poll from the modal timer.

    Usage::

        vx = AsyncVoxelizer()
        vx.start(settings, depsgraph, resolution)   # main thread, cheap
        ...                                          # each timer tick:
        tag, payload = vx.poll()
        # ('running', frac) | ('grid', Grid) | ('error', message)

    ``start`` does all bpy access (extract geometry) and launches the worker.
    ``poll`` never blocks. ``cancel`` kills the worker and cleans up.
    """

    def __init__(self):
        self._proc = None
        self._tmpdir = None
        self._job_path = None
        self._result_path = None
        self._progress_path = None
        self._log_path = None
        self._grid = None
        self._descr = None          # per-query assembly metadata
        self._settings = None
        self._done = False

    def start(self, settings, depsgraph, resolution=None):
        """Extract geometry (main thread) and launch the worker process."""
        self._settings = settings
        grid, reach = _grid_and_reach(settings, resolution)
        self._grid = grid
        depsgraph.update()

        queries = []          # what the worker computes
        descr = []            # how the main thread interprets each mask
        bs = settings.build_space
        v, f = _object_triangles(bs, depsgraph)
        queries.append({'verts': v, 'faces': f, 'target': 'centers'})
        descr.append(('build', None))

        for item in settings.exclude:
            if item.obj is None:
                continue
            v, f = _object_triangles(item.obj, depsgraph)
            queries.append({'verts': v, 'faces': f, 'target': 'centers'})
            descr.append(('exclude', None))

        for item in getattr(settings, "include", []):
            if item.obj is None:
                continue
            v, f = _object_triangles(item.obj, depsgraph)
            queries.append({'verts': v, 'faces': f, 'target': 'centers'})
            descr.append(('include', None))

        for b in settings.bearings:
            if b.obj is None:
                continue
            v, f = _object_triangles(b.obj, depsgraph)
            queries.append({'verts': v, 'faces': f, 'target': 'nodes'})
            descr.append(('bearing', (bool(b.fix_x), bool(b.fix_y),
                                      bool(b.fix_z))))

        for ld in settings.loads:
            if ld.obj is None:
                continue
            v, f = _object_triangles(ld.obj, depsgraph)
            queries.append({'verts': v, 'faces': f, 'target': 'nodes'})
            descr.append(('load', np.asarray(ld.force, dtype=float)))

        self._descr = descr

        self._tmpdir = tempfile.mkdtemp(prefix="blendtopo_vox_")
        self._job_path = os.path.join(self._tmpdir, "job.pkl")
        self._result_path = os.path.join(self._tmpdir, "result.pkl")
        self._progress_path = os.path.join(self._tmpdir, "progress.txt")
        self._log_path = os.path.join(self._tmpdir, "worker.log")

        # Send only a tiny grid description (not the millions of points); the
        # worker regenerates centers/nodes itself. Keeps the main-thread job
        # write trivial even at fine resolutions.
        job = {'direction': inside_worker._RAY_DIR,
               'grid': {'dims': (grid.nx, grid.ny, grid.nz),
                        'origin': np.asarray(grid.origin, dtype=float),
                        'vsize': float(grid.vsize)},
               'queries': queries}
        with open(self._job_path, 'wb') as fh:
            pickle.dump(job, fh)

        cmd = [sys.executable, _worker_script_path(),
               self._job_path, self._result_path, self._progress_path]
        popen_kwargs = {}
        if os.name == 'nt':
            # Don't flash a console window on Windows.
            popen_kwargs['creationflags'] = 0x08000000   # CREATE_NO_WINDOW
        self._logf = open(self._log_path, 'wb')
        self._proc = subprocess.Popen(
            cmd, stdout=self._logf, stderr=self._logf, **popen_kwargs)

    def poll(self):
        """Non-blocking status. Returns one of:
        ('running', frac) | ('grid', Grid) | ('error', message)."""
        if self._done:
            return ('grid', self._grid)

        # Result ready?
        if os.path.exists(self._result_path):
            try:
                with open(self._result_path, 'rb') as fh:
                    result = pickle.load(fh)
            except (EOFError, pickle.UnpicklingError):
                return ('running', self._read_progress())   # still being written
            if result.get('error'):
                self._cleanup()
                return ('error', result['error'])
            grid = self._assemble(result['masks'])
            self._done = True
            self._cleanup()
            return ('grid', grid)

        # Process exited without a result file => it crashed early.
        if self._proc is not None and self._proc.poll() is not None:
            msg = self._read_log() or "worker exited without producing a result"
            self._cleanup()
            return ('error', msg)

        return ('running', self._read_progress())

    def _assemble(self, masks):
        """Combine the worker's masks into the finished Grid on the main
        thread (cheap numpy bookkeeping; mirrors build_grid_steps)."""
        grid = self._grid
        inside = None
        excl = np.zeros(grid.nx * grid.ny * grid.nz, dtype=bool)
        keepin = np.zeros(grid.nx * grid.ny * grid.nz, dtype=bool)
        fixed = []
        force = np.zeros(grid.ndof)

        for (role, meta), mask in zip(self._descr, masks):
            mask = np.asarray(mask, dtype=bool)
            if role == 'build':
                inside = mask
            elif role == 'exclude':
                excl = excl | mask
            elif role == 'include':
                keepin = keepin | mask
            elif role == 'bearing':
                fix_x, fix_y, fix_z = meta
                for n in np.where(mask)[0]:
                    if fix_x:
                        fixed.append(3 * n)
                    if fix_y:
                        fixed.append(3 * n + 1)
                    if fix_z:
                        fixed.append(3 * n + 2)
            elif role == 'load':
                nin = np.where(mask)[0]
                if len(nin) == 0:
                    continue
                fv = np.asarray(meta, dtype=float) / len(nin)
                for n in nin:
                    force[3 * n:3 * n + 3] += fv

        if inside is None:
            inside = np.zeros(grid.nx * grid.ny * grid.nz, dtype=bool)
        keepin = keepin & ~excl
        grid.active = (inside | keepin) & ~excl
        grid.fixed_dofs = (np.unique(np.asarray(fixed, dtype=np.int64))
                           if fixed else np.zeros(0, dtype=np.int64))
        grid.force = force
        return grid

    def _read_progress(self):
        try:
            with open(self._progress_path, 'r') as pf:
                return float(pf.read().strip() or 0.0)
        except Exception:
            return 0.0

    def _read_log(self):
        try:
            with open(self._log_path, 'rb') as lf:
                return lf.read().decode('utf-8', 'replace').strip()
        except Exception:
            return ""

    def cancel(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._cleanup()

    def _cleanup(self):
        logf = getattr(self, "_logf", None)
        if logf is not None:
            try:
                logf.close()
            except Exception:
                pass
            self._logf = None
        if self._tmpdir and os.path.isdir(self._tmpdir):
            import shutil
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None


def subprocess_available():
    """Whether the out-of-process path can be used (needs a real Python
    interpreter to launch and the worker script on disk)."""
    try:
        exe = sys.executable
        return bool(exe) and os.path.exists(_worker_script_path())
    except Exception:
        return False
