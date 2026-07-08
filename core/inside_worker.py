# SPDX-License-Identifier: GPL-3.0-or-later
"""
Point-in-mesh inside test, free of any Blender dependency.

This module is the heavy inner loop of voxelization (deciding which voxel
centers / nodes lie inside the build space, exclusions, bearings and loads).
It deliberately imports **only numpy** at module scope so it can run in a
plain Python subprocess that has no access to ``bpy``/``mathutils`` (those are
Blender C-extensions, only usable on Blender's main process). Keeping the work
out-of-process is what frees Blender's main thread while a level voxelizes.

Two backends:

* ``_inside_mask_numpy`` - a fully vectorized, single-direction parity
  ray-cast (Moeller-Trumbore). No external dependencies, so it always works
  even on a platform for which we didn't bundle a matching wheel. Its temp
  arrays are chunked over *both* points and triangles (see ``_POINT_CHUNK``
  below) to keep memory bounded on heavy meshes.
* ``_inside_mask_trimesh`` - uses ``trimesh`` + ``rtree`` for the broad-phase.
  Both are bundled as wheels under ``./wheels/`` and declared in
  ``blender_manifest.toml`` (see the Extensions Platform's "Be Self
  Contained" / "Bundle Modules" rules: a feature must not depend on a
  library the extension doesn't ship itself). Benchmarked against the numpy
  path in a controlled, paired, repeated-measures experiment (105 trials,
  icosphere test meshes 20-81920 triangles; see ``paper/experiments/`` for
  the full script, raw data, and figure): trimesh loses clearly below a few
  hundred triangles (its ~0.1-0.2s fixed overhead from building the Trimesh
  + rtree index dominates), is roughly break-even around 320 triangles, and
  wins with high statistical significance (paired t-test p < 0.02, p < 1e-4
  above ~5000 triangles) from ~1280 triangles up -- interpolated crossover
  ~551 triangles. The numpy path also becomes impractically slow (seconds
  to tens of seconds, or the memory blow-up described above) well before
  that. Kept as a separate function (rather than always-on) purely so the
  tiny/degenerate-mesh case still gets the faster numpy path, and so we
  have *some* fallback if a platform has no matching wheel.

It is also runnable as a standalone script (this is how the subprocess is
launched)::

    python inside_worker.py JOB.pkl RESULT.pkl PROGRESS.txt

JOB is a pickled dict ``{'direction', 'grid', 'queries'}`` where ``grid`` is the
tiny ``{'dims', 'origin', 'vsize'}`` description (the worker regenerates the
voxel-center / node arrays itself, so the Blender main thread never builds or
pickles those big arrays) and each query is ``{'verts', 'faces', 'target'}``
(target = 'centers' | 'nodes'). RESULT is written as
``{'masks': [bool ndarray, ...], 'error': str|None}`` in the same order as the
queries. PROGRESS receives a float in [0, 1] that the caller polls to drive the
UI percentage.
"""

import numpy as np

# Same tilted, non-axis-aligned ray direction the old BVH path used: with an
# axis-aligned build mesh and an axis-aligned voxel lattice a pure +X ray
# grazes faces edge-on and the crossing-parity test misfires, punching holes
# in the mask. A tilted direction can never lie in a face plane, so parity is
# robust. Kept here (numpy only) so the worker needs no mathutils.
_RAY_DIR = np.array([1.0, 0.0073301, 0.0031337])
_RAY_DIR = _RAY_DIR / np.linalg.norm(_RAY_DIR)

# Base points-per-chunk for the numpy path's (P_chunk x T x 3) temp arrays.
# This alone does NOT bound memory -- it must be scaled down as the triangle
# count grows (see _numpy_chunk_size below). A fixed 20000 with a 5000-tri
# mesh needs ~20000*5000*3*8 bytes (~2.4 GB) *per temp array*, of which there
# are several -- easily tens of GB and an OOM crash, well before trimesh even
# enters the picture.
_POINT_CHUNK = 20000

# Roughly how many (point, triangle) pairs we're willing to hold across the
# handful of same-shaped temp arrays in _inside_mask_numpy at once. Chosen to
# keep peak memory in the low hundreds of MB regardless of mesh size.
_MAX_PAIRS_PER_CHUNK = 4_000_000


def _numpy_chunk_size(n_tris):
    """Points-per-chunk for _inside_mask_numpy, bounded by triangle count.

    Chunking only over points (the old behavior) made memory scale with
    n_tris with no ceiling -- fine for the primitive box/cylinder meshes this
    was written against, but a real crash risk for a denser exclusion/bearing
    mesh a user might reuse from elsewhere.
    """
    if n_tris <= 0:
        return _POINT_CHUNK
    return max(1, min(_POINT_CHUNK, _MAX_PAIRS_PER_CHUNK // n_tris))


# --- grid point generation (mirrors core.voxelize.Grid; kept bpy-free) -------
# The worker generates the (potentially millions of) voxel-center / node
# coordinates itself from the cheap grid description, so the Blender main thread
# never has to build or pickle those big arrays. Formulas MUST match
# voxelize.Grid.voxel_centers / node_coords exactly (FEA node/element ordering).

def voxel_centers(dims, origin, vsize):
    nx, ny, nz = dims
    origin = np.asarray(origin, dtype=np.float64)
    vsize = float(vsize)
    xs = (np.arange(nx) + 0.5) * vsize + origin[0]
    ys = (np.arange(ny) + 0.5) * vsize + origin[1]
    zs = (np.arange(nz) + 0.5) * vsize + origin[2]
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def node_coords(dims, origin, vsize):
    nx, ny, nz = dims
    origin = np.asarray(origin, dtype=np.float64)
    vsize = float(vsize)
    nxp, nyp, nzp = nx + 1, ny + 1, nz + 1
    n = np.arange(nxp * nyp * nzp)
    ix = n % nxp
    iy = (n // nxp) % nyp
    iz = n // (nxp * nyp)
    coords = np.stack([ix, iy, iz], axis=1).astype(np.float64) * vsize
    return coords + origin

# Above this triangle count trimesh+rtree beats the numpy P*T broadcast;
# below it numpy wins outright. This is no longer a guess: see
# paper/experiments/ for a controlled, paired, repeated-measures benchmark
# (105 trials, icosphere test meshes 20-81920 triangles, paired t-tests).
# Trimesh loses clearly below a few hundred triangles (its ~0.1-0.2s fixed
# overhead from building the Trimesh object + rtree index dominates), is
# roughly break-even around 320 triangles, and wins with high statistical
# significance (p < 0.02, p < 1e-4 above ~5000 tris) from ~1280 triangles up.
# Log-log interpolation of the paired speedup curve puts the crossover at
# ~551 triangles; 600 is a round number just past it.
_TRIMESH_TRI_THRESHOLD = 600


def _inside_mask_numpy(verts, faces, points, direction=None, progress_cb=None):
    """Vectorized single-ray parity inside test.

    Casts one fixed-direction ray from every point and counts forward triangle
    crossings; odd => inside a closed mesh. ``faces`` must be triangles
    (N, 3). Returns a bool array of length ``len(points)``.
    """
    if direction is None:
        direction = _RAY_DIR
    d = np.asarray(direction, dtype=np.float64)
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    points = np.asarray(points, dtype=np.float64)

    n = len(points)
    inside = np.zeros(n, dtype=bool)
    if n == 0 or len(faces) == 0:
        return inside

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e1 = v1 - v0                       # (T, 3)
    e2 = v2 - v0                       # (T, 3)
    pvec = np.cross(d, e2)            # (T, 3)
    det = np.einsum('ij,ij->i', e1, pvec)   # (T,)
    eps = 1e-9
    parallel = np.abs(det) < eps
    inv_det = np.zeros_like(det)
    inv_det[~parallel] = 1.0 / det[~parallel]

    chunk = _numpy_chunk_size(len(faces))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        p = points[start:stop]                       # (P, 3)
        # tvec = p - v0 for every (point, triangle) pair -> (P, T, 3)
        tvec = p[:, None, :] - v0[None, :, :]
        u = np.einsum('ptj,tj->pt', tvec, pvec) * inv_det[None, :]
        qvec = np.cross(tvec, e1[None, :, :])        # (P, T, 3)
        v = np.einsum('j,ptj->pt', d, qvec) * inv_det[None, :]
        t = np.einsum('tj,ptj->pt', e2, qvec) * inv_det[None, :]
        hit = (~parallel[None, :] & (u >= -1e-7) & (v >= -1e-7)
               & (u + v <= 1.0 + 1e-7) & (t > 1e-7))
        inside[start:stop] = (hit.sum(axis=1) % 2) == 1
        if progress_cb is not None:
            progress_cb(stop)
    return inside


def _inside_mask_trimesh(verts, faces, points, progress_cb=None):
    """trimesh-based inside test, using rtree for the broad-phase.

    Both trimesh and rtree are bundled wheels (declared in
    blender_manifest.toml), not an optional system dependency -- this is what
    makes the extension self-contained per the Extensions Platform rules.
    We deliberately do NOT bundle embree/pyembree: it's a compiled, per-
    platform binary in the same size/complexity class as the Cu-Py wheel we
    already rejected for the CPU edition, and rtree alone is enough for
    mesh.contains() to work (trimesh raises ModuleNotFoundError without
    *some* acceleration structure). Still wrapped in try/except by the caller
    so a platform we didn't bundle a wheel for cleanly falls back to numpy
    instead of hard-failing.
    """
    import trimesh
    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                           faces=np.asarray(faces, dtype=np.int64),
                           process=False)
    inside = mesh.contains(np.asarray(points, dtype=np.float64))
    if progress_cb is not None:
        progress_cb(len(points))
    return np.asarray(inside, dtype=bool)


def inside_mask(verts, faces, points, direction=None, prefer_trimesh=True,
                progress_cb=None):
    """Inside test with automatic backend choice.

    Uses the bundled trimesh+rtree path once the mesh is heavy enough to
    justify their fixed overhead (see _TRIMESH_TRI_THRESHOLD, and
    paper/experiments/ for the benchmark behind that number); otherwise the
    dependency-free numpy path, which also serves as the safety net if a
    platform has no matching bundled wheel.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if prefer_trimesh and len(faces) >= _TRIMESH_TRI_THRESHOLD:
        try:
            return _inside_mask_trimesh(verts, faces, points,
                                        progress_cb=progress_cb)
        except Exception:
            pass  # no wheel for this platform, or trimesh failed -> numpy
    return _inside_mask_numpy(verts, faces, points, direction=direction,
                              progress_cb=progress_cb)


# ---------------------------------------------------------------------------
# Standalone subprocess entry point
# ---------------------------------------------------------------------------

def _run_job(job_path, result_path, progress_path):
    import pickle

    with open(job_path, 'rb') as fh:
        job = pickle.load(fh)

    direction = job.get('direction')
    # Generate the big point arrays here (in the subprocess) from the tiny grid
    # description, so the Blender main thread never builds or pickles them.
    g = job['grid']
    point_sets = {'centers': voxel_centers(g['dims'], g['origin'], g['vsize']),
                  'nodes': node_coords(g['dims'], g['origin'], g['vsize'])}
    queries = job['queries']

    total = sum(len(point_sets[q['target']]) for q in queries) or 1
    done_before = 0

    def _write_progress(frac):
        try:
            with open(progress_path, 'w') as pf:
                pf.write(f"{max(0.0, min(1.0, frac)):.6f}")
        except Exception:
            pass

    _write_progress(0.0)
    masks = []
    for q in queries:
        pts = point_sets[q['target']]
        base = done_before

        def _cb(done_in_query, _base=base):
            _write_progress((_base + done_in_query) / total)

        mask = inside_mask(q['verts'], q['faces'], pts, direction=direction,
                           progress_cb=_cb)
        masks.append(mask)
        done_before += len(pts)
        _write_progress(done_before / total)

    with open(result_path, 'wb') as fh:
        pickle.dump({'masks': masks, 'error': None}, fh)
    _write_progress(1.0)


def main(argv):
    import pickle
    job_path, result_path, progress_path = argv[1], argv[2], argv[3]
    try:
        _run_job(job_path, result_path, progress_path)
    except Exception as exc:  # surface to the parent via the result file
        import traceback
        try:
            with open(result_path, 'wb') as fh:
                pickle.dump({'masks': None,
                             'error': f"{exc}\n{traceback.format_exc()}"}, fh)
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
