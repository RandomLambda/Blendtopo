# SPDX-License-Identifier: GPL-3.0-or-later
"""
Whole-level topology-optimization worker that runs in a separate process.

This is the heart of keeping Blender's UI responsive: *all* the heavy numeric
work for one coarse-to-fine level happens here, off Blender's main thread:

  1. inside tests (build space / exclusions / bearings / loads) -> grid masks
  2. grid assembly (active voxels, fixed DOFs, force vector)
  3. optional warm-start resampling from the previous (coarser) level
  4. the SIMP optimization loop itself (matrix-free FEA + multigrid + OC)
  5. meshing each iteration's density field into verts/faces (Surface Nets /
     blocky), so the main thread only has to hand finished geometry to Blender.

The worker streams results to a work directory: after every iteration it writes
``step_<it>.pkl`` (verts + faces + metrics) and updates ``status.json``; on
completion it writes ``final.pkl`` (final density + displacement for the next
level's warm start and the Continue cache). The parent (``SolverClient``) polls
those files from the modal timer and never blocks.

The module imports **only numpy + stdlib at top level**; it pulls the pure-numpy
core (``core.fea``/``simp``/``multigrid``/``extract``/``inside_worker``) lazily
inside the worker process, where there is no ``bpy``. ``SolverClient`` (used by
the Blender side) is likewise bpy-free, so importing this module into the add-on
is safe and cheap.

No threads, no multiprocessing-of-bpy: the child is a plain numpy script and
cannot touch (or crash) Blender state.
"""

import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

# EMA smoothing factor for the von Mises heatmap's color-scale ceiling (see
# _worker_main): weight on the *previous* smoothed value, so 0.7 means each
# new iteration only nudges the scale 30% of the way toward its own reading.
# Keeps the color scale from flickering step to step while still tracking
# real trends (e.g. the true max stress climbing as the shape refines).
_STRESS_EMA_ALPHA = 0.7
# Percentile (of active-element stress) used as the scale ceiling instead of
# the raw max, so the handful of singular elements that always sit at point
# loads/supports on a voxel grid don't wash out the rest of the heatmap.
_STRESS_PERCENTILE = 98.0


# ===========================================================================
# Parent side: launch + non-blocking poll (runs inside Blender, bpy-free)
# ===========================================================================

class SolverClient:
    """Launch the worker for one level and poll it without blocking.

    Events from :meth:`poll`:
      ('voxel', frac)            - still building the grid (0..1)
      ('step', dict)             - a new iteration is ready (it, compliance,
                                   change, verts, faces, and - if the job asked
                                   for a stress heatmap - stress3d/stress_vmax/
                                   stress_origin/stress_vsize); build a preview
                                   object
      ('done', dict)             - finished; dict has rho3d/u/dims/origin/vsize
                                   for the warm-start / Continue cache
      ('error', message)         - the worker failed
      ('running', frac)          - working, nothing new to show yet
    """

    def __init__(self):
        self._proc = None
        self._dir = None
        self._logf = None
        self._consumed = 0          # highest iteration index already returned
        self._finished = False

    def start(self, job):
        """Pickle the (small) job and spawn the worker process."""
        self._dir = tempfile.mkdtemp(prefix="blendtopo_solve_")
        job_path = os.path.join(self._dir, "job.pkl")
        with open(job_path, "wb") as fh:
            pickle.dump(job, fh)

        cmd = [sys.executable, os.path.abspath(__file__), job_path, self._dir]
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000   # CREATE_NO_WINDOW
        # The worker (below) imports sibling modules as `core.simp` etc, which
        # only resolves if `blendtopo/` (this file's grandparent) is on the
        # *child's* sys.path. We hand that to the subprocess via PYTHONPATH in
        # its own environment rather than mutating sys.path from inside the
        # worker: Blender's own process/sys.path is never touched either way.
        env = dict(os.environ)
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (pkg_parent, env.get("PYTHONPATH", "")) if p)
        self._logf = open(os.path.join(self._dir, "worker.log"), "wb")
        self._proc = subprocess.Popen(
            cmd, stdout=self._logf, stderr=self._logf, env=env, **kwargs)

    # -- file helpers --------------------------------------------------------

    def _status(self):
        try:
            with open(os.path.join(self._dir, "status.json")) as fh:
                return json.load(fh)
        except Exception:
            return None

    def _load(self, name):
        with open(os.path.join(self._dir, name), "rb") as fh:
            return pickle.load(fh)

    def _log_tail(self):
        try:
            with open(os.path.join(self._dir, "worker.log"), "rb") as fh:
                return fh.read().decode("utf-8", "replace").strip()
        except Exception:
            return ""

    # -- polling -------------------------------------------------------------

    def poll(self):
        if self._finished:
            return ("running", 1.0)

        st = self._status()
        if st is None:
            # No status yet. Only an error if the process already died.
            if self._proc is not None and self._proc.poll() is not None:
                return ("error", self._log_tail() or "worker exited early")
            return ("running", 0.0)

        phase = st.get("phase")
        if phase == "error":
            return ("error", st.get("error") or self._log_tail() or "worker error")

        if phase == "voxel":
            return ("voxel", float(st.get("frac", 0.0)))

        # phase in ('solve', 'done'): consume the next iteration in order so each
        # timer tick builds at most one preview mesh (keeps ticks light).
        nxt = self._consumed + 1
        last = int(st.get("last_step", 0))
        step_file = os.path.join(self._dir, f"step_{nxt}.pkl")
        if nxt <= last and os.path.exists(step_file):
            try:
                data = self._load(f"step_{nxt}.pkl")
            except (EOFError, pickle.UnpicklingError):
                return ("running", float(st.get("frac", 0.0)))   # mid-write
            try:
                os.remove(step_file)
            except OSError:
                pass
            self._consumed = nxt
            return ("step", data)

        if phase == "done" and self._consumed >= last:
            try:
                final = self._load("final.pkl")
            except Exception:
                final = {}
            self._finished = True
            return ("done", final)

        return ("running", float(st.get("frac", 0.0)))

    def request_stop(self):
        """Ask the worker to stop after the current iteration (keeps latest)."""
        if self._dir:
            try:
                open(os.path.join(self._dir, "stop"), "w").close()
            except OSError:
                pass

    def cancel(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self.cleanup()

    def cleanup(self):
        if self._logf is not None:
            try:
                self._logf.close()
            except Exception:
                pass
            self._logf = None
        if self._dir and os.path.isdir(self._dir):
            shutil.rmtree(self._dir, ignore_errors=True)
        self._dir = None


def solver_available():
    """Whether the out-of-process solver can be launched."""
    try:
        return bool(sys.executable) and os.path.exists(os.path.abspath(__file__))
    except Exception:
        return False


# ===========================================================================
# Child side: the actual worker (runs in the subprocess, no bpy)
# ===========================================================================

def _replace_with_retry(tmp, path, attempts=10, delay=0.05):
    """``os.replace`` that tolerates transient WinError 5/32.

    On Windows, ``MoveFileEx`` (what ``os.replace`` uses under the hood) can
    briefly fail with ``PermissionError`` if another process (the parent's
    polling ``open()``, an AV scanner, the search indexer, ...) has the
    destination file open at that exact instant -- even though that reader
    never blocks a POSIX rename. The reader side already tolerates a missing
    or half-written file (see ``SolverClient._status``), so the fix here is
    just to give the writer a few short retries instead of crashing the
    whole worker process over a few-millisecond race.
    """
    last_exc = None
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(delay * (i + 1))
    # Out of retries: clean up the temp file if it's still there and re-raise
    # so the caller's normal error handling (status.json "error" phase / worker
    # log) still kicks in.
    raise last_exc


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    _replace_with_retry(tmp, path)


def _atomic_write_pickle(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    _replace_with_retry(tmp, path)


def _worker_main(job_path, work_dir):
    # Pull in the pure-numpy core (this is the subprocess: no bpy, so these
    # imports are safe and light). `core` resolves as a package because the
    # parent process put `blendtopo/` on *this* process's PYTHONPATH before
    # launching us (see SolverClient.start) -- no sys.path mutation here.
    import numpy as np
    from core import inside_worker, extract
    from core.simp import Problem, resample_density, resample_displacement

    status_path = os.path.join(work_dir, "status.json")
    stop_path = os.path.join(work_dir, "stop")

    def set_status(**kw):
        _atomic_write_json(status_path, kw)

    set_status(phase="voxel", frac=0.0, last_step=0)

    with open(job_path, "rb") as fh:
        job = pickle.load(fh)

    g = job["grid"]
    dims = tuple(g["dims"])
    origin = np.asarray(g["origin"], dtype=float)
    vsize = float(g["vsize"])
    nx, ny, nz = dims
    direction = job.get("direction")

    centers = inside_worker.voxel_centers(dims, origin, vsize)
    nodes = inside_worker.node_coords(dims, origin, vsize)
    point_sets = {"centers": centers, "nodes": nodes}

    # --- inside tests (voxelization), with progress -----------------------
    queries = job["queries"]
    descr = job["descr"]
    total_pts = sum(len(point_sets[q["target"]]) for q in queries) or 1
    done_pts = [0]
    masks = []
    for q in queries:
        pts = point_sets[q["target"]]
        base = done_pts[0]

        def _cb(done_in_query, _base=base):
            set_status(phase="voxel", frac=(_base + done_in_query) / total_pts,
                       last_step=0)

        m = inside_worker.inside_mask(q["verts"], q["faces"], pts,
                                      direction=direction, progress_cb=_cb)
        masks.append(np.asarray(m, dtype=bool))
        done_pts[0] += len(pts)
        set_status(phase="voxel", frac=done_pts[0] / total_pts, last_step=0)

    # --- bearing fallback: snap to nearest node instead of erroring out ---
    # At coarse (early-refinement) resolutions a small bearing mesh can miss
    # every grid node in its inside-test. Rather than aborting the whole run,
    # pin the single nearest node so the level still solves; once the grid
    # is fine enough the normal inside-test takes over again and the bearing
    # is represented properly.
    for i, (q, (role, meta)) in enumerate(zip(queries, descr)):
        if role != "bearing" or masks[i].any():
            continue
        pts = point_sets[q["target"]]
        verts = np.asarray(q["verts"], dtype=float)
        if len(verts) == 0 or len(pts) == 0:
            continue
        centroid = verts.mean(axis=0)
        d2 = np.sum((pts - centroid) ** 2, axis=1)
        idx = int(np.argmin(d2))
        fallback_mask = np.zeros(len(pts), dtype=bool)
        fallback_mask[idx] = True
        masks[i] = fallback_mask
        print(f"[Blendtopo] bearing '{q.get('name', '?')}' covered no grid "
              f"nodes at this resolution - snapped to nearest node instead.")

    # --- assemble grid ----------------------------------------------------
    inside = None
    excl = np.zeros(nx * ny * nz, dtype=bool)
    keepin = np.zeros(nx * ny * nz, dtype=bool)
    fixed = []
    ndof = 3 * (nx + 1) * (ny + 1) * (nz + 1)
    force = np.zeros(ndof)
    for (role, meta), mask in zip(descr, masks):
        if role == "build":
            inside = mask
        elif role == "exclude":
            excl = excl | mask
        elif role == "include":
            keepin = keepin | mask
        elif role == "bearing":
            fix_x, fix_y, fix_z = meta
            for n in np.where(mask)[0]:
                if fix_x:
                    fixed.append(3 * n)
                if fix_y:
                    fixed.append(3 * n + 1)
                if fix_z:
                    fixed.append(3 * n + 2)
        elif role == "load":
            nin = np.where(mask)[0]
            if len(nin):
                fv = np.asarray(meta, dtype=float) / len(nin)
                for n in nin:
                    force[3 * n:3 * n + 3] += fv
    if inside is None:
        inside = np.zeros(nx * ny * nz, dtype=bool)
    # Keep-out always wins over keep-in if the two overlap.
    keepin = keepin & ~excl
    active = (inside | keepin) & ~excl
    fixed_dofs = (np.unique(np.asarray(fixed, dtype=np.int64))
                  if fixed else np.zeros(0, dtype=np.int64))

    if int(active.sum()) == 0:
        set_status(phase="error", error="No active voxels - check build space / resolution")
        return
    if fixed_dofs.size == 0:
        set_status(phase="error", error="Bearings cover no grid nodes at this resolution")
        return

    # --- warm start from a coarser level ----------------------------------
    sp = job["simp"]
    warm = job.get("warm") or {}
    x_init = None
    u_init = None
    if warm.get("prev_rho3d") is not None:
        x_init = resample_density(np.asarray(warm["prev_rho3d"]),
                                  np.asarray(warm["prev_origin"], dtype=float),
                                  float(warm["prev_vsize"]), centers)
    if warm.get("prev_u") is not None:
        cand = resample_displacement(
            np.asarray(warm["prev_u"]),
            np.asarray(warm["prev_origin"], dtype=float),
            float(warm["prev_vsize"]), tuple(warm["prev_node_dims"]), nodes)
        if np.all(np.isfinite(cand)):
            u_init = cand

    # --- SIMP solve, streaming a mesh per iteration -----------------------
    active3d = active.reshape(nx, ny, nz)
    # Keep-in voxels are forced to density=1 for meshing/display only - never
    # fed back into the optimizer's own state (x_init / warm-start / FEA),
    # so the solve itself is unaffected and always sees the real, optimized
    # density. This is a display-time union: the visible shape always
    # includes the keep-in volume and merges smoothly with it, while the
    # optimization is free to make the rest of the design lighter to
    # compensate.
    keepin3d = keepin.reshape(nx, ny, nz) if keepin.any() else None
    prob = Problem(nx, ny, nz, active3d, fixed_dofs, force,
                   volfrac=sp["volfrac"], penalty=sp["penalty"],
                   rmin=sp["rmin"], nu=sp["nu"], e0=sp["e0"],
                   use_multigrid=sp["use_multigrid"],
                   compute_mode=sp.get("compute_mode", "AUTO"),
                   cpu_threads=sp.get("cpu_threads", 0),
                   verbose=sp.get("verbose", False))

    iso = float(job["iso"])
    style = job.get("style", "SMOOTH")
    want_stress = bool(job.get("stress", False))
    stress_vmax_smooth = [None]   # EMA state, carried across this level's iterations

    def mesh_of(rho3d):
        if style == "BLOCKY":
            verts, faces = extract.cubes_from_density(rho3d, iso, origin, vsize)
        else:
            verts, faces = extract.surface_nets(rho3d, iso, origin, vsize)
        verts = np.asarray(verts, dtype=np.float32)
        faces = (np.asarray(faces, dtype=np.int32)
                 if len(faces) else np.zeros((0, 4), dtype=np.int32))
        return verts, faces

    last_it = 0
    last_rho3d = None
    try:
        for it, comp, change, rho in prob.optimize(
                max_iter=sp["max_iter"], x_init=x_init, tol=sp["tol"],
                u_init=u_init):
            rho3d = rho.reshape(nx, ny, nz)
            last_rho3d = rho3d
            mesh_rho3d = np.where(keepin3d, 1.0, rho3d) if keepin3d is not None else rho3d
            verts, faces = mesh_of(mesh_rho3d)
            step_payload = {"it": int(it), "compliance": float(comp),
                            "change": float(change), "verts": verts, "faces": faces}

            if want_stress:
                vm = prob.element_von_mises(rho)
                active_vals = vm[active]
                raw_vmax = (float(np.percentile(active_vals, _STRESS_PERCENTILE))
                           if active_vals.size else 0.0)
                raw_vmax = max(raw_vmax, 1e-12)
                if stress_vmax_smooth[0] is None:
                    stress_vmax_smooth[0] = raw_vmax
                else:
                    stress_vmax_smooth[0] = (
                        _STRESS_EMA_ALPHA * stress_vmax_smooth[0]
                        + (1.0 - _STRESS_EMA_ALPHA) * raw_vmax)
                step_payload["stress3d"] = vm.reshape(nx, ny, nz).astype(np.float32)
                step_payload["stress_vmax"] = float(stress_vmax_smooth[0])
                step_payload["stress_origin"] = np.asarray(origin, dtype=float)
                step_payload["stress_vsize"] = vsize
                # Every active element, not a decimated sample - a
                # cross-section should never be missing material that's
                # actually there.
                step_payload["stress_active3d"] = active.reshape(nx, ny, nz)
                # Reliability weight for the *surface* heatmap's vertex
                # sampling: SIMP density IS the element's fill fraction, so
                # a mostly-void border element (cut obliquely by the iso-
                # surface) is down-weighted instead of dragging the
                # trilinear average toward its near-zero stress estimate -
                # this is what removes the false "blue ring" banding along
                # slanted faces (see extract._sample_field_trilinear_weighted).
                step_payload["stress_weight3d"] = (
                    rho3d * active3d).astype(np.float32)

            _atomic_write_pickle(
                os.path.join(work_dir, f"step_{it}.pkl"), step_payload)
            last_it = it
            set_status(phase="solve", frac=it / max(1, sp["max_iter"]),
                       last_step=int(it))
            if os.path.exists(stop_path):
                break
    finally:
        # Always tear down any multi-CPU/multi-GPU pools before this process
        # exits, so no worker process/GPU context is left dangling if the
        # loop above raised.
        prob.close()

    # --- final payload for warm-start / Continue cache --------------------
    final = {"dims": dims, "origin": np.asarray(origin, dtype=float),
             "vsize": vsize, "last_step": int(last_it)}
    final["rho3d"] = (np.asarray(last_rho3d, dtype=np.float32)
                      if last_rho3d is not None else None)
    final["u"] = (np.asarray(prob.last_u, dtype=np.float64)
                  if prob.last_u is not None else None)
    final["node_dims"] = (nx + 1, ny + 1, nz + 1)
    _atomic_write_pickle(os.path.join(work_dir, "final.pkl"), final)
    set_status(phase="done", frac=1.0, last_step=int(last_it))


def _main(argv):
    job_path, work_dir = argv[1], argv[2]
    try:
        _worker_main(job_path, work_dir)
    except Exception as exc:
        import traceback
        try:
            _atomic_write_json(
                os.path.join(work_dir, "status.json"),
                {"phase": "error",
                 "error": f"{exc}\n{traceback.format_exc()}",
                 "last_step": 0})
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
