# SPDX-License-Identifier: GPL-3.0-or-later
"""
Compute-backend selection: CPU / single GPU / multi-GPU / multi-process CPU.

This module is the single place that decides, for a given problem size (the
FEA's reduced DOF count), which array module and which parallelism to use. It
is identical in both editions (CPU-only and GPU); it only *asks* `backend`
what is available (``is_gpu_build``, ``gpu_usable``, ``gpu_device_count``) and
never imports Cu-Py itself, so it stays import-safe in the hosted CPU
edition.

Thresholds:

  ndof <  GPU_DOF                -> plain CPU (numpy), all available cores.
  GPU_DOF <= ndof < MULTI_GPU_DOF -> single GPU if the GPU edition has one
                                     that passed its self-test, else CPU.
  ndof >= MULTI_GPU_DOF           -> split the matrix-free matvec across every
                                     visible GPU (GPU edition, >1 device),
                                     else falls back to single GPU, then CPU.

GPU_DOF was benchmarked (paper/experiments/pipeline_timing_results_v2.csv):
CPU wins below ~91,875 DOF, GPU wins above; the crossover is set to 50,000,
known only to within that gap.

MULTI_GPU_DOF is set far above any benchmarked size: pooling the outer CG
matvec across 2 GPUs was measured at best a noise-level parity and at worst
a real loss at larger sizes, because only that one matvec is pooled per
iteration while V-cycle smoothing stays single-device -- the pooling
overhead is roughly fixed per call but doesn't shrink as the problem grows.
compute_mode="MULTI_GPU" still works as an explicit, forced choice. See
paper/experiments/MULTIPROCESSING_FINDINGS.md and multigrid.py's module
docstring for the measurements behind this.

Below GPU_DOF, a GPU is not used even if present: kernel-launch / transfer
overhead dominates at that scale, so CPU numpy is faster for the small
grids typical of the coarse SIMP levels.

multi_cpu is not de-prioritized, it is unreachable: the pooled outer-CG
matvec is only ~10% of MGSolver's per-iteration cost, so Amdahl's law caps
the best possible speedup at ~1.11x regardless of core count, while adding
a 2.5-3s pool-spawn cost paid every time Problem()/MGSolver is rebuilt --
untenable for this addon's "remesh often, refine successively" design. As
of v1.1.19 there is no AUTO threshold or forced compute_mode="MULTI_CPU"
path; choose() maps it to CPU. See
paper/experiments/MULTIPROCESSING_FINDINGS.md for the measurements.
"""

import os

from . import backend

GPU_DOF = 50_000
# Set far above the tested range so AUTO never picks multi_gpu until a real
# measured win justifies lowering this. compute_mode="MULTI_GPU" remains
# available as an explicit, forced choice regardless of this threshold (see
# multigrid.py's _init_pool).
MULTI_GPU_DOF = 50_000_000
# No MULTI_CPU_DOF / AUTO branch / forced compute_mode="MULTI_CPU" path
# exists any more -- the Amdahl ceiling makes multi_cpu a loss-or-noise
# result at every size tested. choose() maps mode="MULTI_CPU" to CPU below.

# Threshold for MGSolver to route V-cycle smoothing (not just the outer CG
# matvec) through a CPU pool too -- see MULTIPROCESSING_FINDINGS.md Section
# 5. Conservative: real-hardware measurement found a loss below ~700k DOF
# and a win above ~1.1M DOF at the default sweep count; this sits in the
# untested gap, closer to the confirmed win.
POOLED_SMOOTH_CPU_DOF = 1_000_000

_THREADPOOL_LIMITER = None   # kept alive for the process lifetime


class ComputePlan:
    """Resolved decision for one FEA/multigrid instance."""

    __slots__ = ("kind", "xp", "n_workers", "label")

    def __init__(self, kind, xp, n_workers, label):
        self.kind = kind            # 'cpu' | 'gpu' | 'multi_gpu'
                                    # ('multi_cpu' is no longer produced by
                                    # choose() -- see module docstring)
        self.xp = xp                # array module to use for the (single-
                                    # device) parts of the computation
        self.n_workers = n_workers  # devices (multi_gpu); 1 otherwise
        self.label = label          # human-readable, for the debug log

    @property
    def parallel(self):
        return self.kind == "multi_gpu"


def cpu_worker_count(requested=0):
    """How many CPU worker processes/threads to use.

    requested <= 0 means "auto": all logical cores minus one, so the UI
    thread (Blender, or on this path the solver subprocess's own main
    thread) is never starved. Always at least 1.
    """
    total = os.cpu_count() or 1
    if requested and requested > 0:
        return max(1, min(int(requested), total))
    return max(1, total - 1)


def configure_cpu_threads(n_threads, verbose=False):
    """Limit the BLAS thread pool (openblas/mkl/...) numpy is linked
    against to ``n_threads``. Uses threadpoolctl (bundled wheel), which
    patches the already-loaded BLAS library at runtime -- this works even
    though numpy was imported long before this call (by Blender itself).

    Safe no-op if threadpoolctl is unavailable for any reason: numpy just
    keeps whatever thread count the BLAS library defaulted to.
    """
    global _THREADPOOL_LIMITER
    try:
        import threadpoolctl
    except Exception as exc:
        if verbose:
            print(f"[Blendtopo] threadpoolctl unavailable ({exc}); "
                  f"leaving BLAS thread count at its default")
        return None
    try:
        # Replaces any previous limiter (only one is meant to be active).
        if _THREADPOOL_LIMITER is not None:
            _THREADPOOL_LIMITER.unregister()
        limiter = threadpoolctl.threadpool_limits(limits=int(n_threads))
        _THREADPOOL_LIMITER = limiter
        if verbose:
            info = threadpoolctl.threadpool_info()
            libs = ", ".join(sorted({d.get("internal_api", "?") for d in info})) or "none detected"
            print(f"[Blendtopo] BLAS thread pool set to {n_threads} "
                  f"(libraries: {libs})")
        return limiter
    except Exception as exc:
        if verbose:
            print(f"[Blendtopo] could not set BLAS thread count ({exc})")
        return None


def choose(ndof, mode="AUTO", cpu_threads=0, verbose=False):
    """Resolve a ComputePlan for a problem with ``ndof`` (reduced) DOFs.

    mode: 'AUTO' (thresholds above), or a forced 'CPU' / 'GPU' / 'MULTI_GPU'
    (falls back gracefully -- towards CPU -- if the forced mode is not
    actually available, it is never allowed to silently do nothing).
    'MULTI_CPU' is accepted for backward compatibility but always maps to
    CPU -- see module docstring for why multi_cpu was removed outright.
    """
    mode = (mode or "AUTO").upper()
    n_workers_cpu = cpu_worker_count(cpu_threads)

    gpu_build = backend.is_gpu_build()
    gpu_ok = gpu_build and backend.gpu_usable()
    n_gpus = backend.gpu_device_count() if gpu_ok else 0

    def _cpu():
        return ComputePlan("cpu", backend.get_xp(False), 1,
                            f"CPU (numpy, {n_workers_cpu} BLAS threads)")

    def _gpu():
        return ComputePlan("gpu", backend.get_xp(True), 1, "single GPU (Cu-Py)")

    def _multi_gpu():
        return ComputePlan("multi_gpu", backend.get_xp(True), n_gpus,
                            f"multi-GPU ({n_gpus} devices, Cu-Py)")

    if mode == "CPU":
        plan = _cpu()
    elif mode == "GPU":
        plan = _gpu() if gpu_ok else _cpu()
    elif mode == "MULTI_GPU":
        plan = _multi_gpu() if (gpu_ok and n_gpus > 1) else (_gpu() if gpu_ok else _cpu())
    else:  # AUTO, and MULTI_CPU (no longer selectable -- see module docstring)
        if ndof >= MULTI_GPU_DOF and gpu_ok and n_gpus > 1:
            plan = _multi_gpu()
        elif ndof >= GPU_DOF and gpu_ok:
            plan = _gpu()
        else:
            plan = _cpu()

    # Main-process BLAS thread count is set AFTER the plan is known, not
    # before: it depends on whether a worker pool is also about to exist.
    # For "cpu"/"gpu" plans, this process does all the real matvec work
    # itself, so it gets the full n_workers_cpu BLAS threads. "multi_gpu"
    # leaves the heavy lifting to the pooled devices, not host BLAS, so it
    # is capped to 1 thread here to avoid oversubscribing the same cores
    # its own O(ndof) leftover work still touches.
    main_process_threads = (1 if plan.kind == "multi_gpu"
                             else n_workers_cpu)
    configure_cpu_threads(main_process_threads, verbose=verbose)

    if verbose:
        print(f"[Blendtopo] compute plan: ndof={ndof} mode={mode} -> "
              f"{plan.label} (gpu_build={gpu_build}, gpu_ok={gpu_ok}, "
              f"n_gpus={n_gpus}, cpu_workers={n_workers_cpu})")
    return plan
