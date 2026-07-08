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
                                     else across CPU worker processes if that
                                     is available and beneficial at this size,
                                     else falls back one rung at a time down
                                     to plain CPU.

GPU_DOF (single-GPU-vs-CPU) is a threshold, not a measurement -- it has not
been benchmarked, only reasoned about (kernel-launch/transfer overhead
dominates small problems). MULTI_GPU_DOF/MULTI_CPU_DOF specifically WERE
benchmarked, on real hardware (2x GTX 970, paper/experiments/
benchmark_multigpu.py), across two rounds so far:

Round 1: multi-GPU came back 1.5x-3.5x SLOWER than single-GPU at every size
from 91,875 DOF up to the largest tested (1,635,075 DOF), i.e. throughout
and beyond this threshold. Root cause: core/multigrid.py's V-cycle was
routing its smoothing passes through the pool too, not just the outer CG
loop's own matvec -- roughly 5 pooled calls per outer CG iteration instead
of ~2, each paying a full host round-trip on every device for work too
small to amortize that cost. Fixed by splitting MGSolver._apply (local,
smoothing-only) from _apply_pooled (outer CG only).

Round 2 (after the above fix + streams/pinned-memory in
parallel_gpu_domain.py): small problems improved a lot (~8x FASTER than
single-GPU at 1,911 DOF), but large problems got WORSE (~6.5x slower at
1,101,411 DOF, with measured GPU utilization *dropping* below 30%). Tried
fix (Round 3, since reverted): hypothesized that _apply's now-un-pooled
smoothing, still running on single-threaded host numpy, had become the
bottleneck, so MGSolver was changed to use Cu-Py instead of numpy for that
local work. Measured on the same hardware this made things WORSE AGAIN
(45.6s vs Round 2's 19.6s) -- disproving the hypothesis (Cu-Py's bincount,
a scatter-add over colliding FEA node indices, apparently doesn't
parallelize well via atomics on this hardware, and forcing Cu-Py also added
extra full-vector D2H/H2D copies around _apply_pooled that Round 2 didn't
pay). Reverted back to Round 2's numpy rule. See "multi_gpu xp history" in
core/multigrid.py's module docstring for the full blow-by-blow.

Round 2's 19.6x-slower number is therefore still the current best-known
state for large multi-GPU problems, and its root cause is NOT yet
identified -- two guesses in a row have been wrong. MGSolver.solve() now
prints a wall-clock breakdown (pooled vs. local V-cycle time) when
"Verbose solver log" is enabled; read that output before changing this
again rather than reasoning from aggregate timing alone. Until a Round 4
fix is verified, treat MULTI_GPU_DOF as unverified for large problems -- if
you re-run benchmark_multigpu.py and multi-GPU is still a net loss at your
sizes, raise this threshold (or drop multi-GPU from AUTO entirely and keep
it as an explicit compute_mode="MULTI_GPU" opt-in only).

Below GPU_DOF, a GPU is *not* used even if present and requested: kernel-
launch / host<->device transfer overhead dominates the actual compute at that
scale, so numpy on the CPU is faster in practice for the small grids typical
of the coarse SIMP levels. This module exists so that decision is made in one
place instead of being re-litigated at every call site.

MULTI_CPU uses the same DOF threshold as MULTI_GPU: multi-process CPU work
only pays for itself once there is enough element work per worker to amortize
the shared-memory hand-off/IPC cost every CG iteration; below that scale a
single process (with the BLAS thread pool sized to the machine) is both
simpler and faster.
"""

import os

from . import backend

GPU_DOF = 50_000
MULTI_GPU_DOF = 500_000
MULTI_CPU_DOF = 500_000

_THREADPOOL_LIMITER = None   # kept alive for the process lifetime


class ComputePlan:
    """Resolved decision for one FEA/multigrid instance."""

    __slots__ = ("kind", "xp", "n_workers", "label")

    def __init__(self, kind, xp, n_workers, label):
        self.kind = kind            # 'cpu' | 'gpu' | 'multi_gpu' | 'multi_cpu'
        self.xp = xp                # array module to use for the (single-
                                    # device) parts of the computation
        self.n_workers = n_workers  # devices (multi_gpu) or processes
                                    # (multi_cpu); 1 otherwise
        self.label = label          # human-readable, for the debug log

    @property
    def parallel(self):
        return self.kind in ("multi_gpu", "multi_cpu")


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
    / 'MULTI_CPU' (falls back gracefully -- towards CPU -- if the forced mode
    is not actually available, it is never allowed to silently do nothing).
    """
    mode = (mode or "AUTO").upper()
    n_workers_cpu = cpu_worker_count(cpu_threads)
    configure_cpu_threads(n_workers_cpu, verbose=verbose)

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

    def _multi_cpu():
        return ComputePlan("multi_cpu", backend.get_xp(False), n_workers_cpu,
                            f"multi-process CPU ({n_workers_cpu} workers)")

    if mode == "CPU":
        plan = _cpu()
    elif mode == "GPU":
        plan = _gpu() if gpu_ok else _cpu()
    elif mode == "MULTI_GPU":
        plan = _multi_gpu() if (gpu_ok and n_gpus > 1) else (_gpu() if gpu_ok else _cpu())
    elif mode == "MULTI_CPU":
        plan = _multi_cpu() if n_workers_cpu > 1 else _cpu()
    else:  # AUTO
        if ndof >= MULTI_GPU_DOF and gpu_ok and n_gpus > 1:
            plan = _multi_gpu()
        elif ndof >= MULTI_CPU_DOF and n_workers_cpu > 1 and not gpu_ok:
            # Only reach for multi-process CPU at large scale, and only when
            # there is no usable GPU -- a single GPU is very likely faster
            # than several CPU processes at this size anyway.
            plan = _multi_cpu()
        elif ndof >= GPU_DOF and gpu_ok:
            plan = _gpu()
        else:
            plan = _cpu()

    if verbose:
        print(f"[Blendtopo] compute plan: ndof={ndof} mode={mode} -> "
              f"{plan.label} (gpu_build={gpu_build}, gpu_ok={gpu_ok}, "
              f"n_gpus={n_gpus}, cpu_workers={n_workers_cpu})")
    return plan
