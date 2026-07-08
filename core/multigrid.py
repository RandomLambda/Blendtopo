# SPDX-License-Identifier: GPL-3.0-or-later
"""
Geometric multigrid preconditioner for the voxel FEA.

The conjugate-gradient solver in fea.py uses a Jacobi (diagonal) preconditioner,
whose iteration count grows with grid size. This module provides a matrix-free
geometric-multigrid V-cycle preconditioner instead: a hierarchy of 2x-coarser
grids with weighted-Jacobi smoothing, trilinear prolongation/restriction and
rediscretized coarse operators. CG preconditioned by one V-cycle converges in a
near-constant number of iterations regardless of resolution.

Correctness note: the outer PCG always applies the TRUE fine operator for its
matvec and residual, so the solution is exact regardless of preconditioner
quality - multigrid only changes how many iterations are needed. If the
hierarchy cannot be built (odd/small dims) it degrades to single-level
Jacobi-PCG, identical to fea.solve.

Works on the full regular grid (void elements carry e_min stiffness). Pure
numpy, no external dependencies.

multi_gpu xp history (real-hardware benchmark timeline, 2x GTX 970, see
paper/experiments/benchmark_multigpu.py) -- kept here because the reasoning
that turned out to be WRONG is exactly the kind of thing worth not
re-discovering the hard way a third time:

  Round 1: self.xp was numpy for every "parallel" plan (multi_gpu/multi_cpu),
  copying fea.VoxelFEA's rule, AND the V-cycle smoother was routed through
  the pool along with the outer CG matvec. Multi-GPU was 1.5x-3.5x SLOWER
  than single-GPU. Root cause: routing smoothing through the pool (~5 pooled
  calls/iteration instead of ~2), each paying a full host round-trip per
  device for work too small to amortize. Fix: split MGSolver._apply (local,
  never pooled) from _apply_pooled (outer CG only, pooled).

  Round 2 (after the Round 1 fix + streams/pinned-memory in
  parallel_gpu_domain.py): self.xp still numpy. Small problems got much
  faster (~8x vs single-GPU), but large problems got WORSE (~6.5x slower,
  1,101,411 DOF, GPU utilization measured <30%). Hypothesis at the time:
  the now-un-pooled local smoothing, forced onto single-threaded host numpy,
  had become the bottleneck -- so self.xp was changed to
  backend.get_xp(True) (Cu-Py, single device) for multi_gpu, keeping only
  _apply_pooled's host round-trip to reach the multi-device pool.

  Round 3 (testing the Round 2 hypothesis): multi-GPU got WORSE AGAIN
  (45.6s vs 19.6s at the same 1.1M-DOF case) -- disproving the hypothesis.
  Two likely compounding causes: (1) Cu-Py's bincount is a scatter-add with
  colliding indices (shared FEA nodes between elements), which leans on
  atomics that apparently do not parallelize well on this hardware -- host
  numpy's bincount was actually faster per call, not slower; (2) forcing
  xp=Cu-Py added two extra full-vector D2H/H2D copies per _apply_pooled call
  (backend.asnumpy(u) / xp.asarray(...) around the pool -- a no-op when xp
  is already numpy, real transfers when it isn't) that Round 2 did not pay.
  Reverted self.xp back to numpy for every parallel plan (Round 2's rule).

  Status: Round 2's number (19.6s vs 3.0s single-GPU, ~6.5x slower) is the
  current best-known state for large multi-GPU problems, and its own root
  cause is still NOT identified -- "obviously it must be X" has now been
  wrong twice in a row for this file. MGSolver.solve() gained wall-clock
  instrumentation (self._t_pooled / self._t_local, printed when verbose)
  instead of a third guess; run benchmark_multigpu.py with "Verbose solver
  log" enabled and read the printed breakdown before changing this again.
"""

import time

import numpy as np

from .fea import _hex8_KE, build_edof
from . import backend
from . import compute_plan


def _node_dims(nx, ny, nz):
    return (nx + 1, ny + 1, nz + 1)


def _prolong_op(fine_dims, coarse_dims, xp=np):
    """Trilinear prolongation: fine node <- 8 coarse nodes (idx, weights).

    Coarse node j sits on fine node 2j, so a fine node at integer index i
    interpolates from the coarse lattice at i/2. Returns (idx[Nf,8] int,
    w[Nf,8] float) with coarse node-ids (x fastest).
    """
    Lfx, Lfy, Lfz = fine_dims
    Lcx, Lcy, Lcz = coarse_dims
    Nf = Lfx * Lfy * Lfz
    n = np.arange(Nf)
    ix = n % Lfx
    iy = (n // Lfx) % Lfy
    iz = n // (Lfx * Lfy)
    cc = np.stack([ix, iy, iz], axis=1) / 2.0          # coarse coords (float)
    base = np.floor(cc).astype(np.int64)
    fr = cc - base

    idx = np.empty((Nf, 8), dtype=np.int64)
    w = np.empty((Nf, 8), dtype=float)
    corners = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
               (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    for k, (ox, oy, oz) in enumerate(corners):
        jx = np.clip(base[:, 0] + ox, 0, Lcx - 1)
        jy = np.clip(base[:, 1] + oy, 0, Lcy - 1)
        jz = np.clip(base[:, 2] + oz, 0, Lcz - 1)
        wx = fr[:, 0] if ox else (1.0 - fr[:, 0])
        wy = fr[:, 1] if oy else (1.0 - fr[:, 1])
        wz = fr[:, 2] if oz else (1.0 - fr[:, 2])
        idx[:, k] = jx + Lcx * (jy + Lcy * jz)
        w[:, k] = wx * wy * wz
    return xp.asarray(idx), xp.asarray(w)


class MGSolver:
    """Multigrid-preconditioned CG on the full voxel grid."""

    def __init__(self, nx, ny, nz, fixed_dofs, nu=0.3, xp=None,
                 n_smooth=2, omega=0.6, min_elems=4, max_levels=6,
                 compute_mode="AUTO", cpu_threads=0, verbose=False):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.nelem = nx * ny * nz
        self.ndof = 3 * (nx + 1) * (ny + 1) * (nz + 1)
        self.n_smooth = n_smooth
        self.omega = omega
        self.verbose = verbose
        self._pool = None

        # Wall-clock accounting, reset at the start of every solve() call and
        # printed at the end when verbose -- added after two rounds of real-
        # hardware multi-GPU benchmarking produced numbers that contradicted
        # each other's implied diagnosis (see "multi_gpu xp history" in the
        # module docstring): guessing which part of the V-cycle is actually
        # slow from aggregate solve_iter timing alone was not reliable enough
        # to keep iterating blind. These are plain wall-clock sums (not
        # profiler-grade -- no exclusion of Python overhead between calls),
        # but they are enough to tell whether time is going into the pooled
        # multi-device matvec (_t_pooled) or the un-pooled local V-cycle work
        # that runs on every level (_t_local, covering _apply/_smooth plus
        # _restrict/_prolongate).
        self._t_pooled = 0.0
        self._t_local = 0.0
        self._n_pooled_calls = 0
        self._n_local_calls = 0

        # Same threshold-based plan as fea.VoxelFEA, evaluated on the full
        # (level-0) grid's DOF count. Only level 0 -- the one actually
        # touched every V-cycle smoothing pass -- is ever parallelized;
        # coarser levels are tiny by construction (built by halving until
        # min_elems) so a pool there would be pure overhead.
        if xp is not None:
            # Caller pinned an array module explicitly (e.g. tests): honour
            # it and skip the auto plan/pool entirely.
            self.xp = xp
            self.plan = None
        else:
            self.plan = compute_plan.choose(self.ndof, mode=compute_mode,
                                            cpu_threads=cpu_threads,
                                            verbose=verbose)
            # REVERTED (see "multi_gpu xp history" below): a prior version of
            # this file forced self.xp = backend.get_xp(True) (Cu-Py) here,
            # reasoning that host numpy was the bottleneck for the un-pooled
            # local V-cycle work. Measured on real hardware that made things
            # WORSE (45.6s vs 19.6s at 1.1M DOF) -- so that reasoning was
            # wrong: Cu-Py's bincount (scatter-add with colliding indices,
            # exactly what FEA assembly is) leans on atomics that don't
            # parallelize well on this hardware, and forcing xp=Cu-Py also
            # added two extra full-vector D2H/H2D copies per _apply_pooled
            # call (backend.asnumpy(u) / xp.asarray(...) around the pool,
            # which is a no-op when xp is already numpy) that weren't there
            # before. Back to plain numpy for ALL parallel plans, matching
            # fea.VoxelFEA -- see "multi_gpu xp history" below for the still-
            # unexplained 19.6s-vs-3.0s gap this does NOT fix.
            self.xp = np if self.plan.parallel else self.plan.xp
        xp = self.xp

        KE = _hex8_KE(E=1.0, nu=nu)
        self.KE = xp.asarray(KE)
        self.diagKE = xp.asarray(np.diag(KE))
        self._KE_host = KE

        # Build the grid hierarchy (each level 2x coarser) while dims stay even.
        self.levels = []
        dims = [(nx, ny, nz)]
        while (len(dims) < max_levels and dims[-1][0] % 2 == 0
               and dims[-1][1] % 2 == 0 and dims[-1][2] % 2 == 0
               and min(dims[-1]) // 2 >= min_elems // 2 and min(dims[-1]) >= 2):
            cx, cy, cz = dims[-1]
            dims.append((cx // 2, cy // 2, cz // 2))

        for lx, ly, lz in dims:
            edof = xp.asarray(build_edof(lx, ly, lz))
            self.levels.append({
                'dims': (lx, ly, lz),
                'ndof': 3 * (lx + 1) * (ly + 1) * (lz + 1),
                'edof': edof,
                'g': 1.0,            # geometric stiffness scale (filled below)
                'Evec': None,
                'Minv': None,
                'free': None,
            })
        for li, lv in enumerate(self.levels):
            lv['g'] = float(2 ** li)   # 3D element stiffness ~ h = 2^level

        # Prolongation operators between consecutive levels (fine<-coarse).
        self.prolong = []
        for li in range(len(self.levels) - 1):
            fdims = _node_dims(*self.levels[li]['dims'])
            cdims = _node_dims(*self.levels[li + 1]['dims'])
            self.prolong.append(_prolong_op(fdims, cdims, xp))

        self._build_free(fixed_dofs)

        if self.plan is not None:
            self._init_pool()

    def _init_pool(self):
        """Build the level-0 matvec pool if the plan calls for one, falling
        back one rung if it fails to construct (never lets a broken
        accelerator abort the solve -- see fea.VoxelFEA._init_pool, same
        pattern)."""
        plan = self.plan
        lv0 = self.levels[0]
        edof0_host = backend.asnumpy(lv0['edof'])
        if plan.kind == "multi_cpu":
            from . import parallel_cpu
            try:
                self._pool = parallel_cpu.CPUMatVecPool(
                    edof0_host, self._KE_host, lv0['ndof'], plan.n_workers,
                    verbose=self.verbose)
                return
            except Exception as exc:
                if self.verbose:
                    print(f"[Blendtopo] MG: multi-CPU pool unavailable "
                          f"({exc}); using single-process CPU")
                self.plan = compute_plan.ComputePlan(
                    "cpu", np, 1, "CPU (multi-CPU fallback)")
        elif plan.kind == "multi_gpu":
            # Domain-decomposed pool first (core/parallel_gpu_domain.py, a
            # separate opt-in file -- see its module docstring): only every
            # device's own DOF slice is ever transferred, instead of the
            # full vector to every device. Only valid for the FULL regular
            # grid (level 0 always is), so it always applies here. Falls
            # back to the plain broadcast pool, then single GPU/CPU, exactly
            # like the multi-CPU branch above -- a broken/unavailable
            # accelerator must never abort the solve, only make it slower.
            from . import parallel_gpu_domain
            try:
                self._pool = parallel_gpu_domain.DomainGPUMatVecPool(
                    edof0_host, self._KE_host, lv0['ndof'], plan.n_workers,
                    dims=lv0['dims'], verbose=self.verbose)
                return
            except Exception as exc:
                if self.verbose:
                    print(f"[Blendtopo] MG: domain-decomposed multi-GPU pool "
                          f"unavailable ({exc}); trying plain multi-GPU pool")
            from . import parallel_gpu
            try:
                self._pool = parallel_gpu.GPUMatVecPool(
                    edof0_host, self._KE_host, lv0['ndof'], plan.n_workers,
                    verbose=self.verbose)
                return
            except Exception as exc:
                if self.verbose:
                    print(f"[Blendtopo] MG: multi-GPU pool unavailable "
                          f"({exc}); using single GPU/CPU")
                self.plan = compute_plan.choose(
                    lv0['ndof'], mode="GPU", verbose=self.verbose)

    def close(self):
        """Release the level-0 matvec pool's workers/devices, if any."""
        if self._pool is not None:
            try:
                self._pool.close()
            except Exception:
                pass
            self._pool = None

    # -- boundary conditions per level --------------------------------------
    def _build_free(self, fixed_dofs):
        xp = self.xp
        free0 = np.ones(self.levels[0]['ndof'], dtype=bool)
        fd = np.asarray(fixed_dofs, dtype=np.int64)
        if fd.size:
            free0[fd] = False
        self.levels[0]['free'] = xp.asarray(free0)
        prev_free = free0
        prev_dims = _node_dims(*self.levels[0]['dims'])
        for li in range(1, len(self.levels)):
            Lcx, Lcy, Lcz = _node_dims(*self.levels[li]['dims'])
            Lfx, Lfy, Lfz = prev_dims
            Nc = Lcx * Lcy * Lcz
            n = np.arange(Nc)
            jx = n % Lcx
            jy = (n // Lcx) % Lcy
            jz = n // (Lcx * Lcy)
            fid = (2 * jx) + Lfx * ((2 * jy) + Lfy * (2 * jz))  # collocated fine node
            free = np.ones(3 * Nc, dtype=bool)
            for c in range(3):
                free[3 * n + c] = prev_free[3 * fid + c]
            self.levels[li]['free'] = xp.asarray(free)
            prev_free = free
            prev_dims = (Lcx, Lcy, Lcz)

    # -- density: average down + per-level diagonal -------------------------
    def set_density(self, Evec_full):
        xp = self.xp
        E = xp.asarray(np.asarray(Evec_full, dtype=float))
        for li, lv in enumerate(self.levels):
            lx, ly, lz = lv['dims']
            if li > 0:
                pe = self.levels[li - 1]['Evec']           # finer Evec (z-fastest)
                plx, ply, plz = self.levels[li - 1]['dims']
                pe3 = pe.reshape(plx, ply, plz)
                E = pe3.reshape(lx, 2, ly, 2, lz, 2).mean(axis=(1, 3, 5)).ravel()
            lv['Evec'] = E
            scaled = E * lv['g']
            if li == 0 and self._pool is not None:
                # Keep the pool's density in sync (needed by _apply_pooled),
                # but compute the diagonal locally -- it's one O(nelem) call
                # per SIMP iteration, not per CG/smoothing iteration, so
                # there's nothing to gain from splitting it, and it avoids
                # one more host round-trip on the hot path.
                self._pool.set_density(backend.asnumpy(scaled))
            d = xp.bincount(lv['edof'].ravel(),
                            weights=(scaled[:, None] * self.diagKE[None, :]).ravel(),
                            minlength=lv['ndof'])
            d = xp.where(d == 0, 1.0, d)
            lv['Minv'] = 1.0 / d
            lv['_scaled'] = scaled

    # -- operators ----------------------------------------------------------
    def _apply(self, li, u):
        """Local (never pooled) matrix-free apply. Used by the V-cycle
        smoother/residual (_smooth, _vcycle) -- called ~5x per outer CG
        iteration just at level 0 alone (n_smooth pre- + post-smoothing
        passes, plus the V-cycle's own residual). Splitting *this* across
        devices/processes was the actual cause of a measured 1.5-3.5x
        MULTI-GPU SLOWDOWN vs. single-GPU on real hardware (2x GTX 970,
        see paper/experiments/benchmark_multigpu.py): each smoothing pass
        paid a full host round-trip on every device, for work too small
        and too frequent to amortize that cost. See _apply_pooled for the
        one call site that's actually meant to be split.

        Wall-clock time spent here (across the whole solve() call, all
        levels combined) accumulates into self._t_local -- see "multi_gpu
        xp history" in the module docstring for why this instrumentation
        was added instead of another guess."""
        t0 = time.perf_counter()
        xp = self.xp
        lv = self.levels[li]
        ue = u[lv['edof']]
        ke = (ue @ self.KE.T) * lv['_scaled'][:, None]
        Ku = xp.bincount(lv['edof'].ravel(), weights=ke.ravel(),
                         minlength=lv['ndof'])
        out = xp.where(lv['free'], Ku, 0.0)
        self._t_local += time.perf_counter() - t0
        self._n_local_calls += 1
        return out

    def _apply_pooled(self, u):
        """The level-0 operator application as called directly by the
        outer CG loop in solve() (initial residual + Ap) -- ~2x per outer
        CG iteration, not per smoothing pass, so it's the one call site
        with enough compute per call to be worth splitting across a
        multi-GPU/multi-CPU pool when the compute plan calls for one.

        Wall-clock time spent here accumulates into self._t_pooled -- see
        _apply's docstring."""
        t0 = time.perf_counter()
        lv = self.levels[0]
        xp = self.xp
        if self._pool is not None:
            Ku = xp.asarray(self._pool.apply(backend.asnumpy(u)))
        else:
            ue = u[lv['edof']]
            ke = (ue @ self.KE.T) * lv['_scaled'][:, None]
            Ku = xp.bincount(lv['edof'].ravel(), weights=ke.ravel(),
                             minlength=lv['ndof'])
        out = xp.where(lv['free'], Ku, 0.0)
        self._t_pooled += time.perf_counter() - t0
        self._n_pooled_calls += 1
        return out

    def _smooth(self, li, u, b, iters):
        xp = self.xp
        lv = self.levels[li]
        free = lv['free']
        Minv = lv['Minv']
        w = self.omega
        for _ in range(iters):
            r = b - self._apply(li, u)
            u = u + w * xp.where(free, Minv * r, 0.0)
        return u

    def _restrict(self, li, r_fine):
        # R = P^T : scatter fine residual to coarse nodes.
        xp = self.xp
        idx, wt = self.prolong[li]
        Nc = self.levels[li + 1]['ndof'] // 3
        rf = r_fine.reshape(-1, 3)
        rc = xp.zeros((Nc, 3))
        for c in range(3):
            rc[:, c] = xp.bincount(idx.ravel(),
                                   weights=(wt * rf[:, c][:, None]).ravel(),
                                   minlength=Nc)
        out = rc.reshape(-1)
        return xp.where(self.levels[li + 1]['free'], out, 0.0)

    def _prolongate(self, li, e_coarse):
        xp = self.xp
        idx, wt = self.prolong[li]
        ec = e_coarse.reshape(-1, 3)
        ef = (ec[idx] * wt[:, :, None]).sum(axis=1)     # (Nf, 3)
        out = ef.reshape(-1)
        return xp.where(self.levels[li]['free'], out, 0.0)

    def _vcycle(self, li, b):
        xp = self.xp
        if li == len(self.levels) - 1:
            return self._smooth(li, xp.zeros_like(b), b, 30)   # coarse "solve"
        u = self._smooth(li, xp.zeros_like(b), b, self.n_smooth)
        r = b - self._apply(li, u)
        rc = self._restrict(li, r)
        ec = self._vcycle(li + 1, rc)
        u = u + self._prolongate(li, ec)
        u = self._smooth(li, u, b, self.n_smooth)
        return u

    # -- public solve -------------------------------------------------------
    def solve(self, Evec_full, f, tol=1e-4, max_cg=None, x0=None):
        xp = self.xp
        t_solve0 = time.perf_counter()
        self._t_pooled = 0.0
        self._t_local = 0.0
        self._n_pooled_calls = 0
        self._n_local_calls = 0
        self.set_density(Evec_full)
        free = self.levels[0]['free']
        f = xp.asarray(np.asarray(f, dtype=float))
        f = xp.where(free, f, 0.0)

        if x0 is None:
            u = xp.zeros(self.ndof)
            r = f.copy()
        else:
            u = xp.where(free, xp.asarray(np.asarray(x0, dtype=float)), 0.0)
            r = f - self._apply_pooled(u)
        r = xp.where(free, r, 0.0)

        fnorm = float(xp.linalg.norm(f))
        if fnorm == 0.0:
            return np.zeros(self.ndof)
        z = self._vcycle(0, r)
        p = z.copy()
        rz = float(r @ z)
        max_cg = max_cg or 200
        self.last_iters = 0
        for it in range(max_cg):
            Ap = self._apply_pooled(p)
            alpha = rz / float(p @ Ap)
            u = u + alpha * p
            r = r - alpha * Ap
            self.last_iters = it + 1
            if float(xp.linalg.norm(r)) / fnorm < tol:
                break
            z = self._vcycle(0, r)
            rz_new = float(r @ z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        u = xp.where(free, u, 0.0)
        if self.verbose:
            t_total = time.perf_counter() - t_solve0
            t_other = t_total - self._t_pooled - self._t_local
            pool_label = (self._pool.__class__.__name__
                          if self._pool is not None else "none (local only)")
            print(f"[Blendtopo] MG solve: {self.last_iters} CG iters, "
                  f"total={t_total * 1e3:.1f}ms | "
                  f"pooled={self._t_pooled * 1e3:.1f}ms "
                  f"({self._n_pooled_calls} calls via {pool_label}) | "
                  f"local={self._t_local * 1e3:.1f}ms "
                  f"({self._n_local_calls} calls, _apply/_smooth across all "
                  f"levels) | other (CG vector ops, restrict/prolongate, "
                  f"set_density)={t_other * 1e3:.1f}ms")
        return u if xp is np else xp.asnumpy(u)
