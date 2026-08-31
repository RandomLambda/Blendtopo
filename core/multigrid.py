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

Multi-GPU note: the outer CG matvec (_apply_pooled) is dispatched through
the domain-decomposed pool (see parallel_gpu_domain.py) when one exists;
V-cycle smoothing at every level always runs single-device (self.xp =
plan.xp) and is never pooled per-sweep -- pooling smoothing was measured to
be a net loss (too little compute per call to amortize dispatch/sync
overhead). DomainGPUMatVecPool.smooth() uses a ghost-element ring so each
device's local matvec is exact at its boundary nodes; see that class's
docstring for the derivation. Full history in
paper/experiments/MULTIPROCESSING_FINDINGS.md.
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

    # Round 4e/4f/4g: DomainGPUMatVecPool.smooth() (the ghost-ring,
    # genuinely-parallel level-0 smoother) is mathematically correct but
    # measured SLOWER than single-device smoothing on real hardware (2x
    # GTX 970, Windows/WDDM) -- see _smooth's docstring for the full
    # before/after numbers. Left as a class-level switch (not deleted, not
    # hidden behind a code comment) so it's easy to find and flip to True
    # when re-measuring on hardware with lower per-dispatch overhead.
    _TRY_POOLED_SMOOTH = False

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
        # Set in _init_pool once the plan/pool are known -- True only for a
        # "multi_cpu" plan at/above compute_plan.POOLED_SMOOTH_CPU_DOF (a
        # SECOND, higher bar than MULTI_CPU_DOF itself). See _smooth's
        # docstring and MULTIPROCESSING_FINDINGS.md Section 5.
        self._pool_smooth_enabled = False

        # Wall-clock accounting, reset per solve() call, printed when
        # verbose: plain wall-clock sums (not profiler-grade) that separate
        # time in the pooled multi-device matvec (_t_pooled) from the
        # un-pooled local V-cycle work (_t_local: _apply/_smooth plus
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
        self._requested_mode = (compute_mode or "AUTO").upper()
        if xp is not None:
            # Caller pinned an array module explicitly (e.g. tests): honour
            # it and skip the auto plan/pool entirely.
            self.xp = xp
            self.plan = None
        else:
            self.plan = compute_plan.choose(self.ndof, mode=compute_mode,
                                            cpu_threads=cpu_threads,
                                            verbose=verbose)
            # self.xp is single-device (never pooled) for every plan --
            # measured ~6.5x faster than numpy for this local work. Only
            # _apply_pooled (the outer CG matvec) goes through the
            # multi-device pool; V-cycle smoothing always stays single-device
            # (see module docstring).
            self.xp = self.plan.xp
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
        if plan.kind == "multi_gpu":
            # Domain-decomposed pool first (core/parallel_gpu_domain.py):
            # only each device's own DOF slice is transferred, not the full
            # vector. Only valid for the full regular grid (level 0 always
            # is). Falls back to the plain broadcast pool, then single
            # GPU/CPU -- a broken/unavailable accelerator must never abort
            # the solve, only make it slower.
            from . import parallel_gpu_domain
            # The min-slab-size guard only protects the ghost-ring
            # smoother's quality, irrelevant while _TRY_POOLED_SMOOTH is
            # False (current default). Skipped when the caller explicitly
            # requested MULTI_GPU rather than AUTO picking it, so an
            # explicit choice (e.g. a benchmark script) isn't silently
            # downgraded to a single device.
            enforce_min_slab = (self._TRY_POOLED_SMOOTH
                                 and self._requested_mode != "MULTI_GPU")
            try:
                self._pool = parallel_gpu_domain.DomainGPUMatVecPool(
                    edof0_host, self._KE_host, lv0['ndof'], plan.n_workers,
                    dims=lv0['dims'], verbose=self.verbose,
                    enforce_min_slab=enforce_min_slab)
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
            if li == 0 and self._pool is not None and hasattr(self._pool, 'set_bc'):
                # Keep the pool's per-device Jacobi weight/free-mask in sync
                # for smooth() (see parallel_gpu_domain.DomainGPUMatVecPool
                # .set_bc/.smooth) -- same "cheap enough per SIMP iteration,
                # not per CG/smoothing iteration" reasoning as set_density
                # above. hasattr-guarded because the plain broadcast pool
                # (parallel_gpu.GPUMatVecPool, the multi_gpu fallback one
                # rung down) has no smoother of its own -- MGSolver just
                # keeps using its existing single-device _smooth loop then.
                self._pool.set_bc(backend.asnumpy(lv['Minv']),
                                   backend.asnumpy(lv['free']))

    # -- operators ----------------------------------------------------------
    def _apply(self, li, u):
        """Local (never pooled) matrix-free apply. Used by _vcycle's residual
        computation at every level, and by _smooth as a fallback when no
        domain pool with a smooth() method is available. Splitting
        individual _apply calls across devices (rather than batching a whole
        sweep into one dispatch) measured as a 1.5-3.5x multi-GPU slowdown
        vs. single-GPU -- each call pays a full per-device context-switch
        cost too small to amortize. See _apply_pooled and
        DomainGPUMatVecPool.smooth for the pooled alternatives.

        Wall-clock time spent here accumulates into self._t_local."""
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
        # Pooled multi-GPU smoothing uses DomainGPUMatVecPool's ghost-element
        # ring (each device holds a read-only copy of its neighbour's
        # adjacent element layer, frozen for the whole sweep sequence) so the
        # local matvec is exact at every owned node -- correctness-neutral,
        # only affects preconditioner quality. It's disabled by default
        # (_TRY_POOLED_SMOOTH = False): real-hardware measurement showed it's
        # a net performance loss (fixed per-dispatch overhead roughly
        # triples going from 1 pooled call/iteration to 3), even though it's
        # correct. Flip _TRY_POOLED_SMOOTH to re-measure on other hardware.
        xp = self.xp
        lv = self.levels[li]
        free = lv['free']
        Minv = lv['Minv']
        w = self.omega
        # Two independent gates: _TRY_POOLED_SMOOTH (GPU, off by default per
        # above) and _pool_smooth_enabled (CPU, set from
        # compute_plan.POOLED_SMOOTH_CPU_DOF -- see MULTIPROCESSING_FINDINGS.md).
        # Kept separate so one path's measured result never silently
        # overrides the other's.
        pooled_smooth_ok = (
            (self._TRY_POOLED_SMOOTH and getattr(self.plan, "kind", None) == "multi_gpu")
            or self._pool_smooth_enabled)
        if (pooled_smooth_ok and li == 0 and self._pool is not None
                and hasattr(self._pool, 'smooth')):
            t0 = time.perf_counter()
            u_host = self._pool.smooth(backend.asnumpy(u), backend.asnumpy(b),
                                        iters, w)
            u = xp.asarray(u_host)
            self._t_pooled += time.perf_counter() - t0
            self._n_pooled_calls += 1
            return u
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
        self.last_converged = False
        self.last_resid_ratio = float("nan")
        for it in range(max_cg):
            Ap = self._apply_pooled(p)
            pAp = float(p @ Ap)
            if pAp == 0.0:
                # p@Ap vanishing means the search direction has no more
                # energy left w.r.t. the true operator -- treat as converged
                # (for an SPD system) rather than raising ZeroDivisionError.
                self.last_converged = (self.last_resid_ratio < tol)
                break
            alpha = rz / pAp
            u = u + alpha * p
            r = r - alpha * Ap
            self.last_iters = it + 1
            self.last_resid_ratio = float(xp.linalg.norm(r)) / fnorm
            if self.last_resid_ratio < tol:
                self.last_converged = True
                break
            z = self._vcycle(0, r)
            rz_new = float(r @ z)
            if rz == 0.0:
                # Same reasoning as the pAp guard above: restart the search
                # direction from the raw preconditioned residual instead of
                # dividing by zero (a valid PCG restart).
                p = z.copy()
            else:
                p = z + (rz_new / rz) * p
            rz = rz_new
        u = xp.where(free, u, 0.0)
        if not self.last_converged:
            # One V-cycle should get CG to a near-constant iteration count
            # (typically ~10-30) regardless of resolution, so hitting the cap
            # is a correctness signal worth always printing: either the
            # preconditioner isn't helping for this density state, or (on a
            # pooled multi_gpu plan) the pooled matvec may be inconsistent
            # with the local operator.
            print(f"[Blendtopo] WARNING: MG-PCG did not converge in "
                  f"{max_cg} iterations (tol={tol:g}, final residual ratio="
                  f"{self.last_resid_ratio:.3e}, plan="
                  f"{self.plan.label if self.plan is not None else 'pinned xp'}"
                  f"). Returned displacement may be inaccurate; if reproducible "
                  f"for a size/mode that used to converge, suspect the pooled "
                  f"matvec (parallel_gpu_domain.py / parallel_gpu.py) rather "
                  f"than the multigrid math itself.")
        if self.verbose:
            t_total = time.perf_counter() - t_solve0
            t_other = t_total - self._t_pooled - self._t_local
            pool_label = (self._pool.__class__.__name__
                          if self._pool is not None else "none (local only)")
            print(f"[Blendtopo] MG solve: {self.last_iters} CG iters "
                  f"(converged={self.last_converged}), "
                  f"total={t_total * 1e3:.1f}ms | "
                  f"pooled={self._t_pooled * 1e3:.1f}ms "
                  f"({self._n_pooled_calls} calls via {pool_label}) | "
                  f"local={self._t_local * 1e3:.1f}ms "
                  f"({self._n_local_calls} calls, _apply/_smooth across all "
                  f"levels) | other (CG vector ops, restrict/prolongate, "
                  f"set_density)={t_other * 1e3:.1f}ms")
        return u if xp is np else xp.asnumpy(u)
