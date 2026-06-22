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
numpy; uses cupy transparently if an array module is passed in.
"""

import numpy as np

from .fea import _hex8_KE, build_edof


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
                 n_smooth=2, omega=0.6, min_elems=4, max_levels=6):
        self.xp = xp if xp is not None else np
        xp = self.xp
        self.nx, self.ny, self.nz = nx, ny, nz
        self.nelem = nx * ny * nz
        self.ndof = 3 * (nx + 1) * (ny + 1) * (nz + 1)
        self.n_smooth = n_smooth
        self.omega = omega

        KE = _hex8_KE(E=1.0, nu=nu)
        self.KE = xp.asarray(KE)
        self.diagKE = xp.asarray(np.diag(KE))

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
            d = xp.bincount(lv['edof'].ravel(),
                            weights=(scaled[:, None] * self.diagKE[None, :]).ravel(),
                            minlength=lv['ndof'])
            d = xp.where(d == 0, 1.0, d)
            lv['Minv'] = 1.0 / d
            lv['_scaled'] = scaled

    # -- operators ----------------------------------------------------------
    def _apply(self, li, u):
        xp = self.xp
        lv = self.levels[li]
        ue = u[lv['edof']]
        ke = (ue @ self.KE.T) * lv['_scaled'][:, None]
        Ku = xp.bincount(lv['edof'].ravel(), weights=ke.ravel(),
                         minlength=lv['ndof'])
        return xp.where(lv['free'], Ku, 0.0)

    def _smooth(self, li, u, b, iters):
        xp = self.xp
        lv = self.levels[li]
        free = lv['free']; Minv = lv['Minv']; w = self.omega
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
        self.set_density(Evec_full)
        free = self.levels[0]['free']
        f = xp.asarray(np.asarray(f, dtype=float))
        f = xp.where(free, f, 0.0)

        if x0 is None:
            u = xp.zeros(self.ndof)
            r = f.copy()
        else:
            u = xp.where(free, xp.asarray(np.asarray(x0, dtype=float)), 0.0)
            r = f - self._apply(0, u)
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
            Ap = self._apply(0, p)
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
        return u if xp is np else xp.asnumpy(u)
