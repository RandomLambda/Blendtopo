# SPDX-License-Identifier: GPL-3.0-or-later
"""
SIMP topology optimization driver.

Density-based (SIMP) method with a sensitivity filter and the classic
optimality-criteria (OC) update. Implemented as a generator so the caller can
remesh / show progress after every iteration and stop whenever it likes.

All voxel arrays are flattened with order matching fea.build_edof, i.e.
element index = ex + nx*(ey + ny*ez)  (numpy 'ij' meshgrid ravel).
"""

import itertools

import numpy as np

from .fea import VoxelFEA
from .multigrid import MGSolver


def resample_density(old_rho3d, old_origin, old_vsize, new_centers):
    """Trilinearly sample a coarse density field at new voxel centers.

    Both grids share the same world origin (build-space bbox min corner), so we
    map each new voxel center into the old grid's fractional index space and
    interpolate. Used to warm-start a finer level from a coarser one.

    old_rho3d   : (oxn, oyn, ozn) density on the coarse grid
    old_origin  : world min corner (3,)
    old_vsize   : coarse voxel edge length
    new_centers : (N, 3) world coords of fine voxel centers
    returns     : (N,) interpolated density
    """
    oxn, oyn, ozn = old_rho3d.shape
    old_origin = np.asarray(old_origin, dtype=float)
    fi = (np.asarray(new_centers, dtype=float) - old_origin) / old_vsize - 0.5
    f0 = np.floor(fi).astype(int)
    fr = fi - f0

    out = np.zeros(len(new_centers))
    for corner in itertools.product((0, 1), repeat=3):
        w = np.ones(len(new_centers))
        for d in range(3):
            w *= fr[:, d] if corner[d] else (1.0 - fr[:, d])
        ix = np.clip(f0[:, 0] + corner[0], 0, oxn - 1)
        iy = np.clip(f0[:, 1] + corner[1], 0, oyn - 1)
        iz = np.clip(f0[:, 2] + corner[2], 0, ozn - 1)
        out += w * old_rho3d[ix, iy, iz]
    return out


def resample_displacement(old_u, old_origin, old_vsize, old_node_dims,
                          new_node_coords):
    """Trilinearly prolongate a nodal displacement field to a finer node grid.

    old_u           : full-length displacement (3 per node, node-id order)
    old_node_dims   : (nxc+1, nyc+1, nzc+1) node counts of the coarse grid
    new_node_coords : (N, 3) world coords of the fine grid nodes (node-id order)
    returns         : full-length displacement for the fine grid (3*N,)
    Used to warm-start the first CG solve of each finer level.
    """
    Lx, Ly, Lz = old_node_dims
    u_nodal = np.asarray(old_u, dtype=float).reshape(-1, 3)
    lattice = u_nodal.reshape(Lz, Ly, Lx, 3)          # [iz, iy, ix, :]
    o = np.asarray(old_origin, dtype=float)
    fi = (np.asarray(new_node_coords, dtype=float) - o) / old_vsize
    f0 = np.floor(fi).astype(int)
    fr = fi - f0
    out = np.zeros((len(new_node_coords), 3))
    for corner in itertools.product((0, 1), repeat=3):
        w = np.ones(len(new_node_coords))
        for d in range(3):
            w *= fr[:, d] if corner[d] else (1.0 - fr[:, d])
        ix = np.clip(f0[:, 0] + corner[0], 0, Lx - 1)
        iy = np.clip(f0[:, 1] + corner[1], 0, Ly - 1)
        iz = np.clip(f0[:, 2] + corner[2], 0, Lz - 1)
        out += w[:, None] * lattice[iz, iy, ix, :]
    return out.ravel()


def _build_filter(nx, ny, nz, rmin):
    """Precompute neighbour offsets and weights for the density filter."""
    r = int(np.floor(rmin))
    offsets = []
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                w = rmin - dist
                if w > 0:
                    offsets.append((dx, dy, dz, w))
    return offsets


def _apply_filter(field3d, offsets):
    """Weighted neighbourhood average of a 3D field (zero-padded edges)."""
    out = np.zeros_like(field3d)
    wsum = np.zeros_like(field3d)
    nx, ny, nz = field3d.shape
    for dx, dy, dz, w in offsets:
        sx0, sx1 = max(0, -dx), nx - max(0, dx)
        sy0, sy1 = max(0, -dy), ny - max(0, dy)
        sz0, sz1 = max(0, -dz), nz - max(0, dz)
        dx0, dx1 = max(0, dx), nx - max(0, -dx)
        dy0, dy1 = max(0, dy), ny - max(0, -dy)
        dz0, dz1 = max(0, dz), nz - max(0, -dz)
        out[dx0:dx1, dy0:dy1, dz0:dz1] += w * field3d[sx0:sx1, sy0:sy1, sz0:sz1]
        wsum[dx0:dx1, dy0:dy1, dz0:dz1] += w
    wsum[wsum == 0] = 1.0
    return out / wsum


class Problem:
    """Container for a discretized optimization problem."""

    def __init__(self, nx, ny, nz, active, fixed_dofs, force,
                 volfrac=0.3, penalty=3.0, rmin=1.5, nu=0.3,
                 e0=1.0, e_min=1e-9, use_multigrid=True,
                 compute_mode="AUTO", cpu_threads=0, verbose=False):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.shape = (nx, ny, nz)
        self.active = active.astype(bool).ravel()      # voxels free to change
        self.volfrac = volfrac
        self.penalty = penalty
        self.e0 = e0
        self.e_min = e_min

        self.fea = VoxelFEA(nx, ny, nz, nu=nu, e_min=e_min,
                            active_elems=self.active, compute_mode=compute_mode,
                            cpu_threads=cpu_threads, verbose=verbose)
        self.fea.set_fixed(fixed_dofs)
        self.force = np.asarray(force, dtype=float)

        # Optional geometric-multigrid solver (full grid). Picks its own
        # CPU/GPU/multi-* plan for the full-grid DOF count (see
        # core/compute_plan.py) and falls back to fea.solve if construction
        # fails outright.
        self.mg = None
        if use_multigrid:
            try:
                self.mg = MGSolver(nx, ny, nz, fixed_dofs, nu=nu,
                                   compute_mode=compute_mode,
                                   cpu_threads=cpu_threads, verbose=verbose)
            except Exception:
                self.mg = None

        self._offsets = _build_filter(nx, ny, nz, rmin)
        self.last_u = None

    def close(self):
        """Release any multi-CPU/multi-GPU pools started for this problem."""
        if self.fea is not None:
            self.fea.close()
        if self.mg is not None:
            self.mg.close()

    def _filter_sens(self, rho, dc):
        """Sensitivity filter (Sigmund): smooths dc, weighted by rho."""
        rho3 = rho.reshape(self.shape)
        dc3 = dc.reshape(self.shape)
        num = _apply_filter(rho3 * dc3, self._offsets)
        den = np.maximum(rho3, 1e-3)
        return (num / den).ravel()

    def optimize(self, max_iter=40, x_init=None, tol=0.01, u_init=None):
        """Generator yielding (it, compliance, change, rho) per iteration.

        x_init : optional flat density array (e.g. upsampled from a coarser
        level) used as a warm start. Values outside the active region are
        ignored; the active region is rescaled to hit the volume fraction.
        """
        active = self.active
        n_active = int(active.sum())
        rho = np.zeros(self.nx * self.ny * self.nz)
        if x_init is not None:
            seed = np.clip(np.asarray(x_init, dtype=float).ravel(), 0.0, 1.0)
            rho[active] = seed[active]
            cur = rho[active].sum()
            target = self.volfrac * n_active
            if cur > 1e-9:
                rho[active] = np.clip(rho[active] * (target / cur), 0.0, 1.0)
            else:
                rho[active] = self.volfrac
        else:
            rho[active] = self.volfrac       # uniform start in active region

        p, e0, emin = self.penalty, self.e0, self.e_min
        u_prev = u_init      # CG warm start (seeded from coarser level)
        prev_change = 1.0    # drives the adaptive CG tolerance

        for it in range(1, max_iter + 1):
            # Adaptive CG tolerance: loose while the shape is still moving a lot,
            # tightening as it settles. Early SIMP steps don't need an accurate
            # displacement, so this saves many CG iterations up front.
            cg_tol = min(2e-3, max(1e-5, 0.1 * prev_change))
            Evec = emin + rho ** p * (e0 - emin)
            if self.mg is not None:
                try:
                    u = self.mg.solve(Evec, self.force, x0=u_prev, tol=cg_tol)
                except Exception:
                    self.mg = None
                    u = self.fea.solve(Evec, self.force, x0=u_prev, tol=cg_tol)
            else:
                u = self.fea.solve(Evec, self.force, x0=u_prev, tol=cg_tol)
            u_prev = u
            self.last_u = u

            ce = self.fea.element_strain_energy(u)        # at unit E
            compliance = float(np.sum(Evec * ce))
            dc = -p * rho ** (p - 1) * (e0 - emin) * ce
            dc[~active] = 0.0

            dc = self._filter_sens(rho, dc)
            dc[~active] = 0.0

            rho_new = self._oc_update(rho, dc, active, n_active)
            change = float(np.max(np.abs(rho_new - rho)[active])) if n_active else 0.0
            prev_change = change
            rho = rho_new

            yield it, compliance, change, rho.copy()

            if change < tol:
                break

    def _oc_update(self, rho, dc, active, n_active, move=0.2):
        """Optimality-criteria density update with bisection on the multiplier."""
        l1, l2 = 1e-9, 1e9
        target_vol = self.volfrac * n_active
        rho_a = rho[active]
        dc_a = dc[active]
        be = np.maximum(-dc_a, 0.0)
        cand = rho_a
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            step = np.sqrt(be / lmid)
            cand = np.clip(rho_a * step,
                           np.maximum(0.0, rho_a - move),
                           np.minimum(1.0, rho_a + move))
            if cand.sum() > target_vol:
                l1 = lmid
            else:
                l2 = lmid
        new = rho.copy()
        new[active] = cand
        return new
