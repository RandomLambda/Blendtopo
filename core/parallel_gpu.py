# SPDX-License-Identifier: GPL-3.0-or-later
"""
Multi-GPU matvec for the matrix-free FEA (GPU edition only).

The element list is split into contiguous chunks, one per visible CUDA
device; each device computes a partial output vector (gather -> 24x24 matmul
-> scale by density -> scatter-add), then the partials are summed on the
host. Because the scatter-add only writes by index, summing per-device
partials reproduces exactly what one full-precision bincount over all
elements would give (up to floating-point summation-order round-off).

One Python process drives multiple CUDA devices via ``cp.cuda.Device(i)``
context managers; each "worker" is a per-device array cache. Per apply()
call: broadcast p to every device (host-staged H2D), compute the local
partial, copy back and sum on the host. Peer-to-peer transfers are
deliberately not used -- ``cudaDeviceEnablePeerAccess`` can fail silently on
some topologies (mixed GPU models, different PCIe roots), and host-staging
always works.

Only activates when compute_plan.choose() picks "multi_gpu" (see that
module for the threshold). Superseded for MGSolver's level-0 operator by
the domain-decomposed pool in parallel_gpu_domain.py, which transfers only
each device's own DOF slice; this module remains as fea.VoxelFEA's
multi_gpu fallback for its arbitrary active-element subsets.
"""

import numpy as np


class GPUMatVecPool:
    """Element-partitioned matrix-free matvec across CUDA devices."""

    def __init__(self, edof, KE, ndof, n_devices, verbose=False):
        import cupy as cp
        self._cp = cp

        edof = np.ascontiguousarray(edof, dtype=np.int64)
        KE = np.ascontiguousarray(KE, dtype=np.float64)
        nelem = edof.shape[0]
        n_devices = max(1, int(n_devices))
        bounds = np.linspace(0, nelem, n_devices + 1).astype(np.int64)
        chunks = [(int(bounds[i]), int(bounds[i + 1]))
                  for i in range(n_devices) if bounds[i + 1] > bounds[i]]
        if len(chunks) < 2:
            raise ValueError(
                "GPUMatVecPool needs at least 2 non-empty element chunks "
                f"(got nelem={nelem}, n_devices={n_devices})")

        self.ndof = int(ndof)
        self.verbose = verbose
        self.device_ids = list(range(len(chunks)))
        self._edof_dev = []
        self._KE_dev = []
        self._evec_dev = []
        self._diagKE_dev = []
        self._p_dev = []          # preallocated per-device copy of p (reused)

        for dev_id, (a, b) in zip(self.device_ids, chunks):
            with cp.cuda.Device(dev_id):
                self._edof_dev.append(cp.asarray(edof[a:b]))
                ke = cp.asarray(KE)
                self._KE_dev.append(ke)
                self._diagKE_dev.append(cp.diag(ke))
                self._evec_dev.append(cp.zeros(b - a, dtype=cp.float64))
                # Full-length (ndof) since each device gathers via its own
                # edof slice, which indexes into the whole p vector.
                self._p_dev.append(cp.empty(self.ndof, dtype=cp.float64))

        if verbose:
            print(f"[Blendtopo] GPUMatVecPool: {len(chunks)} CUDA devices "
                  f"{self.device_ids} over {nelem} elements ({self.ndof} DOF) "
                  f"-- host-staged transfers (no P2P)")

    def set_density(self, evec):
        cp = self._cp
        evec = np.asarray(evec, dtype=np.float64).ravel()
        pos = 0
        for dev_id, buf in zip(self.device_ids, self._evec_dev):
            n = buf.shape[0]
            with cp.cuda.Device(dev_id):
                buf[:] = cp.asarray(evec[pos:pos + n])
            pos += n

    def _reduce(self, per_device_partials):
        # Each partial is a device array of length ndof, still resident on
        # its own device; bring each to host and sum there.
        out = None
        for part in per_device_partials:
            host = self._cp.asnumpy(part)
            out = host if out is None else out + host
        return out

    def apply(self, p):
        cp = self._cp
        p_host = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
        partials = []
        for dev_id, edof_d, ke_d, evec_d, p_d in zip(
                self.device_ids, self._edof_dev, self._KE_dev,
                self._evec_dev, self._p_dev):
            with cp.cuda.Device(dev_id):
                # Copy into the preallocated device buffer instead of
                # cp.asarray(), which would allocate a fresh ndof-length
                # array every CG iteration -- this is called thousands of
                # times over a full solve, so the allocator churn adds up.
                p_d.set(p_host)
                ue = p_d[edof_d]
                contrib = (ue @ ke_d.T) * evec_d[:, None]
                part = cp.bincount(edof_d.ravel(), weights=contrib.ravel(),
                                   minlength=self.ndof)
                partials.append(part)
        return self._reduce(partials)

    def diagonal(self):
        cp = self._cp
        partials = []
        for dev_id, edof_d, diagKE_d, evec_d in zip(
                self.device_ids, self._edof_dev, self._diagKE_dev,
                self._evec_dev):
            with cp.cuda.Device(dev_id):
                contrib = evec_d[:, None] * diagKE_d[None, :]
                part = cp.bincount(edof_d.ravel(), weights=contrib.ravel(),
                                   minlength=self.ndof)
                partials.append(part)
        return self._reduce(partials)

    def close(self):
        # Nothing to release explicitly -- Cu-Py device arrays are garbage
        # collected normally; there is no persistent OS-level resource (no
        # separate processes) unlike the CPU pool.
        pass
