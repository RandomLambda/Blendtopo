# SPDX-License-Identifier: GPL-3.0-or-later
"""
Domain-decomposed multi-GPU matvec pool for the geometric-multigrid solver's
level-0 operator (GPU edition only, real `import cupy`).

This is a SEPARATE, opt-in alternative to ``parallel_gpu.GPUMatVecPool``,
which is left completely untouched -- nothing here modifies that file, nor
``fea.py`` (whose ``VoxelFEA`` keeps using the plain ``parallel_gpu`` pool for
its own, rarely-hit multi_gpu fallback path; see "Scope" below). Only
``multigrid.py``'s ``_init_pool`` is told to construct this class instead of
``parallel_gpu.GPUMatVecPool`` when a domain-decomposed pool is requested --
a small, easily-reverted dispatch change, not a rewrite of the trusted CG/
V-cycle code.

Why a new pool at all
----------------------
``parallel_gpu.GPUMatVecPool`` only splits the ELEMENT list across devices;
every ``apply()`` call still broadcasts the FULL ``ndof``-length vector to
EVERY device and gathers a FULL ``ndof``-length result back from each --
O(n_devices * ndof) host<->device transfer per call, on top of the CG/
V-cycle's own O(ndof) host vector arithmetic (`fea.py` / `multigrid.py` force
``xp = numpy`` whenever a pool is active, so that part was never the
bottleneck). At small problems, two GPUs hiding each other's kernel-launch
latency wins outright; at large problems (~1M+ DOF) the doubled transfer
volume outgrows the halved per-device compute, and multi-GPU becomes SLOWER
than a single GPU -- confirmed by measurement (see the paper/experiments
benchmark suite and the reviewer analysis that motivated this file).

The fix: real domain decomposition
-----------------------------------
``core.fea.build_edof`` / ``core.multigrid.MGSolver`` number NODES with ``ix``
fastest and ``iz`` SLOWEST (``node_id = ix + (nx+1)*(iy + (ny+1)*iz)``, see
``fea._node_id``). That means a contiguous range of ``iz`` (a "z-slab" of the
regular voxel grid) maps to a exactly contiguous range of node/DOF ids --
no remapping needed to get a spatial partition that is also an index-
contiguous one.

We partition the full grid's ELEMENTS into ``n_devices`` contiguous z-slabs
(note: elements themselves are ordered ``ix`` slowest / ``iz`` fastest by
``build_edof``'s ``meshgrid(..., indexing='ij').ravel()``, the OPPOSITE
convention from nodes -- so slab membership is computed via ``elem_iz = e %
nz``, not via slicing the flat edof array). Slab k owns elements with
``iz in [a_k, b_k)``; since element ``iz`` touches node planes ``iz`` and
``iz+1``, slab k's own elements only ever touch node planes ``[a_k, b_k]``
(inclusive) -- a contiguous DOF range of size ``3*(b_k-a_k+1)*(nx+1)*(ny+1)``,
one plane wider than its own element range at each *internal* boundary, and
generally much smaller than the full ``ndof``.

So each device only ever needs (and only ever transfers) ITS OWN dof slice:

  * ``apply(p)``: slice ``p`` to ``[lo_k, hi_k)`` on the host, H2D only that
    slice, compute the local gather/matmul/scatter with the device's own
    (rebased-to-local) edof, D2H only the local-sized partial result, and
    ``+=`` it into a host output array at ``[lo_k, hi_k)``. The internal
    shared node plane between slab k and k+1 receives a partial from BOTH
    neighbours via this ``+=`` -- exactly reproducing what one whole-grid
    bincount would give (same non-approximation argument as
    ``parallel_cpu.py`` / ``parallel_gpu.py``, just with the transferred
    slice shrunk too, not only the compute).
  * ``diagonal()``: identical pattern, no ``p`` needed.

Total host<->device transfer volume across ALL devices becomes O(ndof) --
same order as a single device -- instead of O(n_devices * ndof). This is
what actually fixes the large-problem regression; the CG/V-cycle vector
arithmetic in ``multigrid.py`` is untouched and stays on the host exactly as
before (those ops are O(ndof) numpy, already cheap per the existing
docstrings' own reasoning -- the transfer volume of the MATVEC was the
problem, not the vector math around it).

No device-to-device communication is used or needed: the host already sees
every device's slice each call (it's the one calling ``apply(p_host)``), so
routing the small per-slab slices through the host is simplest and requires
no P2P setup -- and per-device volume is already minimal (no full-vector
bounce), so there is nothing further to gain by keeping state resident
across calls without also rewriting the CG loop itself (out of scope here,
see "Scope" below).

Scope
-----
This pool only supports the FULL regular grid (``nelem == nx*ny*nz``), i.e.
it is only usable for ``MGSolver``'s level-0 operator. ``fea.VoxelFEA``
restricts the system to an arbitrary ACTIVE-element subset (a masked, non-box
region) -- there's no clean spatial slab there, so ``VoxelFEA``'s multi_gpu
path deliberately keeps using the existing ``parallel_gpu.GPUMatVecPool``
unchanged. In practice this is a minor gap: ``simp.Problem`` always prefers
the multigrid solver (this module's target) and only falls back to
``VoxelFEA`` if multigrid construction fails outright.

Streams + pinned memory (why utilization was still ~50%)
-----------------------------------------------------------
The first version of this file still issued, per device, a *blocking*
H2D copy (pageable host memory always synchronizes the calling thread),
then queued the kernels, then a *blocking* D2H copy (``cp.asnumpy``
implicitly synchronizes) -- all on the default stream, one device at a
time. Measured on real hardware (2x GTX 970) this left the GPUs at ~50%
utilization: idle during both copy phases, busy only during the actual
kernel execution, with zero overlap between a device's own transfer and
compute, and zero overlap BETWEEN devices either (the Python loop only
moves to device 1 after device 0's blocking D2H has returned).

Fix: each device gets its own non-blocking ``cupy.cuda.Stream`` and a pair
of *pinned* (page-locked) host buffers (one for the p-slice gather, one for
the result scatter), allocated once in ``__init__`` and reused every call.
``ndarray.set(..., stream=...)``/``ndarray.get(stream=..., out=...)`` on
pinned memory queue truly asynchronous H2D/D2H copies -- the apply() loop
now only *dispatches* work (H2D, kernels, D2H, all queued to that device's
stream) without waiting, moves straight to the next device, and only
synchronizes ALL streams once, after every device has had its work queued.
This lets device 1's transfer overlap device 0's compute, and lets each
device's own H2D/compute/D2H pipeline overlap across successive apply()
calls to some extent (CUDA streams preserve issue order but don't force a
device to sit idle between ops the way the old blocking-Python-loop did).

Correctness of the ordering: ``set_density()`` writes ``evec_dev`` on the
SAME per-device stream ``apply()``/``diagonal()`` use (not the default
stream). Non-blocking streams do NOT implicitly synchronize with each
other or with the default stream, so if the density write and the matvec
kernels were on different streams there would be a genuine race (a kernel
could read a stale ``evec_dev`` before the new density value had actually
landed). Issuing both on the identical stream sidesteps this entirely:
CUDA guarantees in-order execution *within* one stream, so by construction
every ``apply()``/``diagonal()`` queued after a ``set_density()`` call on
that device sees the density write completed first -- no explicit
synchronize() needed between them, only once per apply()/diagonal() call
(at the end, before reading the pinned output buffers back).

** Like parallel_gpu.py, this has been written and reasoned through carefully
but not exercised on real multi-GPU hardware. Its core partition/assemble
ALGORITHM is covered by a CPU-only correctness check (see
tests/test_core.py / the fake-cupy harness used during development) that
verifies it reproduces the exact single-domain bincount result bit-for-bit
for the array operations involved -- but real-hardware timing behaviour
(actual transfer speed, multi-GPU topology, driver quirks, and now stream
overlap) can only be validated on your own machine. Run with "Verbose
solver log" enabled and check the printed per-slab DOF sizes / timings the
first time you use it, and watch nvidia-smi / nsys for whether utilization
actually improved. **
"""

import numpy as np


def _element_iz(nx, ny, nz):
    """iz for every element in build_edof's flat order (ix slowest, iz
    fastest -- see fea.build_edof's meshgrid(..., indexing='ij').ravel())."""
    e = np.arange(nx * ny * nz)
    return e % nz


class DomainGPUMatVecPool:
    """Domain-decomposed (z-slab) multi-GPU matvec pool for a FULL regular
    voxel grid's level-0 operator. See module docstring for the design."""

    def __init__(self, edof, KE, ndof, n_devices, dims, verbose=False):
        import cupy as cp
        self._cp = cp

        nx, ny, nz = dims
        edof = np.ascontiguousarray(edof, dtype=np.int64)
        KE = np.ascontiguousarray(KE, dtype=np.float64)
        nelem = edof.shape[0]
        if nelem != nx * ny * nz:
            raise ValueError(
                "DomainGPUMatVecPool requires the FULL regular grid's edof "
                f"(got nelem={nelem}, expected {nx * ny * nz} from "
                f"dims={dims}) -- not usable for an active-element subset")

        n_devices = max(1, int(n_devices))
        bounds = np.linspace(0, nz, n_devices + 1).astype(np.int64)
        slabs = [(int(bounds[i]), int(bounds[i + 1]))
                 for i in range(n_devices) if bounds[i + 1] > bounds[i]]
        if len(slabs) < 2:
            raise ValueError(
                "DomainGPUMatVecPool needs at least 2 non-empty z-slabs "
                f"(got nz={nz}, n_devices={n_devices})")

        iz = _element_iz(nx, ny, nz)
        plane_nodes = (nx + 1) * (ny + 1)
        dof_per_plane = 3 * plane_nodes

        self.ndof = int(ndof)
        self.n_devices = len(slabs)
        self.verbose = verbose
        self._closed = False
        self.device_ids = list(range(len(slabs)))
        self.slabs = slabs

        self._dof_lo = []
        self._dof_hi = []
        self._local_ndof = []
        self._masks = []           # per-device element mask into the ORIGINAL
                                    # (global) element order, for set_density
        self._edof_dev = []
        self._KE_dev = []
        self._evec_dev = []
        self._diagKE_dev = []
        self._p_dev = []           # preallocated device buffer for the
                                    # gathered p-slice (reused every call)
        self._streams = []         # one non-blocking stream per device
        self._p_pinned_mem = []    # kept alive: PinnedMemory objects
        self._p_pinned = []        # numpy views onto them (H2D source)
        self._out_pinned_mem = []
        self._out_pinned = []      # numpy views onto pinned buffers (D2H dest)

        for dev_id, (a, b) in enumerate(slabs):
            mask = (iz >= a) & (iz < b)
            edof_local_global = edof[mask]              # (n_local_elem, 24)
            lo = a * dof_per_plane
            hi = (b + 1) * dof_per_plane
            edof_local = edof_local_global - lo          # rebase to [0, hi-lo)
            local_ndof = hi - lo

            with cp.cuda.Device(dev_id):
                stream = cp.cuda.Stream(non_blocking=True)
                edof_d = cp.asarray(edof_local)
                ke_d = cp.asarray(KE)
                self._edof_dev.append(edof_d)
                self._KE_dev.append(ke_d)
                self._diagKE_dev.append(cp.diag(ke_d))
                self._evec_dev.append(
                    cp.zeros(edof_local.shape[0], dtype=cp.float64))
                self._p_dev.append(cp.empty(local_ndof, dtype=cp.float64))

                p_pin_mem = cp.cuda.alloc_pinned_memory(local_ndof * 8)
                p_pin = np.frombuffer(p_pin_mem, dtype=np.float64,
                                      count=local_ndof)
                out_pin_mem = cp.cuda.alloc_pinned_memory(local_ndof * 8)
                out_pin = np.frombuffer(out_pin_mem, dtype=np.float64,
                                        count=local_ndof)

            self._streams.append(stream)
            self._p_pinned_mem.append(p_pin_mem)
            self._p_pinned.append(p_pin)
            self._out_pinned_mem.append(out_pin_mem)
            self._out_pinned.append(out_pin)
            self._masks.append(mask)
            self._dof_lo.append(int(lo))
            self._dof_hi.append(int(hi))
            self._local_ndof.append(int(local_ndof))

        if verbose:
            sizes = ", ".join(str(h - l) for l, h in
                               zip(self._dof_lo, self._dof_hi))
            print(f"[Blendtopo] DomainGPUMatVecPool: {self.n_devices} CUDA "
                  f"devices, z-slabs {slabs}, local DOF sizes [{sizes}] "
                  f"(full ndof={self.ndof}) -- total transfer per apply() "
                  f"is O(ndof) once, not O(n_devices*ndof)")

    def set_density(self, evec):
        """evec: (nelem,) per-element scale, in build_edof's element order
        (same order the caller built ``edof`` from -- not remapped here).

        Issued on each device's OWN stream (not the default stream) so that
        CUDA's in-order-per-stream guarantee, not incidental timing, is what
        makes the next apply()/diagonal() on that device see this write --
        see the module docstring's "Correctness of the ordering" section.
        """
        cp = self._cp
        evec = np.asarray(evec, dtype=np.float64).ravel()
        for dev_id in self.device_ids:
            mask = self._masks[dev_id]
            stream = self._streams[dev_id]
            with cp.cuda.Device(dev_id):
                self._evec_dev[dev_id].set(
                    np.ascontiguousarray(evec[mask]), stream=stream)

    def apply(self, p):
        """Return K @ p (full ndof-length result). Only transfers each
        device's own DOF slice, not the full vector, and dispatches every
        device's H2D/compute/D2H onto its own stream before synchronizing
        (once, for all devices) at the end -- see module docstring."""
        cp = self._cp
        p_host = np.asarray(p, dtype=np.float64)
        out = np.zeros(self.ndof, dtype=np.float64)

        for dev_id in self.device_ids:
            lo, hi = self._dof_lo[dev_id], self._dof_hi[dev_id]
            local_ndof = self._local_ndof[dev_id]
            edof_d = self._edof_dev[dev_id]
            ke_d = self._KE_dev[dev_id]
            evec_d = self._evec_dev[dev_id]
            p_d = self._p_dev[dev_id]
            stream = self._streams[dev_id]
            p_pin = self._p_pinned[dev_id]
            out_pin = self._out_pinned[dev_id]
            with cp.cuda.Device(dev_id):
                p_pin[:] = p_host[lo:hi]              # host->host, cheap
                p_d.set(p_pin, stream=stream)          # async H2D (pinned)
                ue = p_d[edof_d]
                contrib = (ue @ ke_d.T) * evec_d[:, None]
                part = cp.bincount(edof_d.ravel(), weights=contrib.ravel(),
                                    minlength=local_ndof)
                part.get(stream=stream, out=out_pin)   # async D2H (pinned)
            # No synchronize here: move on to the next device immediately so
            # its H2D/kernels can be dispatched (and overlap on the GPU
            # side) while this device is still mid-flight.

        for stream in self._streams:
            stream.synchronize()
        for dev_id in self.device_ids:
            lo, hi = self._dof_lo[dev_id], self._dof_hi[dev_id]
            out[lo:hi] += self._out_pinned[dev_id]
        return out

    def diagonal(self):
        """Return the (unclamped) diagonal contribution vector. Same
        dispatch-all-then-synchronize-once pattern as apply()."""
        cp = self._cp
        out = np.zeros(self.ndof, dtype=np.float64)

        for dev_id in self.device_ids:
            lo, hi = self._dof_lo[dev_id], self._dof_hi[dev_id]
            local_ndof = self._local_ndof[dev_id]
            edof_d = self._edof_dev[dev_id]
            diagKE_d = self._diagKE_dev[dev_id]
            evec_d = self._evec_dev[dev_id]
            stream = self._streams[dev_id]
            out_pin = self._out_pinned[dev_id]
            with cp.cuda.Device(dev_id):
                contrib = evec_d[:, None] * diagKE_d[None, :]
                part = cp.bincount(edof_d.ravel(), weights=contrib.ravel(),
                                    minlength=local_ndof)
                part.get(stream=stream, out=out_pin)

        for stream in self._streams:
            stream.synchronize()
        for dev_id in self.device_ids:
            lo, hi = self._dof_lo[dev_id], self._dof_hi[dev_id]
            out[lo:hi] += self._out_pinned[dev_id]
        return out

    def close(self):
        # Device arrays and pinned host buffers are released when this pool
        # (and its attribute references to them) is garbage collected;
        # there is no persistent OS-level resource (no separate processes),
        # same as parallel_gpu.GPUMatVecPool.
        self._closed = True
