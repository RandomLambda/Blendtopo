# SPDX-License-Identifier: GPL-3.0-or-later
"""
Domain-decomposed multi-GPU matvec pool for the geometric-multigrid solver's
level-0 operator (GPU edition only, real `import cupy`).

A separate, opt-in alternative to ``parallel_gpu.GPUMatVecPool``, which only
splits the element list across devices while still broadcasting the full
DOF vector to every device each call -- O(n_devices * ndof) transfer, which
outgrows the halved per-device compute at large problem sizes (~1M+ DOF) and
makes multi-GPU slower than single-GPU.

This pool instead partitions the grid into ``n_devices`` contiguous z-slabs
(``core.fea.build_edof`` numbers nodes with ``iz`` slowest, so a z-slab maps
to a contiguous DOF range). Each device transfers and computes only its own
slice; the shared node plane between adjacent slabs receives a partial from
both neighbours via ``+=``, exactly reproducing a whole-grid bincount.
Total transfer volume across all devices becomes O(ndof), same order as a
single device.

Scope: only supports the full regular grid (level 0 of ``MGSolver``).
``fea.VoxelFEA``'s arbitrary active-element subset has no clean spatial
slab, so it keeps using ``parallel_gpu.GPUMatVecPool`` for its (rarely hit)
multi_gpu fallback; ``simp.Problem`` always prefers the multigrid solver.

Each device uses its own non-blocking ``cupy.cuda.Stream`` and pinned
host buffers so H2D/kernel/D2H work is dispatched without blocking the
Python loop between devices, then synchronized once per call.
``set_density()`` writes on the same per-device stream that
``apply()``/``diagonal()`` use, so in-order stream execution guarantees the
density write lands before any matvec kernel reads it -- no explicit
synchronize needed between them.

Its partition/assemble algorithm is covered by a CPU-only correctness check
(tests/test_core.py, fake-cupy harness) verifying it reproduces the exact
single-domain bincount result. Real-hardware timing (transfer speed,
topology, driver quirks) should still be validated on your own machine --
enable verbose solver logging and check nvidia-smi/nsys for utilization.
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

    def __init__(self, edof, KE, ndof, n_devices, dims, verbose=False,
                 enforce_min_slab=True):
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

        # Refuse to build when a slab is too thin relative to smooth()'s
        # 1-element ghost overlap: at very small slabs the frozen ghost
        # boundary becomes a large fraction of the subdomain, degrading
        # convergence (slow, not wrong) -- MGSolver's fallback chain (domain
        # pool -> plain broadcast pool -> single GPU) takes over instead.
        # Threshold matches MGSolver's own min_elems=4 default. Skipped
        # (enforce_min_slab=False) when the caller explicitly requested
        # MULTI_GPU rather than AUTO picking it, or when pooled smoothing is
        # disabled (the current default) -- an explicit choice must actually
        # get multi-GPU, not a silent substitution.
        MIN_SLAB_ELEMS_FOR_SMOOTHER = 4
        min_slab = min(b - a for a, b in slabs)
        if enforce_min_slab and min_slab < MIN_SLAB_ELEMS_FOR_SMOOTHER:
            raise ValueError(
                f"DomainGPUMatVecPool: smallest z-slab has only {min_slab} "
                f"element(s) (nz={nz}, n_devices={n_devices}) -- too thin "
                f"relative to smooth()'s 1-element ghost overlap to be a "
                f"useful preconditioner; falling back")

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
        self._streams = []         # one non-blocking stream per device
        self._p_dev = []           # preallocated device buffer for the
                                    # gathered p-slice (reused every call)
        self._p_pinned_mem = []    # kept alive: PinnedMemory objects
        self._p_pinned = []        # numpy views onto them (H2D source)
        self._out_pinned_mem = []
        self._out_pinned = []      # numpy views onto pinned buffers (D2H dest)

        # -- smooth()-only state: a GHOST-EXTENDED element/dof range per
        # device, one element layer wider than (dof_lo, dof_hi) on each
        # internal side. See smooth()'s docstring ("Round 4e ghost ring")
        # for why apply()/diagonal()'s plain z-slab partition is NOT reused
        # here: those rely on '+=' summing two devices' PARTIAL matvecs at
        # the shared plane, which is correct for a bincount but was found
        # (real hardware, 2026-07-09) to be WRONG when repurposed as a
        # multi-sweep Jacobi smoother -- a device's local matvec there is
        # only half the true stiffness, and that error compounds sweep over
        # sweep instead of averaging out. Filled in the second per-device
        # loop below (needs sm_elem_lo/sm_elem_hi, which depend on the
        # NEIGHBOURING slab's bounds, so it's simplest as its own pass
        # after `slabs` is fully known).
        self._sm_edof_dev = []
        self._sm_evec_dev = []      # set by set_density(), ghost-extended
        self._sm_masks = []         # per-device GLOBAL element mask for the
                                    # ghost-extended element set (own + one
                                    # neighbour layer each side), for
                                    # set_density's gather
        self._sm_minv_dev = []      # length sm_local_ndof; zero OUTSIDE the
                                    # owned (dof_lo,dof_hi) sub-range so the
                                    # Jacobi update never touches ghost DOFs
        self._sm_free_dev = []      # same shape/zero-outside-owned pattern
        self._sm_dof_lo = []
        self._sm_dof_hi = []
        self._sm_local_ndof = []
        self._u_sm_dev = []         # preallocated device buffer, sm-sized
        self._b_sm_dev = []
        self._u_sm_pinned_mem = []
        self._u_sm_pinned = []
        self._b_sm_pinned_mem = []
        self._b_sm_pinned = []

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

        # -- second pass: smooth()'s ghost-extended (overlapping-Schwarz)
        # element/dof range per device. One MORE element layer than
        # (dof_lo, dof_hi) on each INTERNAL side: device k's element range
        # [a_k, b_k) gains a read-only ghost row at iz=a_k-1 (device k-1's
        # elements, if k>0) and at iz=b_k (device k+1's elements, if k is
        # not last) -- exactly the neighbour's elements that touch device
        # k's own boundary node planes a_k and b_k, which is precisely what
        # was MISSING from the plain z-slab partition when repurposed as a
        # smoother (see smooth()'s docstring). Because own elements plus
        # one ghost row on each side are contiguous in iz, the ghost-
        # extended range is just a WIDER contiguous slab: [sm_elem_lo,
        # sm_elem_hi) = [a_k - 1, b_k + 1) clipped to [0, nz).
        self._sm_owned_off = []
        for dev_id, (a, b) in enumerate(slabs):
            sm_elem_lo = a - 1 if dev_id > 0 else a
            sm_elem_hi = b + 1 if dev_id < len(slabs) - 1 else b
            sm_mask = (iz >= sm_elem_lo) & (iz < sm_elem_hi)
            sm_edof_local_global = edof[sm_mask]
            sm_lo = sm_elem_lo * dof_per_plane
            sm_hi = (sm_elem_hi + 1) * dof_per_plane
            sm_edof_local = sm_edof_local_global - sm_lo
            sm_local_ndof = sm_hi - sm_lo

            own_lo, own_hi = self._dof_lo[dev_id], self._dof_hi[dev_id]
            # Where the (dof_lo, dof_hi) owned range sits inside the wider
            # ghost buffer -- e.g. own_off:own_off+local_ndof.
            own_off = own_lo - sm_lo

            with cp.cuda.Device(dev_id):
                sm_edof_d = cp.asarray(sm_edof_local)
                self._sm_edof_dev.append(sm_edof_d)
                self._sm_evec_dev.append(
                    cp.zeros(sm_edof_local.shape[0], dtype=cp.float64))
                self._sm_minv_dev.append(
                    cp.zeros(sm_local_ndof, dtype=cp.float64))
                self._sm_free_dev.append(
                    cp.zeros(sm_local_ndof, dtype=cp.bool_))
                self._u_sm_dev.append(cp.empty(sm_local_ndof, dtype=cp.float64))
                self._b_sm_dev.append(cp.empty(sm_local_ndof, dtype=cp.float64))

                u_pin_mem = cp.cuda.alloc_pinned_memory(sm_local_ndof * 8)
                u_pin = np.frombuffer(u_pin_mem, dtype=np.float64,
                                      count=sm_local_ndof)
                b_pin_mem = cp.cuda.alloc_pinned_memory(sm_local_ndof * 8)
                b_pin = np.frombuffer(b_pin_mem, dtype=np.float64,
                                      count=sm_local_ndof)

            self._u_sm_pinned_mem.append(u_pin_mem)
            self._u_sm_pinned.append(u_pin)
            self._b_sm_pinned_mem.append(b_pin_mem)
            self._b_sm_pinned.append(b_pin)
            self._sm_masks.append(sm_mask)
            self._sm_dof_lo.append(int(sm_lo))
            self._sm_dof_hi.append(int(sm_hi))
            self._sm_local_ndof.append(int(sm_local_ndof))
            # Where inside the sm buffer the true owned (dof_lo, dof_hi)
            # range starts, so set_bc() can write minv/free only there
            # (leaving the ghost region at its zero/False default, which
            # naturally freezes it -- see smooth()'s docstring).
            self._sm_owned_off.append(int(own_off))

        # Exclusive OWNERSHIP ranges for smooth()'s OUTPUT merge (which
        # values to copy back out of each device's owned sub-range into the
        # global result). With the ghost ring above, both neighbours now
        # compute the SAME shared plane EXACTLY (matching stiffness inputs),
        # so which one's copy is kept no longer changes correctness -- keep
        # the same non-overlapping convention as before anyway (simpler
        # than re-averaging two now-equal values, and avoids relying on
        # bit-exact floating-point equality between two different summation
        # orders): give every device its bottom shared plane (dof_lo) but
        # not its top one (ceded to the next device via ITS dof_lo).
        own_lo = list(self._dof_lo)
        own_hi = list(self._dof_hi)
        for k in range(len(own_hi) - 1):
            own_hi[k] -= dof_per_plane
        self._own_lo = own_lo
        self._own_hi = own_hi

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
            sm_mask = self._sm_masks[dev_id]
            stream = self._streams[dev_id]
            with cp.cuda.Device(dev_id):
                self._evec_dev[dev_id].set(
                    np.ascontiguousarray(evec[mask]), stream=stream)
                # Ghost-extended copy for smooth() -- same source array,
                # wider element mask (own elements + one neighbour layer
                # each side, see __init__). Needed so smooth()'s local
                # matvec sees the correct (current SIMP iteration's)
                # density for the ghost row too, not just its own elements.
                self._sm_evec_dev[dev_id].set(
                    np.ascontiguousarray(evec[sm_mask]), stream=stream)

    def set_bc(self, minv_full, free_full):
        """Push the level-0 clamped inverse-diagonal (``minv_full``) and
        free-dof mask (``free_full``), both length ``ndof`` and in the SAME
        global dof order as ``apply()``'s ``p``/output, needed by
        ``smooth()``. ``minv_full`` changes every SIMP iteration (density-
        dependent, see MGSolver.set_density -- call this right after that
        recomputes it); ``free_full`` never changes across a whole solve but
        is cheap enough to re-send every time rather than plumb a separate
        one-time call.

        Only writes the OWNED (dof_lo, dof_hi) sub-range of each device's
        wider ghost buffer -- the ghost region itself is left at its
        __init__ default (minv=0, free=False), which makes smooth()'s
        Jacobi update a no-op there by construction (multiplying by a
        zeroed Minv, and/or masked by free=False), i.e. the ghost DOFs are
        read-only boundary data for the whole sweep sequence and are never
        written back -- see smooth()'s docstring for why freezing them
        (rather than updating from a still-partial local operator there)
        is what makes this correct."""
        cp = self._cp
        minv_full = np.asarray(minv_full, dtype=np.float64).ravel()
        free_full = np.asarray(free_full, dtype=bool).ravel()
        for dev_id in self.device_ids:
            lo, hi = self._dof_lo[dev_id], self._dof_hi[dev_id]
            local_ndof = hi - lo
            off = self._sm_owned_off[dev_id]
            stream = self._streams[dev_id]
            with cp.cuda.Device(dev_id):
                self._sm_minv_dev[dev_id][off:off + local_ndof].set(
                    np.ascontiguousarray(minv_full[lo:hi]), stream=stream)
                self._sm_free_dev[dev_id][off:off + local_ndof].set(
                    np.ascontiguousarray(free_full[lo:hi]), stream=stream)

    def smooth(self, u, b, iters, omega):
        """Run ``iters`` weighted-Jacobi sweeps for the level-0 operator,
        genuinely in parallel across every GPU: the entire local sweep
        sequence runs per device without a host round-trip between sweeps,
        unlike apply() which dispatches one matvec per call.

        Each device's local element set is one layer wider than its own
        z-slab: it holds a read-only ghost copy of the adjacent element row
        from each neighbour (refreshed by set_density() every SIMP
        iteration), so its local bincount at both boundary node planes is
        numerically exact -- standard overlapping Schwarz domain
        decomposition. Ghost DOF values are frozen for the whole sweep
        sequence (set_bc() leaves minv/free at 0/False outside the owned
        sub-range), which is the standard "lagged" boundary data of a block
        Jacobi/Schwarz smoother: an approximation of which residual to
        smooth against, never of the operator itself, so it only affects
        iteration count, never correctness (see the module docstring).

        Batching all sweeps into one dispatch per device is what lets both
        GPUs compute concurrently -- exchanging ghost values every sweep
        would need a host round-trip per sweep, the same overhead that
        makes pooling individual matvecs a net loss (see apply()'s
        docstring). After all sweeps, each device's owned sub-range is
        copied back as a plain non-overlapping slice write.

        Validated on real hardware to match single-GPU's CG iteration count
        exactly at moderate-to-large grid sizes; small grids (fewer than a
        few elements per device) degrade because the frozen ghost boundary
        is then a large fraction of the subdomain -- see __init__'s
        min-slab-size guard.
        """
        cp = self._cp
        u_host = np.asarray(u, dtype=np.float64)
        b_host = np.asarray(b, dtype=np.float64)
        out = np.zeros(self.ndof, dtype=np.float64)

        for dev_id in self.device_ids:
            lo = self._sm_dof_lo[dev_id]
            sm_local_ndof = self._sm_local_ndof[dev_id]
            own_lo_ap, own_hi_ap = self._dof_lo[dev_id], self._dof_hi[dev_id]
            local_ndof = own_hi_ap - own_lo_ap
            off = self._sm_owned_off[dev_id]
            edof_d = self._sm_edof_dev[dev_id]
            ke_d = self._KE_dev[dev_id]
            evec_d = self._sm_evec_dev[dev_id]
            minv_d = self._sm_minv_dev[dev_id]
            free_d = self._sm_free_dev[dev_id]
            stream = self._streams[dev_id]
            u_pin = self._u_sm_pinned[dev_id]
            b_pin = self._b_sm_pinned[dev_id]
            out_pin = self._out_pinned[dev_id]     # sized local_ndof, reused
            with cp.cuda.Device(dev_id), stream:
                u_pin[:] = u_host[lo:lo + sm_local_ndof]
                b_pin[:] = b_host[lo:lo + sm_local_ndof]
                u_d = self._u_sm_dev[dev_id]
                b_d = self._b_sm_dev[dev_id]
                u_d.set(u_pin, stream=stream)
                b_d.set(b_pin, stream=stream)
                for _ in range(int(iters)):
                    ue = u_d[edof_d]
                    contrib = (ue @ ke_d.T) * evec_d[:, None]
                    Ku_d = cp.bincount(edof_d.ravel(), weights=contrib.ravel(),
                                        minlength=sm_local_ndof)
                    r_d = b_d - Ku_d
                    # free_d/minv_d are 0/False outside [off, off+local_ndof)
                    # (the true owned range) -- the ghost region's update
                    # term is always exactly zero, freezing it in place.
                    u_d = u_d + omega * cp.where(free_d, minv_d * r_d, 0.0)
                # Only the OWNED sub-range needs to come back -- ghost DOFs
                # were never updated and the neighbour will return its own
                # (now-exact) copy of any plane it owns.
                u_d[off:off + local_ndof].get(stream=stream, out=out_pin)
            # No synchronize here: same dispatch-all-then-sync-once pattern
            # as apply()/diagonal(), now amortized over `iters` sweeps
            # instead of just one matvec.

        for stream in self._streams:
            stream.synchronize()
        for dev_id in self.device_ids:
            own_lo_ap = self._dof_lo[dev_id]
            own_lo, own_hi = self._own_lo[dev_id], self._own_hi[dev_id]
            # Non-overlapping ownership write (see __init__/docstring above),
            # not the '+=' partial-sum merge apply()/diagonal() use.
            out[own_lo:own_hi] = self._out_pinned[dev_id][
                own_lo - own_lo_ap:own_hi - own_lo_ap]
        return out

    def apply(self, p):
        """Return K @ p (full ndof-length result). Only transfers each
        device's own DOF slice, not the full vector, and dispatches every
        device's H2D/compute/D2H onto its own stream before synchronizing
        (once, for all devices) at the end -- see module docstring.

        Every op below is dispatched inside ``with stream:``, not just the
        explicit ``.set()``/``.get()`` calls -- ``with cp.cuda.Device(...)``
        only selects the device, not the current stream, so ops issued
        without also entering the stream's context would land on cupy's
        default stream and race against the explicit H2D/D2H copies on the
        named stream."""
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
            with cp.cuda.Device(dev_id), stream:
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
        dispatch-all-then-synchronize-once pattern as apply() -- including
        the same ``with stream:`` fix, see apply()'s docstring."""
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
            with cp.cuda.Device(dev_id), stream:
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
