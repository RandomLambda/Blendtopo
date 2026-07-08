# SPDX-License-Identifier: GPL-3.0-or-later
"""
Multi-process CPU matvec for the matrix-free FEA (stdlib only, no wheel).

The dominant cost of one CG iteration is ``K(rho) @ p``: for every element,
gather its 24 nodal values, multiply by the (shared, precomputed) 24x24
element stiffness matrix, scale by that element's density, and scatter-add
into the global vector (see ``fea.VoxelFEA._apply_K`` / ``multigrid.py``'s
``_apply``). Because the scatter-add only ever *writes by index* into the
output vector, this splits perfectly across independent workers by simply
partitioning the ELEMENT list into contiguous chunks -- no spatial halo
exchange is needed at all: each worker computes a partial output vector (the
same full length as the real one, zero everywhere it didn't touch) and the
partials are summed. That sum is exactly what a single-process bincount over
all elements would have produced, so this is not an approximation.

Everything that is O(ndof) (the CG scalar recurrence: dot products, vector
adds, the preconditioner apply) stays on the calling process, unparallelized
-- it is cheap. Only the O(nelem * 24) gather-matmul-scatter is farmed out.

Implementation notes:
  * Workers are persistent (spawned once per solve, not per iteration) and
    communicate through ``multiprocessing.shared_memory`` blocks, so the
    large arrays (`p`, per-worker output, per-worker density) are never
    pickled -- only a tiny "go do this op" signal crosses per call.
  * Uses the 'spawn' start method everywhere (safe on Windows/macOS, and
    avoids ever forking a process that has bpy loaded -- this module is only
    ever imported inside the already-isolated solver subprocess, but 'spawn'
    is used regardless to keep behaviour identical on every platform).
  * Correctness of the partitioning is covered by tests/test_core.py, which
    checks the pooled result against plain single-process bincount.
"""

import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

_CMD_APPLY = 1
_CMD_DIAG = 2
_CMD_STOP = -1


def _worker_main(edof_chunk, KE, ndof, p_shm_name, out_shm_name,
                  evec_shm_name, n_elem_chunk, cmd, evt_go, evt_done):
    """Runs in the child process. Loops: wait -> do one op -> signal done."""
    p_shm = shared_memory.SharedMemory(name=p_shm_name)
    out_shm = shared_memory.SharedMemory(name=out_shm_name)
    evec_shm = shared_memory.SharedMemory(name=evec_shm_name)
    try:
        p = np.ndarray((ndof,), dtype=np.float64, buffer=p_shm.buf)
        out = np.ndarray((ndof,), dtype=np.float64, buffer=out_shm.buf)
        evec = np.ndarray((n_elem_chunk,), dtype=np.float64, buffer=evec_shm.buf)
        diagKE = np.diag(KE)
        edof_flat = edof_chunk.ravel()

        while True:
            evt_go.wait()
            evt_go.clear()
            op = cmd.value
            if op == _CMD_STOP:
                break
            if op == _CMD_APPLY:
                ue = p[edof_chunk]
                contrib = (ue @ KE.T) * evec[:, None]
            elif op == _CMD_DIAG:
                contrib = evec[:, None] * diagKE[None, :]
            else:
                evt_done.set()
                continue
            out[:] = np.bincount(edof_flat, weights=contrib.ravel(),
                                 minlength=ndof)
            evt_done.set()
    finally:
        p_shm.close()
        out_shm.close()
        evec_shm.close()


class CPUMatVecPool:
    """Element-partitioned matrix-free matvec across persistent worker
    processes. ``edof`` and ``KE`` follow the exact conventions already used
    by ``fea.VoxelFEA`` (reduced/active DOF space) and ``multigrid.MGSolver``
    level 0 (full DOF space) -- both can use this unchanged."""

    def __init__(self, edof, KE, ndof, n_workers, verbose=False):
        edof = np.ascontiguousarray(edof, dtype=np.int64)
        KE = np.ascontiguousarray(KE, dtype=np.float64)
        nelem = edof.shape[0]
        n_workers = max(1, int(n_workers))
        bounds = np.linspace(0, nelem, n_workers + 1).astype(np.int64)
        chunks = [(int(bounds[i]), int(bounds[i + 1]))
                  for i in range(n_workers) if bounds[i + 1] > bounds[i]]
        if len(chunks) < 2:
            raise ValueError(
                "CPUMatVecPool needs at least 2 non-empty element chunks "
                f"(got nelem={nelem}, n_workers={n_workers})")

        self.ndof = int(ndof)
        self.n_workers = len(chunks)
        self.verbose = verbose
        self._closed = False

        ctx = mp.get_context("spawn")
        self._p_shm = shared_memory.SharedMemory(create=True,
                                                  size=self.ndof * 8)
        self._p = np.ndarray((self.ndof,), dtype=np.float64,
                              buffer=self._p_shm.buf)

        self._out_shms, self._outs = [], []
        self._evec_shms, self._evecs = [], []
        self._cmds, self._evt_go, self._evt_done = [], [], []
        self._procs = []

        for a, b in chunks:
            n = b - a
            oshm = shared_memory.SharedMemory(create=True, size=self.ndof * 8)
            eshm = shared_memory.SharedMemory(create=True, size=n * 8)
            self._out_shms.append(oshm)
            self._evec_shms.append(eshm)
            self._outs.append(np.ndarray((self.ndof,), dtype=np.float64,
                                         buffer=oshm.buf))
            self._evecs.append(np.ndarray((n,), dtype=np.float64,
                                          buffer=eshm.buf))

            cmd = ctx.Value("i", 0)
            evt_go = ctx.Event()
            evt_done = ctx.Event()
            proc = ctx.Process(
                target=_worker_main,
                args=(edof[a:b], KE, self.ndof, self._p_shm.name, oshm.name,
                      eshm.name, n, cmd, evt_go, evt_done),
                daemon=True,
            )
            proc.start()
            self._cmds.append(cmd)
            self._evt_go.append(evt_go)
            self._evt_done.append(evt_done)
            self._procs.append(proc)

        if verbose:
            print(f"[Blendtopo] CPUMatVecPool: {self.n_workers} worker "
                  f"processes over {nelem} elements ({self.ndof} DOF)")

    def _run(self, op):
        for cmd, evt in zip(self._cmds, self._evt_go):
            cmd.value = op
        for evt in self._evt_go:
            evt.set()
        for evt in self._evt_done:
            evt.wait()
            evt.clear()

    def set_density(self, evec):
        """evec: (nelem,) per-element scale, same order as the edof passed
        to __init__ (i.e. the caller's active-element or full-element
        ordering -- not remapped here)."""
        evec = np.asarray(evec, dtype=np.float64).ravel()
        pos = 0
        for e in self._evecs:
            n = e.shape[0]
            e[:] = evec[pos:pos + n]
            pos += n

    def apply(self, p):
        """Return K @ p (full ndof-length result)."""
        self._p[:] = np.asarray(p, dtype=np.float64)
        self._run(_CMD_APPLY)
        out = self._outs[0].copy()
        for o in self._outs[1:]:
            out += o
        return out

    def diagonal(self):
        """Return the (unclamped) diagonal contribution vector."""
        self._run(_CMD_DIAG)
        out = self._outs[0].copy()
        for o in self._outs[1:]:
            out += o
        return out

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            for cmd, evt in zip(self._cmds, self._evt_go):
                cmd.value = _CMD_STOP
            for evt in self._evt_go:
                evt.set()
            for proc in self._procs:
                proc.join(timeout=2.0)
                if proc.is_alive():
                    proc.terminate()
        finally:
            for shm in (self._p_shm, *self._out_shms, *self._evec_shms):
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
