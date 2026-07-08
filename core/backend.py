# SPDX-License-Identifier: GPL-3.0-or-later
"""Compute-backend selection -- GPU edition (optional Cu-Py/CUDA path).

This file is shipped (renamed to ``backend.py``) ONLY in the separate GPU
edition built by ``build_extension.py``. It is never part of the hosted
extension, because Cu-Py cannot be bundled within the Extensions Platform size
limit.

Cu-Py is NOT bundled: the user installs it themselves into Blender's Python, e.g.

    "<blender>/4.x/python/bin/python" -m pip install cupy-cuda12x

If Cu-Py is missing, no CUDA device is present, or a quick self-test disagrees
with the CPU, every call transparently falls back to numpy -- so results are
identical to the CPU edition, just slower.
"""

import numpy as np

GPU_BUILD = True

_GPU_OK = None


def is_gpu_build():
    return GPU_BUILD


def _cupy():
    """Return the Cu-Py module if a CUDA device is usable, else None."""
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
        return cp
    except Exception:
        return None


def _self_test():
    """Check CPU/GPU agreement on the two ops the solver actually uses
    (matmul and bincount), so a flaky driver can never silently corrupt a
    result -- if anything is off we stay on the CPU."""
    cp = _cupy()
    if cp is None:
        return False
    try:
        a = np.random.rand(128, 24)
        b = np.random.rand(24, 24)
        if not np.allclose(a @ b, cp.asnumpy(cp.asarray(a) @ cp.asarray(b)),
                           rtol=1e-4, atol=1e-9):
            return False
        idx = np.random.randint(0, 50, size=512)
        w = np.random.rand(512)
        cpu = np.bincount(idx, weights=w, minlength=50)
        gpu = cp.asnumpy(cp.bincount(cp.asarray(idx), weights=cp.asarray(w),
                                     minlength=50))
        return bool(np.allclose(cpu, gpu, rtol=1e-4, atol=1e-9))
    except Exception:
        return False


def gpu_usable():
    """Cached: True only if Cu-Py is present AND passes the self-test."""
    global _GPU_OK
    if _GPU_OK is None:
        _GPU_OK = _self_test()
    return _GPU_OK


def gpu_device_count():
    """Number of CUDA devices visible to Cu-Py (0 if Cu-Py/CUDA unusable).

    Only meaningful once :func:`gpu_usable` is True -- callers should check
    that first, since this does not repeat the self-test.
    """
    cp = _cupy()
    if cp is None:
        return 0
    try:
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def get_xp(use_gpu=False):
    """Return Cu-Py when the GPU is requested and validated, else numpy."""
    if use_gpu and gpu_usable():
        cp = _cupy()
        if cp is not None:
            return cp
    return np


def asnumpy(a):
    """Bring an array back to host numpy, whether it is a Cu-Py or numpy array."""
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy as cp
        return cp.asnumpy(a)
    return np.asarray(a)


def gpu_status():
    if _cupy() is None:
        return "GPU: Cu-Py not installed - using CPU"
    if gpu_usable():
        return "GPU: Cu-Py ready (validated)"
    return "GPU: self-test failed - using CPU"
