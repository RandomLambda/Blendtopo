# SPDX-License-Identifier: GPL-3.0-or-later
"""Compute-backend selection -- CPU edition.

This is the CPU-only backend that ships with the hosted extension. It imports
nothing beyond numpy, so the add-on is fully self-contained and needs no Python
wheels.

The separate GPU edition replaces *this file* with ``backend_gpu.py`` (done by
``build_extension.py``), which adds an optional Cu-Py/CUDA path. Every other
module is identical and simply asks this backend which array library to use, so
there is a single source of truth for the actual algorithms.
"""

import numpy as np

# True only in the GPU edition. Lets the UI hide the GPU controls on the CPU
# build without any per-build code edits.
GPU_BUILD = False


def is_gpu_build():
    return GPU_BUILD


def get_xp(use_gpu=False):
    """Array module to compute with. CPU edition: always numpy."""
    return np


def asnumpy(a):
    """Bring an array back to host numpy (a no-op here)."""
    return np.asarray(a)


def gpu_usable():
    return False


def gpu_device_count():
    """Number of usable CUDA devices. Always 0 in the CPU edition."""
    return 0


def gpu_status():
    return "CPU edition (numpy)"
