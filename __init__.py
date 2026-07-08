# SPDX-License-Identifier: GPL-3.0-or-later
"""
Blendtopo - Topology optimization for solid mesh parts inside Blender.

Voxelize a build space, run a SIMP optimization with a matrix-free FEA solver
in a coarse-to-fine schedule, and rebuild a mesh after every step so the user
can stop at any time and keep the latest result.
"""

from . import properties
from . import operators
from . import ui

_MODULES = (properties, operators, ui)


def register():
    for mod in _MODULES:
        mod.register()


def unregister():
    for mod in reversed(_MODULES):
        mod.unregister()


if __name__ == "__main__":
    register()
