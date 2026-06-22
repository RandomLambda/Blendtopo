# SPDX-License-Identifier: GPL-3.0-or-later
"""
Blendtopo - Topology optimization for solid mesh parts inside Blender.

Voxelize a build space, run a SIMP optimization with a matrix-free FEA solver
in a coarse-to-fine schedule, and rebuild a mesh after every step so the user
can stop at any time and keep the latest result.
"""

bl_info = {
    "name": "Blendtopo - Topology Optimization",
    "author": "Philip + Claude",
    "version": (0, 8, 1),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > Blendtopo",
    "description": "Topology optimization on solid mesh parts with build space, "
                   "exclusions, bearings and loads. Coarse-to-fine remeshing.",
    "category": "Mesh",
}

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
