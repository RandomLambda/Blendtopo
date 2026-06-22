# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene-level settings and the lists of bearings / loads / exclusions.

Every property carries a rich `description=` - Blender shows it as the tooltip
when you hover the field, so beginners get what-it-is / what-it-does / how-to-tune.
"""

import bpy
from bpy.props import (
    PointerProperty, CollectionProperty, IntProperty, FloatProperty,
    FloatVectorProperty, BoolProperty, StringProperty, EnumProperty,
)
from bpy.types import PropertyGroup


def _is_mesh(self, obj):
    return obj.type == 'MESH'


class TO_ExcludeItem(PropertyGroup):
    """A keep-out region: voxels inside this object are forced empty."""
    obj: PointerProperty(
        type=bpy.types.Object, poll=_is_mesh,
        name="Keep-out mesh",
        description="A region that must stay empty (e.g. a bolt hole or "
                    "clearance zone). Any voxel inside this mesh is removed "
                    "from the design. Use a simple closed mesh that overlaps "
                    "the build space",
    )


class TO_Bearing(PropertyGroup):
    """A support: voxels inside this object get their DOFs fixed."""
    obj: PointerProperty(
        type=bpy.types.Object, poll=_is_mesh,
        name="Support mesh",
        description="Where the part is held / anchored. Grid nodes inside this "
                    "mesh are clamped, so material tends to grow toward it. "
                    "Place a small closed mesh overlapping the build space",
    )
    fix_x: BoolProperty(
        name="X", default=True,
        description="Prevent movement along X at this support. Turn off to "
                    "allow the part to slide along X here (e.g. a roller)")
    fix_y: BoolProperty(
        name="Y", default=True,
        description="Prevent movement along Y at this support")
    fix_z: BoolProperty(
        name="Z", default=True,
        description="Prevent movement along Z at this support")


class TO_Load(PropertyGroup):
    """A load: force vector applied to voxels inside this object."""
    obj: PointerProperty(
        type=bpy.types.Object, poll=_is_mesh,
        name="Load mesh",
        description="Where an external force is applied. The force is spread "
                    "over the grid nodes inside this mesh. Place a small closed "
                    "mesh overlapping the build space where the load acts",
    )
    force: FloatVectorProperty(
        name="Force", subtype='XYZ', size=3, default=(0.0, 0.0, -1.0),
        description="Force direction and magnitude (X, Y, Z). Only the relative "
                    "sizes between loads matter for the resulting shape. "
                    "Example: (0, 0, -1) pushes straight down",
    )


class TO_Settings(PropertyGroup):
    # --- Geometry inputs ---
    build_space: PointerProperty(
        name="Build Space", type=bpy.types.Object, poll=_is_mesh,
        description="The volume the optimizer is allowed to fill with material. "
                    "Pick a closed mesh; the result is generated inside it. "
                    "Shown as wireframe so you can see the result form inside",
    )
    exclude: CollectionProperty(type=TO_ExcludeItem)
    exclude_index: IntProperty(default=0)
    bearings: CollectionProperty(type=TO_Bearing)
    bearings_index: IntProperty(default=0)
    loads: CollectionProperty(type=TO_Load)
    loads_index: IntProperty(default=0)

    # --- Discretization (coarse-to-fine) ---
    resolution: IntProperty(
        name="Final Resolution", default=40, min=4, soft_max=160,
        description="Detail level: voxels along the longest edge at the finest "
                    "stage. Higher = finer features but much slower (cost grows "
                    "roughly with the cube). Start ~32-48 and raise if needed",
    )
    refine_levels: IntProperty(
        name="Refine Levels", default=3, min=1, max=6,
        description="How many coarse-to-fine stages to run. Each stage doubles "
                    "the grid and warm-starts from the previous result, so it "
                    "reaches detail fast. 3 is a good default; raise for very "
                    "high final resolution",
    )
    iters_per_level: IntProperty(
        name="Iterations / Level", default=20, min=1, max=200,
        description="Maximum optimization steps per stage (it stops earlier if "
                    "it converges - see Convergence). More = more refined but "
                    "slower. 10-20 is usually plenty",
    )
    convergence_tol: FloatProperty(
        name="Convergence", default=0.1, min=0.0005, max=0.2, precision=4,
        description="Auto-advance to the next stage once the shape stops "
                    "changing. It is the largest per-step density change "
                    "allowed before moving on. Lower = run longer/more refined; "
                    "higher = finish sooner",
    )
    use_gpu: BoolProperty(
        name="Use GPU if available", default=True,
        description="Run the solver on the graphics card via CuPy when it is "
                    "installed and a CUDA GPU is present; otherwise uses the "
                    "CPU automatically. See the GPU line below for status",
    )
    use_multigrid: BoolProperty(
        name="Multigrid solver", default=True,
        description="Use a geometric-multigrid preconditioner for the FEA "
                    "solve. Much faster on fine grids; falls back to the plain "
                    "solver automatically if it cannot be built",
    )

    # --- Optimization targets ---
    volume_fraction: FloatProperty(
        name="Volume Fraction", default=0.2, min=0.05, max=1.0,
        description="How much of the build space to keep as material (0.3 = "
                    "30%). Lower = lighter, more skeletal; higher = bulkier "
                    "and stiffer. This is the main weight/stiffness trade-off",
    )
    penalty: FloatProperty(
        name="Penalty (p)", default=3.0, min=1.0, max=6.0,
        description="Pushes the result toward clean solid-or-empty material "
                    "instead of grey in-between. 3 is standard. Raise slightly "
                    "for crisper structures; too high can trap a bad shape",
    )
    filter_radius: FloatProperty(
        name="Filter Radius", default=1.5, min=1.0, max=6.0,
        description="Minimum feature size, in voxels. Controls how thin struts "
                    "can get and prevents checkerboard artefacts. DECREASE for "
                    "finer/thinner features; INCREASE for chunkier, smoother "
                    "members",
    )

    # --- Material (relative; only ratios affect the result shape) ---
    youngs_modulus: FloatProperty(
        name="E", default=1.0, min=1e-6,
        description="Material stiffness (Young's modulus). Only relative values "
                    "matter for the shape, so the default of 1.0 is fine unless "
                    "mixing materials")
    poisson: FloatProperty(
        name="Poisson", default=0.3, min=0.0, max=0.49,
        description="Poisson's ratio of the material (how much it bulges "
                    "sideways when compressed). 0.3 suits most metals/plastics")

    # --- Output / progressive remesh ---
    iso_level: FloatProperty(
        name="Iso Level", default=0.5, min=0.05, max=0.95,
        description="Density threshold that defines the visible surface. Lower "
                    "= keep more material (thicker, captures faint struts); "
                    "higher = only the densest core remains",
    )
    preview_style: EnumProperty(
        name="Preview", default='SMOOTH',
        items=[('SMOOTH', "Smooth", "Smooth iso-surface (surface nets) - the "
                                    "organic look, best for the final shape"),
               ('BLOCKY', "Blocky", "Raw voxel cubes - faster to build, useful "
                                     "for a quick rough glance")],
        description="How each result mesh is built. Smooth follows the density "
                    "field organically; Blocky shows the raw voxels",
    )
    keep_steps: BoolProperty(
        name="Keep Each Step", default=True,
        description="Save a separate object for every iteration in the 'Blendtopo "
                    "Results' collection (you can scrub/compare them). Turn off "
                    "to keep only one continuously-updated result object",
    )
    smooth_iterations: IntProperty(
        name="Smooth", default=5, min=0, max=50,
        description="Extra smoothing passes applied to the final result. Higher "
                    "= smoother but can shrink thin features. Set 0 to keep the "
                    "raw surface",
    )

    # --- Optimize Mesh (post-process) ---
    opt_smooth_iters: IntProperty(
        name="Opt Smooth", default=1, min=0, max=20,
        description="Smoothing passes used by 'Optimize Mesh'. 1 is gentle; "
                    "raise for a softer surface (can shrink thin features)",
    )
    opt_decimate_ratio: FloatProperty(
        name="Opt Decimate", default=0.1, min=0.01, max=1.0,
        description="Target fraction of faces kept by 'Optimize Mesh'. 0.1 = "
                    "keep 10%. Lower = lighter mesh; 1.0 = no reduction",
    )

    # --- Runtime state (not user-facing) ---
    running: BoolProperty(default=False)
    current_iter: IntProperty(default=0)
    status: StringProperty(default="")


_CLASSES = (TO_ExcludeItem, TO_Bearing, TO_Load, TO_Settings)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.blendtopo = PointerProperty(type=TO_Settings)


def unregister():
    del bpy.types.Scene.blendtopo
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
