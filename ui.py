# SPDX-License-Identifier: GPL-3.0-or-later
"""Sidebar UI for Blendtopo (View3D > N panel)."""

import os

import bpy
from bpy.types import Panel

from .core.backend import is_gpu_build, gpu_status, gpu_usable, gpu_device_count


class TO_PT_base:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Blendtopo"


class TO_PT_main(TO_PT_base, Panel):
    bl_idname = "TO_PT_main"
    bl_label = "Blendtopo"

    def draw(self, context):
        layout = self.layout
        s = context.scene.blendtopo

        col = layout.column(align=True)
        col.prop(s, "build_space", text="Build")
        col.operator("blendtopo.set_build_space", icon='MESH_CUBE')

        box = layout.box()
        box.label(text="Exclusions", icon='GHOST_ENABLED')
        box.operator("blendtopo.add_exclude", icon='ADD')
        for i, item in enumerate(s.exclude):
            row = box.row(align=True)
            row.prop(item, "obj", text="")
            op = row.operator("blendtopo.remove_item", text="", icon='X')
            op.collection_name = "exclude"
            op.index = i


class TO_PT_bc(TO_PT_base, Panel):
    bl_idname = "TO_PT_bc"
    bl_parent_id = "TO_PT_main"
    bl_label = "Bearings & Loads"

    def draw(self, context):
        layout = self.layout
        s = context.scene.blendtopo

        box = layout.box()
        box.label(text="Bearings (supports)", icon='CON_PIVOT')
        box.operator("blendtopo.add_bearing", icon='ADD')
        for i, b in enumerate(s.bearings):
            col = box.column(align=True)
            row = col.row(align=True)
            row.prop(b, "obj", text="")
            op = row.operator("blendtopo.remove_item", text="", icon='X')
            op.collection_name = "bearings"
            op.index = i
            r2 = col.row(align=True)
            r2.prop(b, "fix_x")
            r2.prop(b, "fix_y")
            r2.prop(b, "fix_z")

        box = layout.box()
        box.label(text="Loads (forces)", icon='FORCE_FORCE')
        box.operator("blendtopo.add_load", icon='ADD')
        for i, ld in enumerate(s.loads):
            col = box.column(align=True)
            row = col.row(align=True)
            row.prop(ld, "obj", text="")
            op = row.operator("blendtopo.remove_item", text="", icon='X')
            op.collection_name = "loads"
            op.index = i
            col.prop(ld, "force", text="")


class TO_PT_settings(TO_PT_base, Panel):
    bl_idname = "TO_PT_settings"
    bl_parent_id = "TO_PT_main"
    bl_label = "Settings"

    def draw(self, context):
        layout = self.layout
        s = context.scene.blendtopo

        col = layout.column(align=True)
        col.prop(s, "resolution")
        col.prop(s, "refine_levels")
        col.prop(s, "iters_per_level")
        col.prop(s, "convergence_tol")
        col.prop(s, "volume_fraction", slider=True)
        col.prop(s, "use_multigrid")

        box = layout.box()
        header = box.row(align=True)
        header.prop(s, "show_compute_advanced",
                    icon='TRIA_DOWN' if s.show_compute_advanced else 'TRIA_RIGHT',
                    icon_only=True, emboss=False)
        header.prop(s, "compute_mode")
        if s.show_compute_advanced:
            col = box.column(align=True)
            threads_text = ("CPU Threads: 0 (auto)" if s.cpu_threads == 0
                            else "CPU Threads")
            col.prop(s, "cpu_threads", text=threads_text)
            cores = os.cpu_count() or 1
            col.label(text=f"CPU: {cores} logical cores detected",
                      icon='INFO')
            # GPU status is only ever meaningful in the GPU edition -- the
            # CPU/hosted build never shows anything GPU-related here.
            if is_gpu_build():
                n_gpu = gpu_device_count() if gpu_usable() else 0
                col.label(text=gpu_status(),
                          icon='CHECKMARK' if gpu_usable() else 'INFO')
                if gpu_usable():
                    col.label(text=f"GPU devices visible: {n_gpu}")
            col.prop(s, "verbose_log")

        col = layout.column(align=True)
        col.prop(s, "penalty")
        col.prop(s, "filter_radius")
        col.prop(s, "iso_level", slider=True)
        col.prop(s, "preview_style")

        col = layout.column(align=True)
        col.prop(s, "youngs_modulus")
        col.prop(s, "poisson")

        col = layout.column(align=True)
        col.prop(s, "keep_steps")
        col.prop(s, "smooth_iterations")


class TO_PT_run(TO_PT_base, Panel):
    bl_idname = "TO_PT_run"
    bl_parent_id = "TO_PT_main"
    bl_label = "Run"

    def draw(self, context):
        layout = self.layout
        s = context.scene.blendtopo

        row = layout.row()
        row.scale_y = 1.5
        row.enabled = not s.running
        op = row.operator("blendtopo.run", icon='PLAY', text="Run Optimization")
        op.resume = False

        row = layout.row()
        row.enabled = not s.running
        op = row.operator("blendtopo.run", icon='TRACKING_FORWARDS',
                          text="Continue (more iterations)")
        op.resume = True

        if s.running:
            layout.label(text="Running - press ESC to stop & keep", icon='REC')
        if s.status:
            layout.label(text=s.status)

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Post-process:")
        col.prop(s, "opt_smooth_iters")
        col.prop(s, "opt_decimate_ratio")
        col.operator("blendtopo.optimize_mesh", icon='MOD_SMOOTH',
                     text="Optimize Mesh (clean low-poly)")


_CLASSES = (TO_PT_main, TO_PT_bc, TO_PT_settings, TO_PT_run)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    # Guard against a class never having been registered (e.g. register()
    # aborted partway through -- Blender then still calls unregister() on
    # every module as cleanup, which would otherwise raise "missing bl_rna
    # attribute ... may not be registered").
    for cls in reversed(_CLASSES):
        if hasattr(cls, "bl_rna"):
            bpy.utils.unregister_class(cls)
