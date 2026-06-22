# SPDX-License-Identifier: GPL-3.0-or-later
"""Operators: list management, and the modal optimization run.

The heavy FEA + SIMP work runs in a background thread so the viewport stays
responsive; the modal timer (main thread) drains density snapshots from a queue
and builds the preview meshes (bpy must be touched only on the main thread).
"""

import queue
import threading

import numpy as np

import bpy
from bpy.types import Operator
from bpy.props import IntProperty, StringProperty, BoolProperty

from .core import voxelize, extract
from .core.simp import Problem, resample_density, resample_displacement


RESULTS_COLLECTION = "Blendtopo Results"

# In-process cache of the last finished finest level, for the Continue button.
_RESUME_CACHE = None
_RUN_COUNTER = 0


def _set_wire(obj):
    """Show an input object as wireframe so you can see/select inside it."""
    if obj is not None:
        try:
            obj.display_type = 'WIRE'
        except Exception:
            pass


def _results_collection(context):
    """Get (or create) the collection that holds per-iteration results."""
    coll = bpy.data.collections.get(RESULTS_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(RESULTS_COLLECTION)
        context.scene.collection.children.link(coll)
    return coll


# ---------------------------------------------------------------------------
# Assignment / list management
# ---------------------------------------------------------------------------

class TO_OT_set_build_space(Operator):
    bl_idname = "blendtopo.set_build_space"
    bl_label = "Set Build Space from Active"
    bl_description = "Use the active object as the build space (shown as wireframe)"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object is not a mesh")
            return {'CANCELLED'}
        context.scene.blendtopo.build_space = obj
        _set_wire(obj)
        return {'FINISHED'}


class _ListAdd(Operator):
    """Base: add the active object to a named collection."""
    collection_name = ""

    def execute(self, context):
        s = context.scene.blendtopo
        item = getattr(s, self.collection_name).add()
        if context.active_object and context.active_object.type == 'MESH':
            item.obj = context.active_object
            _set_wire(context.active_object)
        return {'FINISHED'}


class TO_OT_add_exclude(_ListAdd):
    bl_idname = "blendtopo.add_exclude"
    bl_label = "Add Exclusion"
    collection_name = "exclude"


class TO_OT_add_bearing(_ListAdd):
    bl_idname = "blendtopo.add_bearing"
    bl_label = "Add Bearing"
    collection_name = "bearings"


class TO_OT_add_load(_ListAdd):
    bl_idname = "blendtopo.add_load"
    bl_label = "Add Load"
    collection_name = "loads"


class TO_OT_remove_item(Operator):
    bl_idname = "blendtopo.remove_item"
    bl_label = "Remove"
    collection_name: StringProperty()
    index: IntProperty()

    def execute(self, context):
        s = context.scene.blendtopo
        coll = getattr(s, self.collection_name)
        if 0 <= self.index < len(coll):
            coll.remove(self.index)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# The modal optimization run (threaded solver)
# ---------------------------------------------------------------------------

class TO_OT_run(Operator):
    bl_idname = "blendtopo.run"
    bl_label = "Run Optimization"
    bl_description = "Voxelize, then optimize in the background. ESC keeps the latest result"

    resume: BoolProperty(default=False, options={'SKIP_SAVE'})

    @staticmethod
    def _level_resolutions(final_res, levels):
        """Geometric coarse->fine ladder, e.g. final=40 levels=3 -> [10,20,40]."""
        res = [max(4, int(round(final_res / (2 ** k))))
               for k in range(levels - 1, -1, -1)]
        seen, out = set(), []
        for r in res:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    def invoke(self, context, event):
        s = context.scene.blendtopo
        if s.build_space is None:
            self.report({'ERROR'}, "Set a build space first")
            return {'CANCELLED'}
        if len(s.loads) == 0 or len(s.bearings) == 0:
            self.report({'ERROR'}, "Need at least one bearing and one load")
            return {'CANCELLED'}

        global _RUN_COUNTER
        _RUN_COUNTER += 1
        self._run_id = _RUN_COUNTER
        self._settings = s
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._worker = None
        self._level_idx = 0
        self._grid = None
        self._last_rho = None
        self._last_obj_name = ""
        self._interrupted = False
        self._all_done = False
        self._error = ""
        self._coll = _results_collection(context)
        self._depsgraph = context.evaluated_depsgraph_get()

        if self.resume:
            if not _RESUME_CACHE:
                self.report({'ERROR'}, "Nothing to resume - run an optimization first")
                return {'CANCELLED'}
            self._prev_grid = _RESUME_CACHE['grid']
            self._prev_rho3d = _RESUME_CACHE['rho3d']
            self._prev_u = _RESUME_CACHE['u']
            self._last_obj_name = _RESUME_CACHE.get('last_obj', '')
            self._levels = [s.resolution]          # one more full pass at final res
        else:
            self._prev_grid = None
            self._prev_rho3d = None
            self._prev_u = None
            self._levels = self._level_resolutions(s.resolution, s.refine_levels)

        s.running = True
        s.current_iter = 0
        s.status = "resuming..." if self.resume else "voxelizing..."

        try:
            self._start_level(context)
        except Exception as exc:  # noqa: BLE001
            s.running = False
            s.status = "Stopped due to error: " + str(exc)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.03, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _start_level(self, context):
        """Main thread: voxelize this level (bpy), then spawn the solver thread."""
        s = self._settings
        res = self._levels[self._level_idx]
        grid = voxelize.build_grid(s, self._depsgraph, resolution=res)
        if grid.active.sum() == 0:
            raise RuntimeError("No active voxels - check build space / resolution")
        if grid.fixed_dofs.size == 0:
            raise RuntimeError("Bearings cover no grid nodes at this resolution")
        # Console diagnostics (Window > Toggle System Console) to localise issues.
        nact = int(grid.active.sum())
        ntot = grid.nx * grid.ny * grid.nz
        print(f"[Blendtopo] level res={res} grid={grid.nx}x{grid.ny}x{grid.nz} "
              f"active={nact}/{ntot} ({100.0*nact/ntot:.0f}%) "
              f"fixedDOFs={int(grid.fixed_dofs.size)} |F|={float(np.linalg.norm(grid.force)):.3g}")
        self._grid = grid

        x_init = None
        u_init = None
        if self._prev_rho3d is not None:
            x_init = resample_density(
                self._prev_rho3d, self._prev_grid.origin, self._prev_grid.vsize,
                grid.voxel_centers())
        if self._prev_u is not None and self._prev_grid is not None:
            pg = self._prev_grid
            cand = resample_displacement(
                self._prev_u, pg.origin, pg.vsize,
                (pg.nx + 1, pg.ny + 1, pg.nz + 1), grid.node_coords())
            # Only use it as a warm start if it is sane; a bad guess could only
            # slow CG, never corrupt the result, but be safe anyway.
            if np.all(np.isfinite(cand)):
                u_init = cand

        active3d = grid.active.reshape(grid.nx, grid.ny, grid.nz)
        prob = Problem(
            grid.nx, grid.ny, grid.nz, active3d, grid.fixed_dofs, grid.force,
            volfrac=s.volume_fraction, penalty=s.penalty,
            rmin=s.filter_radius, nu=s.poisson, e0=s.youngs_modulus,
            use_gpu=s.use_gpu, use_multigrid=s.use_multigrid,
        )

        li, res_v, n_iters = self._level_idx, res, s.iters_per_level
        # Final pass runs all iterations (no early stop) for a fully refined
        # result; coarse passes still early-stop on Convergence for speed.
        is_final = self._level_idx == len(self._levels) - 1
        tol = 0.0 if is_final else s.convergence_tol
        q, stop = self._q, self._stop

        def work():
            try:
                for it, comp, change, rho in prob.optimize(
                        max_iter=n_iters, x_init=x_init, tol=tol, u_init=u_init):
                    q.put(('iter', li, res_v, it, comp, change, rho))
                    if stop.is_set():
                        break
                q.put(('level_done', prob.last_u))
            except Exception as exc:  # noqa: BLE001
                q.put(('error', str(exc)))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def modal(self, context, event):
        s = context.scene.blendtopo

        if event.type == 'ESC' and event.value == 'PRESS':
            self._stop.set()
            self._interrupted = True
            s.status = "stopping (keeping latest)..."
            return {'RUNNING_MODAL'}

        if event.type == 'TIMER':
            try:
                self._drain(context)
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                self.report({'ERROR'}, str(exc))
                self._all_done = True
            if self._all_done:
                return self._finish(context)

        return {'PASS_THROUGH'}

    def _drain(self, context):
        s = context.scene.blendtopo
        n_levels = len(self._levels)
        redraw = False
        while True:
            try:
                msg = self._q.get_nowait()
            except queue.Empty:
                break
            tag = msg[0]
            if tag == 'iter':
                _, li, res, it, comp, change, rho = msg
                self._last_rho = rho
                s.current_iter = it
                s.status = (f"L{li + 1}/{n_levels} ({res}^3)  it {it}  "
                            f"C={comp:.3e}  d={change:.3f}")
                self._emit_result(context, rho, li, it)
                redraw = True
            elif tag == 'error':
                raise RuntimeError(msg[1])
            elif tag == 'level_done':
                self._prev_u = msg[1] if len(msg) > 1 else None
                self._on_level_done(context)
        if redraw:
            for area in context.screen.areas:
                area.tag_redraw()

    def _on_level_done(self, context):
        self._prev_grid = self._grid
        if self._last_rho is not None:
            g = self._grid
            self._prev_rho3d = self._last_rho.reshape(g.nx, g.ny, g.nz)
        if self._interrupted:
            self._all_done = True
            return
        self._level_idx += 1
        if self._level_idx >= len(self._levels):
            self._all_done = True
        else:
            context.scene.blendtopo.status = "voxelizing next level..."
            self._start_level(context)

    def _emit_result(self, context, rho, level, it):
        s = context.scene.blendtopo
        g = self._grid
        density3d = rho.reshape(g.nx, g.ny, g.nz)
        prev_name = self._last_obj_name
        if s.keep_steps:
            name = f"Blendtopo_r{self._run_id}_L{level + 1}_it{it:03d}"
        else:
            name = "Blendtopo_Result"
        obj = extract.density_to_object(
            density3d, s.iso_level, g.origin, g.vsize, name,
            style=s.preview_style, collection=self._coll)

        # Early iterations (uniform low density) can mesh to nothing. Don't let
        # that blank the viewport: discard the empty step and keep the previous
        # result visible until a real shape exists.
        has_geometry = len(obj.data.polygons) > 0
        if not has_geometry and s.keep_steps and prev_name and prev_name != obj.name:
            me = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if me.users == 0:
                bpy.data.meshes.remove(me)
            return

        # New step has geometry: now hide the previous (morph-in-place effect).
        # Kept (not deleted) so you can scrub/reveal them afterwards.
        if has_geometry and s.keep_steps and prev_name and prev_name != obj.name:
            prev = bpy.data.objects.get(prev_name)
            if prev is not None:
                try:
                    prev.hide_set(True)
                except Exception:
                    prev.hide_viewport = True
        self._last_obj_name = obj.name

    def _finish(self, context):
        s = context.scene.blendtopo
        wm = context.window_manager
        self._stop.set()
        if getattr(self, "_timer", None) is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        s.running = False
        obj = bpy.data.objects.get(self._last_obj_name)
        if obj is not None and not self._interrupted and not self._error:
            # Guarantee a single watertight shell on the finished result.
            extract.finalize_watertight(obj)
            vox = self._grid.vsize * 0.75 if s.preview_style == 'BLOCKY' else None
            extract.apply_remesh_smooth(
                obj, voxel_size=vox, smooth_iters=s.smooth_iterations)
        if self._error:
            s.status = "Stopped due to error: " + self._error
        elif self._interrupted:
            s.status = "stopped (result kept)"
        else:
            s.status = f"done at iter {s.current_iter}"
        # Cache the finest result so "Continue" can resume more iterations.
        global _RESUME_CACHE
        if (not self._error and self._prev_grid is not None
                and self._prev_rho3d is not None):
            _RESUME_CACHE = {'grid': self._prev_grid,
                             'rho3d': self._prev_rho3d, 'u': self._prev_u,
                             'last_obj': self._last_obj_name}
        self.report({'INFO'}, s.status)
        for area in context.screen.areas:
            area.tag_redraw()
        return {'FINISHED'}



class TO_OT_optimize_mesh(Operator):
    bl_idname = "blendtopo.optimize_mesh"
    bl_label = "Optimize Mesh"
    bl_description = ("Make a clean low-poly copy of the active result: "
                     "smooth (x5), subdivide (x2), decimate to 10%")

    def execute(self, context):
        src = context.active_object
        if src is None or src.type != 'MESH':
            self.report({'ERROR'}, "Select a result mesh first")
            return {'CANCELLED'}

        new = src.copy()
        new.data = src.data.copy()
        new.name = src.name + "_opt"
        for c in src.users_collection:
            c.objects.link(new)

        s = context.scene.blendtopo
        if s.opt_smooth_iters > 0:
            sm = new.modifiers.new("TO_Smooth", 'SMOOTH')
            sm.iterations = s.opt_smooth_iters
            sm.factor = 0.5
        dec = new.modifiers.new("TO_Decimate", 'DECIMATE')
        dec.decimate_type = 'COLLAPSE'
        dec.ratio = s.opt_decimate_ratio

        try:
            context.view_layer.update()
            dg = context.evaluated_depsgraph_get()
            baked = bpy.data.meshes.new_from_object(new.evaluated_get(dg))
        except Exception as exc:  # noqa: BLE001
            self.report({'ERROR'}, f"Bake failed: {exc}")
            return {'CANCELLED'}

        new.modifiers.clear()
        old = new.data
        new.data = baked
        if old.users == 0:
            bpy.data.meshes.remove(old)
        new.display_type = 'TEXTURED'

        for o in context.selected_objects:
            o.select_set(False)
        new.select_set(True)
        context.view_layer.objects.active = new
        src.hide_set(True)
        self.report({'INFO'}, f"Optimized: {len(baked.vertices)} verts "
                              f"(from {len(src.data.vertices)})")
        return {'FINISHED'}


_CLASSES = (
    TO_OT_set_build_space,
    TO_OT_add_exclude, TO_OT_add_bearing, TO_OT_add_load,
    TO_OT_remove_item, TO_OT_run, TO_OT_optimize_mesh,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
