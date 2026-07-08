# SPDX-License-Identifier: GPL-3.0-or-later
"""Operators: list management, and the modal optimization run.

Preferred path: voxelization, the FEA/SIMP solve and per-iteration meshing for
a whole level run in a plain-Python subprocess (see core.solver_worker); the
modal timer just polls it and turns finished verts/faces into a Blender
object, so Blender's main thread does the bare minimum and the viewport stays
interactive even on fine grids.

Fallback (no subprocess available): both heavy phases run as cooperative
generators advanced by the modal timer instead, so no single tick blocks the
UI:

* voxelizing a level (per-point ray-casts) is drained under a per-tick time
  budget, showing a percentage instead of freezing while a finer grid is built;
* the SIMP solve yields one iteration per tick, rebuilding the preview mesh.

Either way, no background threads are used (threads are a known Blender crash
source) - only subprocesses, which cannot touch or crash Blender state.
"""

import time

import numpy as np

import bpy
from bpy.types import Operator
from bpy.props import IntProperty, StringProperty, BoolProperty

from .core import voxelize, extract, solver_worker
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
        self._gen = None          # active SIMP generator for the current level
        self._prob = None         # active Problem (for last_u after a level)
        self._voxgen = None       # active in-process voxelization generator
        self._voxizer = None      # active out-of-process voxelizer (in-proc mode)
        self._solver = None       # active unified solver subprocess (preferred)
        self._stop_requested = False
        self._phase = 'voxel'     # 'voxel' (building grid) or 'solve' (in-proc)
        self._level_idx = 0
        self._grid = None
        self._last_rho = None
        self._last_vsize = None
        self._last_obj_name = ""
        self._interrupted = False
        self._all_done = False
        self._error = ""
        self._cache_final = None  # raw fields for the Continue cache
        self._coll = _results_collection(context)
        self._depsgraph = context.evaluated_depsgraph_get()

        # Preferred path: run each whole level (voxelize + FEA/SIMP + meshing)
        # in a subprocess so Blender's main thread does the bare minimum. Falls
        # back to the in-process generators if no subprocess can be launched.
        self._mode = 'solver' if solver_worker.solver_available() else 'inproc'

        if self.resume:
            if not _RESUME_CACHE:
                self.report({'ERROR'}, "Nothing to resume - run an optimization first")
                return {'CANCELLED'}
            rc = _RESUME_CACHE
            self._last_obj_name = rc.get('last_obj', '')
            self._levels = [s.resolution]          # one more full pass at final res
            # Warm-start dict for the subprocess solver.
            self._prev = {'prev_rho3d': rc['rho3d'], 'prev_origin': rc['origin'],
                          'prev_vsize': rc['vsize'], 'prev_u': rc['u'],
                          'prev_node_dims': rc['node_dims']}
            # Equivalent fields for the in-process fallback.
            self._prev_grid = voxelize.Grid(rc['dims'], rc['origin'], rc['vsize'])
            self._prev_rho3d = rc['rho3d']
            self._prev_u = rc['u']
        else:
            self._prev = None
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
        """Begin a level.

        Preferred ('solver') mode: hand the *entire* level - voxelization, FEA/
        SIMP and per-iteration meshing - to a subprocess. The main thread then
        only polls and turns finished verts/faces into a Blender object, so the
        viewport never freezes. If no subprocess can be launched we fall back to
        the in-process two-phase generators ('inproc').
        """
        s = self._settings
        res = self._levels[self._level_idx]
        self._cur_res = res

        if self._mode == 'solver':
            try:
                job = self._build_solver_job(context, res)
                self._solver = solver_worker.SolverClient()
                self._solver.start(job)
                context.scene.blendtopo.status = (
                    f"voxelizing L{self._level_idx + 1}/{len(self._levels)} "
                    f"({res}^3)... [bg]")
                return
            except Exception as exc:  # noqa: BLE001 - degrade, don't fail
                print(f"[Blendtopo] subprocess solver unavailable, using "
                      f"in-process path: {exc}")
                self._mode = 'inproc'
                self._solver = None

        # In-process fallback (still chunked, but competes for the main thread).
        self._grid = None
        self._gen = None
        self._voxgen = None
        self._voxizer = None
        if voxelize.subprocess_available():
            try:
                vx = voxelize.AsyncVoxelizer()
                vx.start(s, self._depsgraph, resolution=res)
                self._voxizer = vx
            except Exception as exc:  # noqa: BLE001
                print(f"[Blendtopo] async voxelize unavailable: {exc}")
                self._voxizer = None
        if self._voxizer is None:
            self._voxgen = voxelize.build_grid_steps(s, self._depsgraph,
                                                     resolution=res)
        self._phase = 'voxel'
        context.scene.blendtopo.status = (
            f"voxelizing L{self._level_idx + 1}/{len(self._levels)} "
            f"({res}^3)...")

    def _build_solver_job(self, context, res):
        """Assemble the (small) job for the unified solver subprocess: extract
        each object's triangles (cheap bpy work) plus SIMP params and the
        warm-start payload from the previous level."""
        s = self._settings
        grid, _reach = voxelize._grid_and_reach(s, resolution=res)
        self._last_vsize = float(grid.vsize)
        self._depsgraph.update()

        queries, descr = [], []
        v, f = voxelize._object_triangles(s.build_space, self._depsgraph)
        queries.append({'verts': v, 'faces': f, 'target': 'centers'})
        descr.append(('build', None))
        for item in s.exclude:
            if item.obj is None:
                continue
            v, f = voxelize._object_triangles(item.obj, self._depsgraph)
            queries.append({'verts': v, 'faces': f, 'target': 'centers'})
            descr.append(('exclude', None))
        for b in s.bearings:
            if b.obj is None:
                continue
            v, f = voxelize._object_triangles(b.obj, self._depsgraph)
            queries.append({'verts': v, 'faces': f, 'target': 'nodes'})
            descr.append(('bearing', (bool(b.fix_x), bool(b.fix_y),
                                      bool(b.fix_z))))
        for ld in s.loads:
            if ld.obj is None:
                continue
            v, f = voxelize._object_triangles(ld.obj, self._depsgraph)
            queries.append({'verts': v, 'faces': f, 'target': 'nodes'})
            descr.append(('load', np.asarray(ld.force, dtype=float)))

        is_final = self._level_idx == len(self._levels) - 1
        tol = 0.0 if is_final else s.convergence_tol
        return {
            'direction': voxelize.inside_worker._RAY_DIR,
            'grid': {'dims': (grid.nx, grid.ny, grid.nz),
                     'origin': np.asarray(grid.origin, dtype=float),
                     'vsize': float(grid.vsize)},
            'queries': queries, 'descr': descr,
            'simp': {'volfrac': s.volume_fraction, 'penalty': s.penalty,
                     'rmin': s.filter_radius, 'nu': s.poisson,
                     'e0': s.youngs_modulus, 'use_multigrid': s.use_multigrid,
                     'compute_mode': s.compute_mode,
                     'cpu_threads': int(s.cpu_threads),
                     'verbose': bool(s.verbose_log),
                     'max_iter': s.iters_per_level, 'tol': tol},
            'iso': float(s.iso_level), 'style': s.preview_style,
            'warm': self._prev or {},
        }

    def _setup_solve(self, context, grid):
        """Grid is ready: build the Problem and the SIMP generator for it."""
        s = self._settings
        res = self._levels[self._level_idx]
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
            use_multigrid=s.use_multigrid, compute_mode=s.compute_mode,
            cpu_threads=int(s.cpu_threads), verbose=bool(s.verbose_log),
        )

        n_iters = s.iters_per_level
        # Final pass runs all iterations (no early stop) for a fully refined
        # result; coarse passes still early-stop on Convergence for speed.
        is_final = self._level_idx == len(self._levels) - 1
        tol = 0.0 if is_final else s.convergence_tol

        # The generator yields (it, compliance, change, rho) per iteration. We
        # keep it and step it from the modal timer (main thread); no threads.
        self._prob = prob
        self._gen = prob.optimize(
            max_iter=n_iters, x_init=x_init, tol=tol, u_init=u_init)
        self._phase = 'solve'

    def modal(self, context, event):
        s = context.scene.blendtopo

        if event.type == 'ESC' and event.value == 'PRESS':
            self._interrupted = True
            s.status = "stopping (keeping latest)..."
            return {'RUNNING_MODAL'}

        if event.type == 'TIMER':
            try:
                self._step(context)
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                self.report({'ERROR'}, str(exc))
                self._all_done = True
            if self._all_done:
                return self._finish(context)

        return {'PASS_THROUGH'}

    def _step(self, context):
        """Advance the run by one timer tick.

        In 'solver' mode this is just a non-blocking poll of the subprocess plus
        (at most) building one finished preview mesh - the only main-thread work.
        In 'inproc' mode it advances the in-process generators as before.
        """
        if self._mode == 'solver':
            self._step_solver(context)
            return

        s = context.scene.blendtopo
        if self._interrupted:
            if self._prob is not None:
                self._prob.close()
                self._prob = None
            self._on_level_done(context)
            return

        if self._phase == 'voxel':
            self._step_voxelize(context)
            return

        n_levels = len(self._levels)
        try:
            it, comp, change, rho = next(self._gen)
        except StopIteration:
            # Level finished: keep its displacement field for the next level's
            # warm start, then move on (or finish).
            self._prev_u = self._prob.last_u
            self._prob.close()   # release any multi-CPU/multi-GPU pool
            self._on_level_done(context)
            return

        res = self._levels[self._level_idx]
        self._last_rho = rho
        s.current_iter = it
        s.status = (f"L{self._level_idx + 1}/{n_levels} ({res}^3)  it {it}  "
                    f"C={comp:.3e}  d={change:.3f}")
        self._emit_result(context, rho, self._level_idx, it)
        for area in context.screen.areas:
            area.tag_redraw()

    # -- solver (subprocess) mode -------------------------------------------

    def _step_solver(self, context):
        """One tick in subprocess mode: poll, and at most build one preview."""
        s = context.scene.blendtopo

        # On ESC ask the worker to stop after its current iteration; it will
        # still flush the latest step + final cache, so the result is kept.
        if self._interrupted and not self._stop_requested:
            self._solver.request_stop()
            self._stop_requested = True

        ev, payload = self._solver.poll()
        n_levels = len(self._levels)
        res = self._cur_res

        if ev == 'error':
            raise RuntimeError(payload)

        if ev == 'voxel':
            pct = 100.0 * float(payload)
            s.status = (f"voxelizing L{self._level_idx + 1}/{n_levels} "
                        f"({res}^3)  {pct:.0f}% [bg]")
        elif ev == 'step':
            it = int(payload['it'])
            s.current_iter = it
            s.status = (f"L{self._level_idx + 1}/{n_levels} ({res}^3)  it {it}  "
                        f"C={payload['compliance']:.3e}  "
                        f"d={payload['change']:.3f} [bg]")
            self._emit_step(context, payload)
        elif ev == 'done':
            self._store_cache(payload)
            self._solver.cleanup()
            self._solver = None
            self._advance_or_finish(context)

        for area in context.screen.areas:
            area.tag_redraw()

    def _emit_step(self, context, payload):
        """Turn streamed verts/faces into a Blender object (the only main-thread
        work). Mirrors _emit_result's keep-steps / discard-empty behaviour."""
        s = context.scene.blendtopo
        verts, faces = payload['verts'], payload['faces']
        it, level = int(payload['it']), self._level_idx
        prev_name = self._last_obj_name
        if s.keep_steps:
            name = f"Blendtopo_r{self._run_id}_L{level + 1}_it{it:03d}"
        else:
            name = "Blendtopo_Result"

        obj = extract.mesh_to_object(verts, faces, name, collection=self._coll)
        has_geometry = len(obj.data.polygons) > 0

        # Early iterations can mesh to nothing; don't blank the viewport.
        if not has_geometry and s.keep_steps and prev_name and prev_name != obj.name:
            me = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if me.users == 0:
                bpy.data.meshes.remove(me)
            return

        if has_geometry and s.keep_steps and prev_name and prev_name != obj.name:
            prev = bpy.data.objects.get(prev_name)
            if prev is not None:
                try:
                    prev.hide_set(True)
                except Exception:
                    prev.hide_viewport = True
        self._last_obj_name = obj.name

    def _store_cache(self, payload):
        """Stash the finished level's density/displacement as the warm start for
        the next level and as the Continue cache."""
        rho3d = payload.get('rho3d')
        u = payload.get('u')
        origin = payload.get('origin')
        vsize = payload.get('vsize')
        dims = payload.get('dims')
        node_dims = payload.get('node_dims')
        self._last_vsize = float(vsize) if vsize is not None else self._last_vsize
        if rho3d is not None:
            self._prev = {'prev_rho3d': rho3d, 'prev_origin': origin,
                          'prev_vsize': vsize, 'prev_u': u,
                          'prev_node_dims': node_dims}
            self._cache_final = {'dims': dims, 'origin': origin, 'vsize': vsize,
                                 'rho3d': rho3d, 'u': u, 'node_dims': node_dims}

    def _advance_or_finish(self, context):
        """Move to the next level, or end the run (solver mode)."""
        if self._interrupted:
            self._all_done = True
            return
        self._level_idx += 1
        if self._level_idx >= len(self._levels):
            self._all_done = True
        else:
            context.scene.blendtopo.status = "voxelizing next level..."
            self._start_level(context)

    def _step_voxelize(self, context):
        """Advance voxelization for one timer tick, then return so Blender can
        redraw. With the AsyncVoxelizer this is just a non-blocking poll (the
        actual work runs in another process); with the in-process fallback it
        drains the generator under a small time budget. Either way the UI stays
        responsive and shows a progress %."""
        s = context.scene.blendtopo

        if self._voxizer is not None:
            tag, payload = self._voxizer.poll()
            if tag == 'grid':
                self._voxizer = None
                self._setup_solve(context, payload)     # -> phase 'solve'
            elif tag == 'error':
                self._voxizer = None
                raise RuntimeError(payload)
            else:   # 'running'
                pct = 100.0 * float(payload)
                res = self._levels[self._level_idx]
                s.status = (f"voxelizing L{self._level_idx + 1}/"
                            f"{len(self._levels)} ({res}^3)  {pct:.0f}% [bg]")
            for area in context.screen.areas:
                area.tag_redraw()
            return

        budget = 0.02            # seconds of work per timer tick
        t0 = time.perf_counter()
        while True:
            try:
                tag, *rest = next(self._voxgen)
            except StopIteration:
                raise RuntimeError("voxelization produced no grid")
            if tag == 'grid':
                self._voxgen = None
                self._setup_solve(context, rest[0])     # -> phase 'solve'
                for area in context.screen.areas:
                    area.tag_redraw()
                return
            # tag == 'progress': (done_points, total_points)
            if time.perf_counter() - t0 >= budget:
                done_pts, total_pts = rest
                pct = (100.0 * done_pts / total_pts) if total_pts else 0.0
                res = self._levels[self._level_idx]
                s.status = (f"voxelizing L{self._level_idx + 1}/"
                            f"{len(self._levels)} ({res}^3)  {pct:.0f}% [main]")
                for area in context.screen.areas:
                    area.tag_redraw()
                return

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
        self._gen = None
        # Kill any still-running subprocess and clean its tmp files.
        for attr in ("_voxizer", "_solver"):
            obj_ = getattr(self, attr, None)
            if obj_ is not None:
                try:
                    obj_.cancel()
                except Exception:
                    pass
                setattr(self, attr, None)
        if getattr(self, "_timer", None) is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        s.running = False
        obj = bpy.data.objects.get(self._last_obj_name)
        if obj is not None and not self._interrupted and not self._error:
            # Guarantee a single watertight shell on the finished result.
            extract.finalize_watertight(obj)
            # Voxel size: from the live grid (in-proc) or the last level (solver).
            vsize = self._grid.vsize if self._grid is not None else self._last_vsize
            vox = (vsize * 0.75 if (s.preview_style == 'BLOCKY' and vsize)
                   else None)
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
        if not self._error and self._cache_final is not None:
            # Preferred: raw fields captured from the solver's final payload.
            _RESUME_CACHE = dict(self._cache_final)
            _RESUME_CACHE['last_obj'] = self._last_obj_name
        elif (not self._error and self._prev_grid is not None
              and self._prev_rho3d is not None):
            # In-process fallback path: derive the same raw fields from the grid.
            g = self._prev_grid
            _RESUME_CACHE = {
                'dims': (g.nx, g.ny, g.nz), 'origin': np.asarray(g.origin),
                'vsize': float(g.vsize), 'rho3d': self._prev_rho3d,
                'u': self._prev_u, 'node_dims': (g.nx + 1, g.ny + 1, g.nz + 1),
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
    # Guard against a class never having been registered (e.g. register()
    # aborted partway through -- Blender then still calls unregister() on
    # every module as cleanup, which would otherwise raise "missing bl_rna
    # attribute ... may not be registered").
    for cls in reversed(_CLASSES):
        if hasattr(cls, "bl_rna"):
            bpy.utils.unregister_class(cls)
