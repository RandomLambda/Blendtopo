# Blendtopo — Topology Optimization for Blender

Generate strong, lightweight, organic structures inside any mesh — the kind of
load-bearing lattices you see in 3D-printed brackets and shelves — right in
Blender. You give it a build space, where it's held (bearings) and where the
forces act (loads); it figures out where material needs to be.

Free, open source (GPL-3.0), and zero external dependencies: it runs on
Blender's bundled numpy and built-in modifiers. Nothing to pip-install.

![The Blendtopo panel and a result forming inside the build space](docs/panel.png)

## Gallery

| | |
|---|---|
| ![Setting up the build space, load and bearing](videos/output_gifs/Video_2026-06-21_162332.gif) | ![Live optimization](videos/output_gifs/Video_2026-06-21_190131.gif) |
| *Set up: build space, load and bearing* | *Live coarse-to-fine optimization* |
| ![Refining to the final result](videos/output_gifs/Video_2026-06-21_195839.gif) | ![Optimized organic result](videos/output_gifs/Video_2026-06-21_200149.gif) |
| *Refining the structure* | *Optimized organic result* |

## Install (Blender 4.2+)

Drag the `blendtopo-*.zip` into the Blender window and confirm — it installs as
an extension. (Or Edit ▸ Preferences ▸ Add-ons ▸ Install from Disk.) Then open
the 3D-viewport sidebar with **N** and pick the **Blendtopo** tab. To update,
disable the old version first, then drop in the new zip.

## Quick start

1. Add a mesh for the build space → **Set Build Space from Active**.
2. **Add Bearing(s)**: small helper meshes marking where the part is anchored;
   choose which axes are fixed.
3. **Add Load(s)**: helper meshes where forces act; set each force vector.
4. (Optional) **Add Exclusion(s)** for keep-out volumes (holes, clearances).
5. **Run Optimization**. The shape forms and refines live; **ESC** stops and
   keeps the latest. **Continue** runs more iterations from where you stopped.
6. **Optimize Mesh** turns the result into a clean, low-poly, watertight body
   ready to export or 3D-print.

Helper meshes (bearings/loads/exclusions) just need to overlap the build space;
any grid node inside them is fixed / loaded / removed. Inputs are shown as
wireframe so you can watch the result form inside.

## Settings (hover any field for a full tooltip)

- **Final Resolution / Refine Levels** — detail and the coarse-to-fine ladder.
- **Iterations / Level**, **Convergence** — work per stage; coarse stages stop
  early on Convergence, the final stage runs in full.
- **Volume Fraction** — how much material to keep (weight vs. stiffness).
- **Filter Radius** — minimum feature size (decrease for finer struts).
- **Iso Level** — surface threshold; **Preview** — smooth or blocky.
- **Multigrid solver / Use GPU** — speed (see below).

## Speed

The solve is matrix-free (no global matrix assembled) and restricted to the
build-space region. It warm-starts each step and each refinement level from the
previous one, uses an adaptive solve tolerance, and is preconditioned by a
**geometric-multigrid V-cycle** so iteration counts stay low as the grid grows.
If **CuPy** + a CUDA GPU are present it runs on the GPU automatically (validated
by a self-test at startup; it silently falls back to CPU otherwise).


## Changelog
<img width="970" height="691" alt="panel" src="https://github.com/user-attachments/assets/c08d2d76-73c3-471f-8200-bd21a51695a3" />

<img width="800" height="426" alt="Video_2026-06-21_190131" src="https://github.com/user-attachments/assets/395e7637-6855-4fa3-bc81-935ce89ae645" />
<img width="800" height="499" alt="Video_2026-06-21_162332" src="https://github.com/user-attachments/assets/1c1ea9d6-898d-40df-8246-5289d54cc054" />
<img width="800" height="426" alt="Video_2026-06-21_200149" src="https://github.com/user-attachments/assets/3be827b9-9b71-4744-a7f0-f8c9aa2bbeda" />

- **0.8.0** — Geometric-multigrid preconditioner (grid-independent solve speed);
  "Continue" to resume more iterations; final pass runs full (no early stop);
  Optimize Mesh now reduces vertex count with adjustable smooth/decimate;
  clearer error status.
- **0.7.x** — Robust point-in-mesh voxelisation (fixed scattered holes);
  GPU self-test with auto CPU fallback; faster matvec (matmul + bincount);
  cross-level displacement warm-start; adaptive CG tolerance; tooltips.
- **0.6.0** — Build-space-restricted solver; centred grid; GPU status; rich
  tooltips; GPL license + publishing guide.
- **0.5.0** — CG warm-start; convergence early-stop; optional CuPy GPU path;
  Optimize Mesh button.
- **0.4.x** — Smooth surface-nets preview; watertight output; results
  collection; background-thread solve; wireframe inputs; morph-style preview.
- **0.3.x** — Coarse-to-fine multi-resolution; packaged as an extension.
- **0.2.0 / 0.1.0** — Voxeliser, matrix-free FEA + SIMP core, sidebar UI.

## License

GPL-3.0-or-later. © 2026 Philip Gruenenfelder. See `LICENSE`.
