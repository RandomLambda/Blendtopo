# SPDX-License-Identifier: GPL-3.0-or-later
"""
Turn Blender objects into a voxel grid for the optimizer.

This module *does* touch bpy/mathutils (it reads scene geometry), but it
returns plain numpy arrays so the optimizer core stays Blender-agnostic.

Node/element ordering matches core.fea (element ex + nx*(ey + ny*ez);
node ix + (nx+1)*(iy + (ny+1)*iz); DOFs 3n, 3n+1, 3n+2).
"""

import numpy as np

try:
    import bpy  # noqa: F401
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    _HAS_BPY = True
except Exception:  # allow import in headless tests
    _HAS_BPY = False


class Grid:
    def __init__(self, dims, origin, vsize):
        self.nx, self.ny, self.nz = dims
        self.origin = np.asarray(origin, dtype=float)   # world min corner
        self.vsize = float(vsize)
        self.active = None
        self.fixed_dofs = None
        self.force = None

    @property
    def ndof(self):
        return 3 * (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    def voxel_centers(self):
        """(nelem, 3) world-space centers, element-index order."""
        xs = (np.arange(self.nx) + 0.5) * self.vsize + self.origin[0]
        ys = (np.arange(self.ny) + 0.5) * self.vsize + self.origin[1]
        zs = (np.arange(self.nz) + 0.5) * self.vsize + self.origin[2]
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    def node_coords(self):
        """((nx+1)*(ny+1)*(nz+1), 3) world node positions, ordered so the array
        index EQUALS the FEA global node id used in core.fea
        (id = ix + (nx+1)*(iy + (ny+1)*iz), i.e. x fastest). This alignment is
        what makes bearings/loads land on the correct nodes."""
        nxp, nyp, nzp = self.nx + 1, self.ny + 1, self.nz + 1
        n = np.arange(nxp * nyp * nzp)
        ix = n % nxp
        iy = (n // nxp) % nyp
        iz = n // (nxp * nyp)
        coords = np.stack([ix, iy, iz], axis=1).astype(float) * self.vsize
        return coords + self.origin

    def node_id(self, ix, iy, iz):
        return ix + (self.nx + 1) * (iy + (self.ny + 1) * iz)


def _bvh_from_object(obj, depsgraph):
    """World-space BVHTree of an evaluated mesh object."""
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    mw = obj.matrix_world
    verts = [mw @ v.co for v in mesh.vertices]
    polys = [tuple(p.vertices) for p in mesh.polygons]
    tree = BVHTree.FromPolygons(verts, polys, all_triangles=False, epsilon=0.0)
    eval_obj.to_mesh_clear()
    return tree


# Slightly tilted, non-axis-aligned ray. With an axis-aligned build mesh and an
# axis-aligned voxel grid, a pure +X ray grazes faces/edges exactly edge-on and
# the crossing-parity test misfires, punching scattered holes in the mask. A
# tilted direction can never lie in a face plane, so parity is robust.
_rd = np.array([1.0, 0.0073301, 0.0031337])
_RAY_DIR_T = tuple(_rd / np.linalg.norm(_rd))   # normalized, no mathutils needed


def _inside_mask(tree, points, span0):
    """Robust parity ray-cast: True if a point is inside the closed mesh."""
    inside = np.zeros(len(points), dtype=bool)
    d = Vector(_RAY_DIR_T)
    eps = 1e-5
    span0 = float(span0)
    for i, p in enumerate(points):
        origin = Vector((float(p[0]), float(p[1]), float(p[2])))
        cur = origin.copy()
        count = 0
        remaining = span0
        while remaining > 0.0:
            loc, nrm, idx, dist = tree.ray_cast(cur, d, remaining)
            if loc is None:
                break
            count += 1
            step = (loc - cur).length + eps
            cur = loc + d * eps
            remaining -= step
            if count > 512:
                break
        inside[i] = (count % 2) == 1
    return inside


def build_grid(settings, depsgraph, resolution=None):
    """Construct a Grid from the scene's Blendtopo settings.

    resolution overrides settings.resolution (used by the coarse-to-fine
    driver to build successively finer grids over the same build space).
    """
    bs = settings.build_space
    if bs is None:
        raise ValueError("No build space set")
    if resolution is None:
        resolution = settings.resolution

    # Bounding box of build space in world space.
    corners = [bs.matrix_world @ Vector(c) for c in bs.bound_box]
    cmin = np.min([[c.x, c.y, c.z] for c in corners], axis=0)
    cmax = np.max([[c.x, c.y, c.z] for c in corners], axis=0)
    ext = cmax - cmin
    longest = float(np.max(ext))
    vsize = longest / resolution
    dims = tuple(max(1, int(np.ceil(e / vsize))) for e in ext)

    # Centre the grid on the AABB so the voxel lattice is symmetric about the
    # build space (previously anchored at the min corner -> visible drift).
    total = np.array(dims, dtype=float) * vsize
    origin = cmin - 0.5 * (total - ext)
    grid = Grid(dims, origin, vsize)

    depsgraph.update()
    reach = float(np.sqrt((total ** 2).sum())) + 2.0 * vsize

    # Build-space inside test on voxel centers.
    bs_tree = _bvh_from_object(bs, depsgraph)
    centers = grid.voxel_centers()
    inside = _inside_mask(bs_tree, centers, reach)

    # Exclusions force voxels empty.
    excl = np.zeros(len(centers), dtype=bool)
    for item in settings.exclude:
        if item.obj is None:
            continue
        t = _bvh_from_object(item.obj, depsgraph)
        excl |= _inside_mask(t, centers, reach)

    grid.active = (inside & ~excl)

    # Bearings -> fixed node DOFs.
    nodes = grid.node_coords()
    fixed = []
    for b in settings.bearings:
        if b.obj is None:
            continue
        t = _bvh_from_object(b.obj, depsgraph)
        nin = np.where(_inside_mask(t, nodes, reach))[0]
        for n in nin:
            if b.fix_x:
                fixed.append(3 * n)
            if b.fix_y:
                fixed.append(3 * n + 1)
            if b.fix_z:
                fixed.append(3 * n + 2)
    grid.fixed_dofs = (np.unique(np.asarray(fixed, dtype=np.int64))
                       if fixed else np.zeros(0, dtype=np.int64))

    # Loads -> distributed nodal force vector.
    force = np.zeros(grid.ndof)
    for ld in settings.loads:
        if ld.obj is None:
            continue
        t = _bvh_from_object(ld.obj, depsgraph)
        nin = np.where(_inside_mask(t, nodes, reach))[0]
        if len(nin) == 0:
            continue
        fv = np.asarray(ld.force, dtype=float) / len(nin)
        for n in nin:
            force[3 * n:3 * n + 3] += fv
    grid.force = force

    return grid
