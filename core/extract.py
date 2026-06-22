# SPDX-License-Identifier: GPL-3.0-or-later
"""
Density field -> mesh.

Two extraction styles:
  * 'BLOCKY'  : boundary faces of solid voxels (fast, faceted) - good for a
                rough first glance.
  * 'SMOOTH'  : Naive Surface Nets - a dual-contouring method that places one
                vertex per straddling cell at the averaged iso-crossing and
                stitches them into quads. Gives a smooth surface straight from
                the density field, no marching-cubes tables, and it follows the
                continuous density so the live preview looks organic.

The pure-numpy functions (cubes_from_density, surface_nets) are headless-
testable; density_to_object / apply_remesh_smooth build the Blender mesh.
"""

import numpy as np

try:
    import bpy
    _HAS_BPY = True
except Exception:
    _HAS_BPY = False


# ---------------------------------------------------------------------------
# BLOCKY: boundary faces of solid voxels
# ---------------------------------------------------------------------------

_FACES = {
    (-1, 0, 0): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    (1, 0, 0):  [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    (0, 1, 0):  [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
    (0, 0, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    (0, 0, 1):  [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
}


def cubes_from_density(density3d, iso, origin, vsize):
    """Boundary surface of the solid region as axis-aligned quads."""
    solid = density3d >= iso
    nx, ny, nz = solid.shape
    origin = np.asarray(origin, dtype=float)

    verts, faces, vindex = [], [], {}

    def vert(ix, iy, iz):
        key = (ix, iy, iz)
        idx = vindex.get(key)
        if idx is None:
            idx = len(verts)
            vindex[key] = idx
            verts.append(origin + np.array([ix, iy, iz]) * vsize)
        return idx

    for ix, iy, iz in np.argwhere(solid):
        for (dx, dy, dz), corners in _FACES.items():
            jx, jy, jz = ix + dx, iy + dy, iz + dz
            inside = (0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz
                      and solid[jx, jy, jz])
            if inside:
                continue
            faces.append([vert(ix + cx, iy + cy, iz + cz)
                          for cx, cy, cz in corners])

    if not verts:
        return np.zeros((0, 3)), []
    return np.asarray(verts, dtype=float), faces


# ---------------------------------------------------------------------------
# SMOOTH: Naive Surface Nets (dual contouring)
# ---------------------------------------------------------------------------

def _edge_cross(a, b, iso, axis, base_xyz, vsize):
    """Crossing positions + validity for one family of grid edges.

    a, b   : scalar values at the two endpoints (arrays, same shape)
    axis   : 0/1/2 - the axis the edge runs along
    base_xyz : tuple of (X, Y, Z) world coords of endpoint 'a' (arrays)
    Returns (pos[..., 3], valid_mask).
    """
    valid = ((a < iso) & (b >= iso)) | ((a >= iso) & (b < iso))
    denom = b - a
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(denom != 0, (iso - a) / denom, 0.5)
    t = np.clip(t, 0.0, 1.0)
    X, Y, Z = base_xyz
    pos = np.stack([X, Y, Z], axis=-1).astype(float)
    pos[..., axis] += t * vsize
    return pos, valid


def surface_nets(density3d, iso, origin, vsize):
    """Smooth iso-surface as quads via Naive Surface Nets.

    Scalar samples live at voxel centers. A 'cell' spans 8 neighbouring centers;
    cells crossed by the iso get one vertex (mean of their edge crossings).
    Each interior grid edge that crosses the iso links its 4 surrounding cell
    vertices into a quad.
    """
    S = np.asarray(density3d, dtype=float)
    if min(S.shape) < 1:
        return np.zeros((0, 3)), []
    origin = np.asarray(origin, dtype=float)

    # Pad one empty layer on every side so material touching the domain
    # boundary gets capped -> watertight surface. The ghost layer sits half a
    # voxel outside the original sample grid, so origin shifts by -vsize.
    S = np.pad(S, 1, mode='constant', constant_values=0.0)
    origin = origin - vsize
    nx, ny, nz = S.shape
    if nx < 2 or ny < 2 or nz < 2:
        return np.zeros((0, 3)), []

    # World coords of every sample point (voxel center).
    ax = origin[0] + (np.arange(nx) + 0.5) * vsize
    ay = origin[1] + (np.arange(ny) + 0.5) * vsize
    az = origin[2] + (np.arange(nz) + 0.5) * vsize
    X, Y, Z = np.meshgrid(ax, ay, az, indexing='ij')

    # Edge crossings for the three families.
    Ex, Mx = _edge_cross(S[:-1, :, :], S[1:, :, :], iso, 0,
                         (X[:-1, :, :], Y[:-1, :, :], Z[:-1, :, :]), vsize)
    Ey, My = _edge_cross(S[:, :-1, :], S[:, 1:, :], iso, 1,
                         (X[:, :-1, :], Y[:, :-1, :], Z[:, :-1, :]), vsize)
    Ez, Mz = _edge_cross(S[:, :, :-1], S[:, :, 1:], iso, 2,
                         (X[:, :, :-1], Y[:, :, :-1], Z[:, :, :-1]), vsize)

    cx, cy, cz = nx - 1, ny - 1, nz - 1
    vsum = np.zeros((cx, cy, cz, 3))
    vcnt = np.zeros((cx, cy, cz))

    def add(E, M):
        m = M[..., None]
        vsum[:] += np.where(m, E, 0.0)
        vcnt[:] += M

    # x-edges touch cells offset in (y,z); slice Ex/Mx to cell shape.
    add(Ex[:, :cy, :cz], Mx[:, :cy, :cz])
    add(Ex[:, 1:, :cz], Mx[:, 1:, :cz])
    add(Ex[:, :cy, 1:], Mx[:, :cy, 1:])
    add(Ex[:, 1:, 1:], Mx[:, 1:, 1:])
    # y-edges: offset in (x,z)
    add(Ey[:cx, :, :cz], My[:cx, :, :cz])
    add(Ey[1:, :, :cz], My[1:, :, :cz])
    add(Ey[:cx, :, 1:], My[:cx, :, 1:])
    add(Ey[1:, :, 1:], My[1:, :, 1:])
    # z-edges: offset in (x,y)
    add(Ez[:cx, :cy, :], Mz[:cx, :cy, :])
    add(Ez[1:, :cy, :], Mz[1:, :cy, :])
    add(Ez[:cx, 1:, :], Mz[:cx, 1:, :])
    add(Ez[1:, 1:, :], Mz[1:, 1:, :])

    has_v = vcnt > 0
    vidx = -np.ones((cx, cy, cz), dtype=np.int64)
    order = np.argwhere(has_v)
    verts = np.zeros((len(order), 3))
    for n, (i, j, k) in enumerate(order):
        vidx[i, j, k] = n
        verts[n] = vsum[i, j, k] / vcnt[i, j, k]

    faces = []

    def quad(a, b, c, d, flip):
        if a < 0 or b < 0 or c < 0 or d < 0:
            return
        faces.append([a, b, c, d] if flip else [d, c, b, a])

    # x-edges interior in (y,z): link cells (i, j-1..j, k-1..k)
    solid = S >= iso
    xs = np.argwhere(Mx[:, 1:cy, 1:cz])  # j,k shifted by +1
    for i, jj, kk in xs:
        j, k = jj + 1, kk + 1
        flip = S[i + 1, j, k] < S[i, j, k]
        quad(vidx[i, j - 1, k - 1], vidx[i, j, k - 1],
             vidx[i, j, k], vidx[i, j - 1, k], flip)

    ys = np.argwhere(My[1:cx, :, 1:cz])
    for ii, j, kk in ys:
        i, k = ii + 1, kk + 1
        flip = S[i, j + 1, k] < S[i, j, k]
        quad(vidx[i - 1, j, k - 1], vidx[i, j, k - 1],
             vidx[i, j, k], vidx[i - 1, j, k], not flip)

    zs = np.argwhere(Mz[1:cx, 1:cy, :])
    for ii, jj, k in zs:
        i, j = ii + 1, jj + 1
        flip = S[i, j, k + 1] < S[i, j, k]
        quad(vidx[i - 1, j - 1, k], vidx[i, j - 1, k],
             vidx[i, j, k], vidx[i - 1, j, k], flip)

    return verts, faces


# ---------------------------------------------------------------------------
# Blender object building
# ---------------------------------------------------------------------------

def density_to_object(density3d, iso, origin, vsize, name,
                      style='SMOOTH', collection=None, world_matrix=None):
    """Create (or reuse) a Blender mesh object from a density field."""
    if not _HAS_BPY:
        raise RuntimeError("density_to_object requires Blender")

    if style == 'BLOCKY':
        verts, faces = cubes_from_density(density3d, iso, origin, vsize)
    else:
        verts, faces = surface_nets(density3d, iso, origin, vsize)

    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    mesh.update()
    # Smooth shading + consistent normals for a clean preview.
    try:
        mesh.validate(clean_customdata=False)
        for poly in mesh.polygons:
            poly.use_smooth = True
    except Exception:
        pass

    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, mesh)
        target = collection if collection is not None else bpy.context.scene.collection
        target.objects.link(obj)
    else:
        old = obj.data
        obj.data = mesh
        if old.users == 0:
            bpy.data.meshes.remove(old)

    if world_matrix is not None:
        obj.matrix_world = world_matrix
    return obj


def finalize_watertight(obj):
    """Weld coincident verts and recalc outward normals on the final mesh.

    The surface-nets output is already closed (boundary padding caps anything
    touching the domain edge), so this is belt-and-suspenders: it guarantees a
    single welded shell with consistent normals before export/3D-print.
    """
    if not _HAS_BPY:
        return
    import bmesh
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.update()


def apply_remesh_smooth(obj, voxel_size=None, smooth_iters=5):
    """Optional final pass: voxel remesh + Laplacian smooth."""
    if not _HAS_BPY:
        return
    if voxel_size:
        rem = obj.modifiers.new("TO_Remesh", 'REMESH')
        rem.mode = 'VOXEL'
        rem.voxel_size = voxel_size
        rem.use_smooth_shade = True
    if smooth_iters > 0:
        sm = obj.modifiers.new("TO_Smooth", 'SMOOTH')
        sm.iterations = smooth_iters
        sm.factor = 0.5
