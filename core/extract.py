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

import itertools

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

def mesh_to_object(verts, faces, name, collection=None, world_matrix=None):
    """Create (or reuse) a Blender mesh object from ready-made verts/faces.

    This is the *only* step that must run on Blender's main thread; the meshing
    itself (surface_nets / cubes_from_density) is pure numpy and can be done in
    a worker process, which then streams the verts/faces here. ``verts`` may be
    an (N,3) array; ``faces`` an (M,k) array or a list of index sequences.
    """
    if not _HAS_BPY:
        raise RuntimeError("mesh_to_object requires Blender")

    import numpy as _np
    verts = _np.asarray(verts, dtype=float)
    vlist = verts.tolist() if len(verts) else []
    if isinstance(faces, _np.ndarray):
        flist = faces.tolist()
    else:
        flist = [list(f) for f in faces]

    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(vlist, [], flist)
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


def density_to_object(density3d, iso, origin, vsize, name,
                      style='SMOOTH', collection=None, world_matrix=None):
    """Create (or reuse) a Blender mesh object from a density field."""
    if not _HAS_BPY:
        raise RuntimeError("density_to_object requires Blender")

    if style == 'BLOCKY':
        verts, faces = cubes_from_density(density3d, iso, origin, vsize)
    else:
        verts, faces = surface_nets(density3d, iso, origin, vsize)

    return mesh_to_object(verts, faces, name, collection=collection,
                          world_matrix=world_matrix)


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


# ---------------------------------------------------------------------------
# Von Mises stress heatmap: per-voxel scalar field -> vertex colors
# ---------------------------------------------------------------------------

_STRESS_ATTR = "StressVM"
_STRESS_MAT = "Blendtopo_StressPreview"


def _jet_rgb(t):
    """Classic FEA blue -> cyan -> green -> yellow -> red colormap.

    t : array-like in [0, 1] (values outside are clamped). Returns (N, 3)
    RGB floats in [0, 1]. Pure numpy, headless-testable.
    """
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    r = np.clip(np.minimum(4 * t - 1.5, -4 * t + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(4 * t - 0.5, -4 * t + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(4 * t + 0.5, -4 * t + 2.5), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _sample_field_trilinear(field3d, origin, vsize, points):
    """Trilinear sample of a voxel-centered scalar field at world points.

    Same interpolation scheme as simp.resample_density (kept independent
    here so extract.py stays a pure numpy + bpy module with no solver
    dependency). Points outside the grid are clamped to the boundary cell.
    """
    nx, ny, nz = field3d.shape
    origin = np.asarray(origin, dtype=float)
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0)
    fi = (points - origin) / vsize - 0.5
    f0 = np.floor(fi).astype(int)
    fr = fi - f0
    out = np.zeros(len(points))
    for corner in itertools.product((0, 1), repeat=3):
        w = np.ones(len(points))
        for d in range(3):
            w *= fr[:, d] if corner[d] else (1.0 - fr[:, d])
        ix = np.clip(f0[:, 0] + corner[0], 0, nx - 1)
        iy = np.clip(f0[:, 1] + corner[1], 0, ny - 1)
        iz = np.clip(f0[:, 2] + corner[2], 0, nz - 1)
        out += w * field3d[ix, iy, iz]
    return out


def _sample_field_trilinear_weighted(field3d, weight3d, origin, vsize, points):
    """Trilinear sample like ``_sample_field_trilinear``, but each of the 8
    corner contributions is additionally weighted by ``weight3d`` (SIMP
    density * active-mask, i.e. how much real, reliable material sits in
    that element) and the result is renormalized.

    Plain trilinear sampling treats a nearly-void border element (one that
    an oblique iso-surface cuts through, leaving little real material and an
    unreliable, near-zero stress estimate) exactly the same as a fully solid
    neighbor. Every vertex near a slanted face then blends in that
    unreliable low value at full geometric weight, which is what produced
    the false "blue ring" banding around such faces even though nothing
    about the actual stress state changed there. Weighting each corner by
    its own fill fraction lets a mostly-empty element barely influence the
    average while a full one still dominates - the same idea as
    Zienkiewicz-Zhu superconvergent patch recovery, just using SIMP density
    as the reliability weight instead of element quality. Peak stress in
    solid interior elements is untouched (their weight is ~1 either way);
    only the unreliable border blending is suppressed.
    """
    nx, ny, nz = field3d.shape
    origin = np.asarray(origin, dtype=float)
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0)
    fi = (points - origin) / vsize - 0.5
    f0 = np.floor(fi).astype(int)
    fr = fi - f0
    num = np.zeros(len(points))
    den = np.zeros(len(points))
    for corner in itertools.product((0, 1), repeat=3):
        w = np.ones(len(points))
        for d in range(3):
            w *= fr[:, d] if corner[d] else (1.0 - fr[:, d])
        ix = np.clip(f0[:, 0] + corner[0], 0, nx - 1)
        iy = np.clip(f0[:, 1] + corner[1], 0, ny - 1)
        iz = np.clip(f0[:, 2] + corner[2], 0, nz - 1)
        cw = w * weight3d[ix, iy, iz]
        num += cw * field3d[ix, iy, iz]
        den += cw
    # A vertex whose whole 8-corner neighborhood is unreliable (den ~ 0,
    # e.g. right at the outer edge of the design domain) has no trustworthy
    # weighted estimate at all - fall back to the plain distance-weighted
    # value there rather than returning a meaningless 0.
    safe = den > 1e-6
    plain = _sample_field_trilinear(field3d, origin, vsize, points)
    return np.where(safe, num / np.where(safe, den, 1.0), plain)


def _weighted_smooth3d(field3d, weight3d):
    """One light pass of a reliability-weighted 3D blur (center + 6 face
    neighbors, center counted twice so the field isn't washed out).

    The corner-reliability weighting above removes the systematic "blue
    ring" bias, but a per-element (voxel-resolution) stress field still has
    a visible per-voxel-layer staircase/aliasing pattern once mapped onto a
    much denser iso-surface mesh - thin stripes that track the voxel grid
    rather than anything physical. This knocks that residual quantization
    down without touching the border-reliability fix: neighbors are still
    weighted by their own fill fraction, so an unreliable near-void voxel
    still can't drag a solid one down, same as in the trilinear sampler.
    """
    f = np.asarray(field3d, dtype=np.float64)
    w = np.asarray(weight3d, dtype=np.float64)
    fw = np.pad(f * w, 1, mode='constant')
    wp = np.pad(w, 1, mode='constant')
    c = (slice(1, -1),) * 3
    num = 2.0 * fw[c]
    den = 2.0 * wp[c]
    for axis in range(3):
        for shift in (-1, 1):
            sl = list(c)
            sl[axis] = slice(1 + shift, (-1 + shift) or None)
            sl = tuple(sl)
            num = num + fw[sl]
            den = den + wp[sl]
    safe = den > 1e-9
    return np.where(safe, num / np.where(safe, den, 1.0), f)


def stress_vertex_colors(stress3d, origin, vsize, verts, vmax, vmin=0.0,
                          weight3d=None, smooth=True):
    """Map a per-voxel von Mises stress field onto per-vertex RGBA colors.

    vmin/vmax define the color scale - the caller is expected to have
    already percentile-clipped and EMA-smoothed vmax across iterations
    (see operators._stress_scale) so a handful of singular elements at
    point loads/supports don't blow out the whole heatmap, and the scale
    doesn't flicker step to step. ``weight3d`` (optional), if given, is used
    for a reliability-weighted sample instead of plain trilinear - see
    ``_sample_field_trilinear_weighted``. ``smooth`` (only applies when
    ``weight3d`` is given) runs one light reliability-weighted blur pass
    first (see ``_weighted_smooth3d``) to knock down residual per-voxel-
    layer aliasing. Returns (N, 4) float RGBA in [0, 1].
    """
    verts = np.asarray(verts, dtype=float)
    if len(verts) == 0:
        return np.zeros((0, 4))
    if weight3d is not None:
        field = _weighted_smooth3d(stress3d, weight3d) if smooth else stress3d
        vals = _sample_field_trilinear_weighted(
            field, weight3d, origin, vsize, verts)
    else:
        vals = _sample_field_trilinear(stress3d, origin, vsize, verts)
    span = max(float(vmax) - float(vmin), 1e-12)
    t = (vals - float(vmin)) / span
    rgb = _jet_rgb(t)
    return np.concatenate([rgb, np.ones((len(rgb), 1))], axis=-1)


def _ensure_stress_material():
    """Attribute -> Emission material: reads the color attribute straight
    through, so the heatmap looks the same regardless of scene lighting or
    viewport shading mode (it's data, not a lit surface)."""
    mat = bpy.data.materials.get(_STRESS_MAT)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(_STRESS_MAT)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    emit = nt.nodes.new('ShaderNodeEmission')
    attr = nt.nodes.new('ShaderNodeAttribute')
    attr.attribute_type = 'GEOMETRY'
    attr.attribute_name = _STRESS_ATTR
    out.location = (300, 0)
    emit.location = (100, 0)
    attr.location = (-150, 0)
    nt.links.new(attr.outputs['Color'], emit.inputs['Color'])
    nt.links.new(emit.outputs['Emission'], out.inputs['Surface'])
    return mat


def _bake_vertex_colors(obj, rgba):
    """Write a (N,4) RGBA array (N = vertex count) onto obj as a POINT-domain
    FLOAT_COLOR attribute, plus the flat Attribute->Emission preview material
    in slot 0. Shared by apply_stress_colors (surface/cloud with a value per
    real vertex) - the actual sampling/lookup happens in the caller.
    """
    me = obj.data
    if len(me.vertices) == 0:
        return
    attr = me.color_attributes.get(_STRESS_ATTR)
    if attr is None:
        attr = me.color_attributes.new(
            name=_STRESS_ATTR, type='FLOAT_COLOR', domain='POINT')
    attr.data.foreach_set('color', np.asarray(rgba, dtype=float).ravel())
    try:
        me.color_attributes.active_color_name = _STRESS_ATTR
    except Exception:
        pass
    me.update()

    mat = _ensure_stress_material()
    if len(me.materials) == 0:
        me.materials.append(mat)
    else:
        me.materials[0] = mat


def apply_stress_colors(obj, stress3d, origin, vsize, vmax, vmin=0.0,
                         weight3d=None):
    """Bake a von Mises stress heatmap onto obj by trilinearly sampling
    stress3d at each of the object's actual vertex positions. Since every
    preview mesh is a fresh datablock (see mesh_to_object), there is nothing
    to clean up when the heatmap is toggled off - a mesh simply won't have
    this attribute/material unless this is called for it. ``weight3d``, if
    given, switches sampling to the reliability-weighted variant (see
    ``stress_vertex_colors`` / ``_sample_field_trilinear_weighted``) so
    partially-empty border elements don't wash out the surface heatmap.
    """
    if not _HAS_BPY:
        return
    me = obj.data
    if len(me.vertices) == 0:
        return
    verts = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', verts)
    rgba = stress_vertex_colors(stress3d, origin, vsize,
                                verts.reshape(-1, 3), vmax, vmin,
                                weight3d=weight3d)
    _bake_vertex_colors(obj, rgba)


# ---------------------------------------------------------------------------
# Voxel-cloud bake: every active element as its own real, independently-
# colored cube, for a cross-section view. Deliberately real Python-built cube
# geometry (verts + faces, colored exactly like the surface mesh) rather than
# Geometry-Nodes instancing: instancing needs the color attribute to survive
# an Instance-on-Points -> Realize-Instances round trip, which is a known
# fragile spot (and a plain per-instance size input silently not landing
# would explain a solid white block the size of the whole build space - the
# failure mode this replaced). Baking real geometry in Python sidesteps that
# entirely: it is exactly the mechanism already proven on the surface mesh.
# Geometry Nodes' only remaining job is the clip itself, driven by a movable
# Empty (see ensure_stress_cloud_geonodes). This is deliberately every active
# element, not a decimated sample - a partial cloud would hide material from
# the cross-section that is actually there.
# ---------------------------------------------------------------------------

_CLIP_GROUP = "Blendtopo_VoxelClip_v2"   # _v2: force a rebuild past any stale
                                         # cached group from the old instanced
                                         # (Mesh To Points -> Instance On
                                         # Points -> Realize Instances) design
_CLIP_EMPTY = "Blendtopo_ClipPlane"
_CLIP_MODIFIER = "Blendtopo_VoxelClip"

# Local-space corners/faces of a unit cube centered on the origin, used to
# stamp out one independent (unwelded) cube per active voxel.
_CUBE_LOCAL_VERTS = np.array([
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
], dtype=float)
_CUBE_LOCAL_FACES = np.array([
    [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
    [3, 2, 6, 7], [0, 3, 7, 4], [1, 5, 6, 2],
], dtype=np.int64)


def _active_voxel_centers(active3d, origin, vsize):
    """World-space centers of every True cell in active3d, in the same flat
    (C/'ij') order as active3d.ravel() - so a field3d[active3d] slice lines
    up 1:1 with these points for coloring."""
    nx, ny, nz = active3d.shape
    origin = np.asarray(origin, dtype=float)
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing='ij')
    centers = np.stack([ix, iy, iz], axis=-1).astype(float)
    centers = origin + (centers + 0.5) * vsize
    return centers[np.asarray(active3d, dtype=bool)]


def _voxel_cubes(active3d, origin, vsize):
    """One independent cube per active element: (N*8, 3) verts, (N*6, 4)
    faces. Vertices for cube i occupy [8*i : 8*i+8], in the same order as
    _active_voxel_centers - so per-voxel data (e.g. stress) can be repeated
    8x to color every cube's own vertices without any lookup."""
    centers = _active_voxel_centers(active3d, origin, vsize)
    n = len(centers)
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64)
    verts = (centers[:, None, :]
            + _CUBE_LOCAL_VERTS[None, :, :] * vsize).reshape(-1, 3)
    faces = (_CUBE_LOCAL_FACES[None, :, :]
            + (np.arange(n, dtype=np.int64) * 8)[:, None, None]).reshape(-1, 4)
    return verts, faces


def _cubes_mesh_to_object(verts, faces, name, collection=None):
    """Lean mesh builder for the voxel-cube cloud: skips mesh_to_object's
    per-polygon smooth-shading loop and validate() pass (not wanted here -
    each voxel should read as a flat-shaded cube, and a Python per-polygon
    loop doesn't scale to the millions of faces a fine, dense grid produces).
    """
    if not _HAS_BPY:
        raise RuntimeError("_cubes_mesh_to_object requires Blender")
    verts = np.asarray(verts, dtype=float)
    vlist = verts.tolist() if len(verts) else []
    flist = faces.tolist() if isinstance(faces, np.ndarray) else [list(f) for f in faces]

    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(vlist, [], flist)
    mesh.update()

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
    return obj


# Cull elements below this fraction of the vmax..vmin color range - i.e.
# "practically zero" stress - from the cloud entirely, not just color them
# dark. Otherwise the exterior is a solid low-stress shell you'd have to clip
# through before seeing anything interesting.
_CLOUD_MIN_FRAC = 0.05


def voxel_cloud_to_object(stress3d, active3d, origin, vsize, vmax, vmin,
                          name, collection=None, min_frac=_CLOUD_MIN_FRAC):
    """Build (or reuse) the stress cross-section object: one real, fully-
    colored cube for every active element that clears min_frac of the color
    range - every one that clears it, not a decimated sample; elements below
    it are dropped from the geometry, not just colored dark. Colors are exact
    (no interpolation) - each cube's 8 vertices just take that element's own
    (already smoothed) stress value.
    """
    active_mask = np.asarray(active3d, dtype=bool)
    stress3d = np.asarray(stress3d)
    threshold = float(vmin) + min_frac * max(float(vmax) - float(vmin), 1e-12)
    keep = active_mask & (stress3d >= threshold)

    verts, faces = _voxel_cubes(keep, origin, vsize)
    obj = _cubes_mesh_to_object(verts, faces, name, collection=collection)
    if len(verts):
        vals = stress3d[keep]
        span = max(float(vmax) - float(vmin), 1e-12)
        t = (vals - float(vmin)) / span
        rgb = _jet_rgb(t)
        rgba = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=-1)
        _bake_vertex_colors(obj, np.repeat(rgba, 8, axis=0))
    return obj


def _build_voxel_clip_group(grp):
    """Geometry it receives in, geometry clipped by a plane out - nothing
    else. The plane is defined by an Empty's world position + local Z axis:
    a face survives if dot(face_position - empty.location, empty.z_axis) <=
    0. 'Enable Clip' gates the whole thing so the cloud stays intact until
    you deliberately turn slicing on.
    """
    iface = grp.interface
    iface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    iface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    iface.new_socket("Clip Empty", in_out='INPUT',
                     socket_type='NodeSocketObject')
    enable_in = iface.new_socket("Enable Clip", in_out='INPUT',
                                 socket_type='NodeSocketBool')
    enable_in.default_value = False

    nodes, links = grp.nodes, grp.links
    nodes.clear()

    n_in = nodes.new('NodeGroupInput')
    n_in.location = (-700, 0)
    n_out = nodes.new('NodeGroupOutput')
    n_out.location = (500, 0)

    n_objinfo = nodes.new('GeometryNodeObjectInfo')
    n_objinfo.transform_space = 'ORIGINAL'
    n_objinfo.location = (-450, -250)

    n_rot = nodes.new('ShaderNodeVectorRotate')
    n_rot.rotation_type = 'EULER_XYZ'
    n_rot.inputs['Vector'].default_value = (0.0, 0.0, 1.0)
    n_rot.location = (-250, -250)

    n_pos = nodes.new('GeometryNodeInputPosition')
    n_pos.location = (-250, -450)

    n_sub = nodes.new('ShaderNodeVectorMath')
    n_sub.operation = 'SUBTRACT'
    n_sub.location = (-50, -350)

    n_dot = nodes.new('ShaderNodeVectorMath')
    n_dot.operation = 'DOT_PRODUCT'
    n_dot.location = (150, -350)

    n_cmp = nodes.new('FunctionNodeCompare')
    n_cmp.data_type = 'FLOAT'
    n_cmp.operation = 'GREATER_THAN'
    n_cmp.inputs['B'].default_value = 0.0
    n_cmp.location = (350, -350)

    n_del = nodes.new('GeometryNodeDeleteGeometry')
    n_del.domain = 'FACE'
    n_del.mode = 'ALL'
    n_del.location = (150, 100)

    n_switch = nodes.new('GeometryNodeSwitch')
    n_switch.input_type = 'GEOMETRY'
    n_switch.location = (300, 0)

    links.new(n_in.outputs['Clip Empty'], n_objinfo.inputs['Object'])
    links.new(n_objinfo.outputs['Rotation'], n_rot.inputs['Rotation'])
    links.new(n_pos.outputs['Position'], n_sub.inputs[0])
    links.new(n_objinfo.outputs['Location'], n_sub.inputs[1])
    links.new(n_sub.outputs['Vector'], n_dot.inputs[0])
    links.new(n_rot.outputs['Vector'], n_dot.inputs[1])
    links.new(n_dot.outputs['Value'], n_cmp.inputs['A'])
    links.new(n_cmp.outputs['Result'], n_del.inputs['Selection'])
    links.new(n_in.outputs['Geometry'], n_del.inputs['Geometry'])

    links.new(n_in.outputs['Enable Clip'], n_switch.inputs['Switch'])
    links.new(n_in.outputs['Geometry'], n_switch.inputs['False'])
    links.new(n_del.outputs['Geometry'], n_switch.inputs['True'])
    links.new(n_switch.outputs['Output'], n_out.inputs['Geometry'])


def _ensure_voxel_clip_group():
    grp = bpy.data.node_groups.get(_CLIP_GROUP)
    if grp is not None:
        return grp
    grp = bpy.data.node_groups.new(_CLIP_GROUP, 'GeometryNodeTree')
    try:
        _build_voxel_clip_group(grp)
    except Exception:
        # Don't leave a half-built group cached under this name - the next
        # call should retry from scratch instead of reusing something broken.
        bpy.data.node_groups.remove(grp)
        raise
    return grp


def _ensure_clip_empty(collection=None):
    obj = bpy.data.objects.get(_CLIP_EMPTY)
    if obj is not None:
        return obj
    obj = bpy.data.objects.new(_CLIP_EMPTY, None)
    obj.empty_display_type = 'PLAIN_AXES'
    obj.empty_display_size = 1.0
    target = collection if collection is not None else bpy.context.scene.collection
    target.objects.link(obj)
    return obj


def ensure_stress_cloud_geonodes(obj, collection=None, enable_clip=None):
    """Attach (or refresh) the voxel-clip modifier: once 'Enable Clip' is on,
    it deletes every cube face on one side of the Blendtopo_ClipPlane Empty.

    enable_clip : None leaves the modifier's current toggle alone (per-
    iteration refreshes shouldn't fight the user's own click); True/False
    forces it - used to flip the cross-section on automatically once a run
    finishes, so you see the cut immediately instead of a solid shell.

    Never raises - a Geometry Nodes API mismatch on some Blender version
    should just mean "no cross-section today", not a failed optimization run.
    """
    if not _HAS_BPY:
        return
    try:
        grp = _ensure_voxel_clip_group()
        empty = _ensure_clip_empty(collection)
        mod = obj.modifiers.get(_CLIP_MODIFIER)
        is_new = mod is None
        if mod is None or mod.type != 'NODES':
            mod = obj.modifiers.new(_CLIP_MODIFIER, 'NODES')
        mod.node_group = grp
        ids = {item.name: item.identifier for item in grp.interface.items_tree
              if getattr(item, "item_type", None) == 'SOCKET'
              and item.in_out == 'INPUT'}
        if "Clip Empty" in ids:
            mod[ids["Clip Empty"]] = empty
        if "Enable Clip" in ids:
            if enable_clip is not None:
                mod[ids["Enable Clip"]] = bool(enable_clip)
            elif is_new:
                mod[ids["Enable Clip"]] = False   # off by default - opt in
    except Exception as exc:  # noqa: BLE001
        print(f"[Blendtopo] voxel clip setup skipped: {exc}")
