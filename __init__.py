# Add-on metadata lives in blender_manifest.toml next to this file.

import bpy
import blf
import gpu
import math
import heapq
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix
from bpy_extras import view3d_utils
from bpy.props import (
    FloatVectorProperty, CollectionProperty, IntProperty,
    EnumProperty, PointerProperty, StringProperty, BoolProperty, FloatProperty,
)
from bpy.types import PropertyGroup, Operator, Panel

# Set to True to print island detection, bone-binding, and branching
# decisions to Blender's System Console (Window > Toggle System Console on
# Windows; on Mac/Linux, launch Blender from a terminal to see stdout there).
POSE_CHAIN_DEBUG = False

# Fixed internal default for clean_up_weight_islands, replacing what was
# previously a live N-panel slider (Island Cleanup). Removed from the UI
# because testing showed no visible difference across its whole range on
# a rest-pose mesh   vertex-group weight changes are only ever visible
# once the armature is actually posed away from rest, which made the
# slider look inert regardless of whether the underlying reassignment
# logic was working (it was, confirmed via console output showing real
# reassignments happening every time). The cleanup logic itself is
# unchanged and still runs on every build/refresh   only the ability to
# adjust its threshold live is gone.
ISLAND_CLEANUP_THRESHOLD = 0.15

# Vertex group the Armature modifier uses as its own mask (the modifier's
# "Vertex Group" field). Blender's Armature modifier divides each vertex's
# bone weights by their total before deforming, so a vertex with only 0.1
# of one bone's weight still moves 100% with that bone -- bone weights
# alone can only say WHICH bone a vertex follows, never HOW MUCH it moves.
# The modifier's mask group is what actually blends a vertex between its
# rest position and the posed one, so every falloff (into the unposed
# body, at the edge of a limb) is written here instead.
MASK_GROUP_NAME = "PCT_Deform_Mask"

# Measured limb radius is multiplied by this before being used as a bone's
# minimum reach, so bumps and folds on a sculpted surface stay inside it.
THICKNESS_MARGIN = 1.3


def dprint(*args):
    if POSE_CHAIN_DEBUG:
        print("[PoseChainTool]", *args)


# Cache of compute_bone_positions() output, keyed by mesh object name.
# Populated when a chain is built; consumed by _schedule_corridor_refresh
# when Weight Falloff or Falloff Softness change afterward, so those
# sliders can re-run the weighting pipeline live without re-extracting
# bone positions and mesh data from scratch. Cleared in cleanup_chain
# (Clear/Apply) so it never outlives the chain it belongs to.
_geodesic_cache = {}

# Mesh object names with a live-refresh already scheduled   prevents
# stacking multiple timers while a slider is actively being dragged; the
# single pending timer always reads whatever the CURRENT slider value is
# when it finally fires, so nothing is lost by coalescing like this.
_refresh_pending = set()

# Mesh object names currently being dragged in Object Mode (Fast Dragging).
# The N-panel normally only shows in Sculpt Mode; its poll() checks this so
# the panel doesn't vanish and reappear on every drag.
_object_mode_drag = set()


def _schedule_corridor_refresh(obj_name):
    """Debounced live-update: re-run apply_uniform_falloff_weights shortly
    after Weight Falloff or Falloff Softness changes, using the cached
    bone/mesh data from the last chain build. Debounced via
    bpy.app.timers so dragging a slider doesn't trigger a full mesh-wide
    recompute on every single intermediate value   only once, a short
    pause after the last change."""
    if obj_name in _refresh_pending:
        return
    _refresh_pending.add(obj_name)

    def _do_refresh():
        _refresh_pending.discard(obj_name)
        obj = bpy.data.objects.get(obj_name)
        if obj is None or obj.type != 'MESH':
            return None
        props = obj.pose_chain_tool
        if props.state != 'READY' or props.armature_obj is None:
            return None
        positions = _geodesic_cache.get(obj_name)
        if positions is None:
            return None
        bone_names = [f"Bone_{i}" for i in range(1, len(props.points))]
        parent_indices = [p.parent_index for p in props.points]
        try:
            _pruned, valid_by_bone, weight_by_bone = apply_uniform_falloff_weights(
                obj, props.armature_obj, bone_names, parent_indices,
                positions, props.weight_falloff, props.falloff_softness,
                props.joint_blend, props.auto_thickness)
            if ISLAND_CLEANUP_THRESHOLD > 0:
                clean_up_weight_islands(obj, bone_names, positions,
                                         valid_by_bone, weight_by_bone,
                                         props.falloff_softness,
                                         ISLAND_CLEANUP_THRESHOLD)
        except Exception as e:
            print(f"Pose Chain Tool: live weight update failed ({e})")
            return None
        obj.data.update()
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        return None  # don't repeat   this is a one-shot debounce timer

    bpy.app.timers.register(_do_refresh, first_interval=0.2)


def _on_corridor_setting_changed(self, context):
    """update= callback for weight_falloff and falloff_softness  
    schedules a live re-weighting so changes are visible on the mesh
    immediately, without needing to Clear Points and rebuild the whole
    chain."""
    obj = context.active_object
    if obj is None or obj.type != 'MESH':
        return
    _schedule_corridor_refresh(obj.name)


def _on_preserve_volume_changed(self, context):
    """update= callback for preserve_volume -- just a modifier flag, so no
    re-weighting is needed."""
    obj = context.active_object
    if obj is None or obj.type != 'MESH':
        return
    props = obj.pose_chain_tool
    mod = obj.modifiers.get(props.modifier_name) if props.modifier_name else None
    if mod is not None and mod.type == 'ARMATURE':
        mod.use_deform_preserve_volume = props.preserve_volume


def _bone_preview_items(self, context):
    """Dynamic dropdown listing every non-anchor point in the current
    chain, by the same Bone_i naming build_chain uses   recomputed on the
    fly from props.points rather than stored, since bone_names itself is
    always deterministic from the point count."""
    obj = context.object
    if obj is None:
        return [("NONE", "No Chain Built", "")]
    props = obj.pose_chain_tool
    n = len(props.points)
    if n < 2:
        return [("NONE", "No Chain Built", "")]
    return [(f"Bone_{i}", f"Point {i + 1}", f"Preview Bone_{i}'s current weight influence")
            for i in range(1, n)]


#                                      -
# Property groups
#                                      -

class PCT_Point(PropertyGroup):
    co: FloatVectorProperty(size=3, subtype='XYZ')
    # Index of the point this one branches off of. Unused/ignored for point 0
    # (the anchor). Defaults to -1 so an uninitialized value is obviously
    # wrong rather than silently pointing at point 0.
    parent_index: IntProperty(default=-1)


class PCT_GroupNameEntry(PropertyGroup):
    """Just tracks a vertex group name the tool created, so it can be
    cleaned up again on Clear/Apply   uses the .name field every
    PropertyGroup already has, nothing else needed."""
    pass


def _update_debug_logging(self, context):
    """update= callback for debug_logging   dprint() and the code paths
    it gates (e.g. the per-drag island diagnostic) check the module-level
    POSE_CHAIN_DEBUG flag directly, since many functions that call dprint
    don't have access to this property. This callback keeps that flag in
    sync with the checkbox."""
    global POSE_CHAIN_DEBUG
    POSE_CHAIN_DEBUG = self.debug_logging


class PCT_Properties(PropertyGroup):
    state: EnumProperty(
        items=[
            ('IDLE', "Idle", ""),
            ('PLACING', "Placing", ""),
            ('READY', "Ready", ""),  # chain built; modal is live and points are clickable
        ],
        default='IDLE',
    )
    points: CollectionProperty(type=PCT_Point)
    max_points: IntProperty(name="Max Points", default=3, min=2, max=17, soft_max=17,
                             description="Total points in the whole tree, including "
                                         "branches.")
    weight_falloff: FloatProperty(
        name="Weight Falloff", default=0.4, min=0.01, max=2.0,
        update=_on_corridor_setting_changed,
        description="How wide a bone's influence reaches from its line. "
                    "Raise for thicker limbs, lower if it reaches into a "
                    "neighbor's area. Updates live.")
    falloff_softness: FloatProperty(
        name="Falloff Softness", default=0.6, min=0.3, max=1.0,
        update=_on_corridor_setting_changed,
        description="How gradually weight fades near a joint or bone tip. "
                    "Low values can leave dead zones with no weight. Higher "
                    "values smooth it out. Updates live.")
    joint_blend: FloatProperty(
        name="Joint Blend", default=0.75, min=0.0, max=1.0,
        update=_on_corridor_setting_changed,
        description="How wide the smooth bend is at each joint, centered on "
                    "the joint. 0 is a hard crease. 1 blends across about "
                    "half of the shorter bone. Updates live.")
    auto_thickness: BoolProperty(
        name="Fit Limb Thickness", default=True,
        update=_on_corridor_setting_changed,
        description="Measure how thick the mesh is around each bone and "
                    "never let Weight Falloff reach less than that, so thick "
                    "limbs don't tear. Turn off if a bone grabs a nearby body "
                    "part. Updates live.")
    preserve_volume: BoolProperty(
        name="Preserve Volume", default=True,
        update=_on_preserve_volume_changed,
        description="Dual-quaternion deformation: keeps bent joints from "
                    "pinching or collapsing. Updates live.")
    preview_bone: EnumProperty(
        name="Preview Bone", items=_bone_preview_items,
        description="Bone to visualize with Preview Bone Influence, below.")
    armature_obj: PointerProperty(type=bpy.types.Object)
    created_vertex_groups: CollectionProperty(type=PCT_GroupNameEntry)
    modifier_name: StringProperty(default="")
    show_labels: BoolProperty(
        name="Show Point Numbers", default=True,
        description="Show the 1/2/3... order label on each point marker")
    show_help: BoolProperty(
        name="Show Help", default=True,
        description="Show the how-to-use instructions in this panel")
    fast_drag: BoolProperty(
        name="Fast Dragging", default=True,
        description="Switch to Object Mode while dragging a point and back to "
                    "Sculpt Mode when you let go. Much smoother on dense "
                    "meshes: Sculpt Mode makes every pose update far slower")
    debug_logging: BoolProperty(
        name="Debug Logging", default=False,
        update=_update_debug_logging,
        description="Print detailed weighting info to the console. Off by "
                    "default for performance. Turn on to troubleshoot.")


#                                      -
# Shared helpers
#                                      -

def mouse_in_viewport_window(context, event):
    """True if the mouse is currently over the 3D viewport's own drawing
    area (as opposed to the N-panel, header, or another editor).

    context.region stays pinned to the region the operator was invoked in
    (the viewport's WINDOW region) for the entire life of a modal operator  
    it never updates to reflect wherever the mouse currently is, which is
    exactly why checking context.region.type doesn't work here. What DOES
    change per-event is event.mouse_region_x/y, which Blender always reports
    relative to that same fixed region   so once the mouse leaves that
    region's actual rectangle (e.g. it's now over the N-panel sitting right
    next to it), those coordinates go negative or exceed the region's own
    width/height. That's a reliable, simple signal with no extra lookups
    needed.
    """
    region = context.region
    if region is None:
        return False
    return 0 <= event.mouse_region_x <= region.width and 0 <= event.mouse_region_y <= region.height


def get_props(context):
    obj = context.active_object
    if not obj or obj.type != 'MESH':
        return None, None
    return obj, obj.pose_chain_tool


def raycast_object(context, obj, mouse_coord):
    """Cast into the mesh and return a point at the CENTER of the mesh's
    volume under the cursor, not just the near surface. Finds the entry hit,
    then casts again from just past it to find the exit (back-facing) hit,
    and returns their midpoint. Falls back to the entry point if no exit is
    found (e.g. an open/non-manifold mesh at that spot).

    Raycasts against the EVALUATED mesh (what's actually displayed) first  
    the original (base) mesh data can momentarily lag behind what's on
    screen during/just after a sculpt stroke   and falls back to the
    original mesh if that ever comes back empty. Either way this uses
    Object.ray_cast() directly against a known object, which never returns
    an object identity to compare, sidestepping the evaluated-copy identity
    bug that scene-wide raycasting has entirely.
    """
    region = context.region
    rv3d = context.region_data
    origin_world = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_coord)
    direction_world = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_coord).normalized()

    mat_inv = obj.matrix_world.inverted()
    origin_local = mat_inv @ origin_world
    direction_local = (mat_inv.to_3x3() @ direction_world).normalized()

    try:
        obj_eval = obj.evaluated_get(context.evaluated_depsgraph_get())
    except RuntimeError:
        obj_eval = obj

    def try_cast(target, origin, direction):
        try:
            return target.ray_cast(origin, direction)
        except RuntimeError:
            return (False, None, None, -1)

    result, loc1, normal1, index1 = try_cast(obj_eval, origin_local, direction_local)
    if not result:
        result, loc1, normal1, index1 = try_cast(obj, origin_local, direction_local)
    if not result:
        return None

    # epsilon is computed in world units, then scaled down to local space so
    # it means the same thing regardless of the object's scale
    scale = obj.matrix_world.to_scale()
    avg_scale = max((abs(scale.x) + abs(scale.y) + abs(scale.z)) / 3.0, 1e-6)
    epsilon_local = (max(obj.dimensions.length, 0.01) * 1e-4) / avg_scale

    second_origin_local = loc1 + direction_local * epsilon_local
    result2, loc2, normal2, index2 = try_cast(obj_eval, second_origin_local, direction_local)
    if not result2:
        result2, loc2, normal2, index2 = try_cast(obj, second_origin_local, direction_local)

    mid_local = (loc1 + loc2) / 2.0 if result2 else loc1
    return obj.matrix_world @ mid_local


def get_live_chain_points_tree(armature_obj):
    """Current (possibly posed) world-space position of every chain point,
    read straight from the live pose bones. Returns a dict {point_index:
    Vector}, since a branching tree has no single "next" point to walk to."""
    result = {}
    if not armature_obj or armature_obj.name not in bpy.data.objects:
        return result
    mat = armature_obj.matrix_world
    pbones = armature_obj.pose.bones
    root = pbones.get("Bone_Root")
    if root is None:
        return result
    result[0] = mat @ root.head
    for b in pbones:
        if b.name == "Bone_Root" or not b.name.startswith("Bone_"):
            continue
        try:
            idx = int(b.name.split("_", 1)[1])
        except ValueError:
            continue
        result[idx] = mat @ b.tail
    return result


def get_children_map(props):
    """{parent_point_index: [child_point_index, ...]} from the placed points'
    parent_index fields."""
    children = {}
    for i, p in enumerate(props.points):
        if i == 0:
            continue
        children.setdefault(p.parent_index, []).append(i)
    return children


def get_descendants(idx, children_map):
    """idx plus every point transitively branching off of it."""
    result = {idx}
    stack = [idx]
    while stack:
        cur = stack.pop()
        for child in children_map.get(cur, []):
            if child not in result:
                result.add(child)
                stack.append(child)
    return result


def capture_pose_snapshot(armature_obj):
    """{bone_name: matrix_basis} for every pose bone. matrix_basis is each
    bone's own local pose offset, independent of the parent chain, so it can
    be captured/restored per-bone without needing to re-resolve hierarchy."""
    return {b.name: b.matrix_basis.copy() for b in armature_obj.pose.bones}


def restore_pose_snapshot(armature_obj, snapshot):
    for name, mat in snapshot.items():
        b = armature_obj.pose.bones.get(name)
        if b is not None:
            b.matrix_basis = mat.copy()


def circle_verts_2d(cx, cy, radius, segments=16):
    verts = [(cx, cy)]
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        verts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return verts


def draw_filled_circle_2d(shader, cx, cy, radius, color, segments=16):
    verts = circle_verts_2d(cx, cy, radius, segments)
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
    shader.uniform_float("color", color)
    batch.draw(shader)


def draw_chain_edges_3d(edges, edge_colors):
    """3D world-space lines, one per parent-child pair (POST_VIEW). edges is
    a list of (pos_a, pos_b) tuples; edge_colors is a matching list of
    (color_a, color_b) tuples so a line blends from the parent point's color
    to the child point's color, making an affected/unaffected boundary
    visually obvious even across branches."""
    if not edges:
        return
    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('ALWAYS')
    gpu.state.line_width_set(2.0)
    positions = []
    colors = []
    for (a, b), (ca, cb) in zip(edges, edge_colors):
        positions.extend([a, b])
        colors.extend([ca, cb])
    batch = batch_for_shader(shader, 'LINES', {"pos": positions, "color": colors})
    batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')


def draw_chain_markers_2d(context, points_by_index, colors_by_index, show_labels=True,
                           highlight_index=None):
    """Screen-space numbered circle for each point (POST_PIXEL). points_by_index
    and colors_by_index are {point_index: value} dicts. Uses real triangle
    geometry rather than GL point-sprites, which render inconsistently (or
    not at all) across GPUs/backends. highlight_index (the active branch
    parent during placement) gets an extra ring so it's obvious where new
    points will attach."""
    if not points_by_index:
        return
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')

    font_id = 0
    blf.size(font_id, 13)

    for i, p in points_by_index.items():
        p2d = view3d_utils.location_3d_to_region_2d(region, rv3d, p)
        if p2d is None:
            continue  # point is behind the camera
        color = colors_by_index.get(i, (1.0, 1.0, 1.0, 1.0))

        if i == highlight_index:
            draw_filled_circle_2d(shader, p2d.x, p2d.y, 13, (1.0, 0.9, 0.2, 0.9), 20)
        draw_filled_circle_2d(shader, p2d.x, p2d.y, 10, (1.0, 1.0, 1.0, 0.55), 18)
        draw_filled_circle_2d(shader, p2d.x, p2d.y, 7.5, color, 18)

        if show_labels:
            label = str(i + 1)
            tw, th = blf.dimensions(font_id, label)
            blf.color(font_id, 1, 1, 1, 1)
            blf.position(font_id, p2d.x - tw / 2, p2d.y - th / 2, 0)
            blf.draw(font_id, label)

    gpu.state.blend_set('NONE')


def connected_component_labels(n, edge_a, edge_b):
    """Connected components of a graph with n vertices and the given edges
    (two equal-length numpy int arrays). Returns a numpy int array where
    labels[v] is the smallest vertex index in v's component.

    Pure numpy (scipy isn't part of Blender's bundled Python, only numpy):
    a parallel union-find. Each round, every component root hooks onto the
    smallest root it shares an edge with, then pointer-jumping flattens
    the trees. Edges whose ends are already in the same component are
    dropped as it goes, so later rounds only touch edges that still matter.
    Replaces a per-vertex Python BFS (benchmarked ~1.5 s at 1M vertices)."""
    import numpy as np
    parent = np.arange(n, dtype=np.int64)
    a = np.asarray(edge_a, dtype=np.int64)
    b = np.asarray(edge_b, dtype=np.int64)
    while True:
        while True:
            grand = parent[parent]
            if np.array_equal(grand, parent):
                break
            parent = grand
        ra = parent[a]
        rb = parent[b]
        spanning = ra != rb
        if not spanning.any():
            return parent
        a = a[spanning]
        b = b[spanning]
        ra = ra[spanning]
        rb = rb[spanning]
        # Hook the larger root under the smaller one. Both are roots, and
        # lo < hi always, so this can never create a cycle.
        np.minimum.at(parent, np.maximum(ra, rb), np.minimum(ra, rb))


def get_mesh_islands(n_verts, edges):
    """Each disconnected mesh piece (e.g. separate objects Ctrl+J joined
    into this one) as its own numpy array of vertex indices. edges is the
    (edge_a, edge_b) pair from build_mesh_adjacency."""
    import numpy as np
    labels = connected_component_labels(n_verts, edges[0], edges[1])
    order = np.argsort(labels, kind='stable')
    _unique, starts = np.unique(labels[order], return_index=True)
    return np.split(order, starts[1:])


def build_mesh_adjacency(mesh, mesh_mat):
    """Mesh connectivity as a pair of numpy int arrays (edge_a, edge_b),
    one entry per edge, plus a numpy (V,3) array of world-space vertex
    coordinates.

    Everything here is bulk foreach_get + numpy. It used to also build a
    Python list-of-lists adjacency (one Python append per edge end, ~2.5 s
    at 1M vertices); nothing needs that anymore now that island detection
    works directly on the edge arrays."""
    import numpy as np
    n = len(mesh.vertices)
    m = len(mesh.edges)

    flat_co = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get('co', flat_co)
    local_co = flat_co.reshape(n, 3)
    mat = np.array(mesh_mat, dtype=np.float64)
    homog = np.hstack([local_co, np.ones((n, 1))])
    world_co = (homog @ mat.T)[:, :3]

    flat_edges = np.empty(m * 2, dtype=np.int64)
    if m > 0:
        mesh.edges.foreach_get('vertices', flat_edges)
    edge_verts = flat_edges.reshape(m, 2)
    edges = (edge_verts[:, 0].copy(), edge_verts[:, 1].copy())
    return edges, world_co


def compute_bone_positions(mesh_obj, arm_obj, bone_names):
    """Lightweight replacement for the old Dijkstra-based geodesic search:
    just extracts each bone's head/tail world positions and path_length,
    plus the mesh's own world-space vertex coordinates (as a numpy array)
    and edge connectivity (kept only for clean_up_weight_islands' island
    detection   the weighting itself below no longer needs it at all).
    No graph search happens here.

    Returns {"coords_np": numpy (V,3) array, "edges": (edge_a, edge_b)
    numpy arrays, "bones": {name: {"path_length", "head_pos", "tail_pos",
    "thickness", "tip_thickness"}, ...}}.
    """
    mesh = mesh_obj.data
    mesh_mat = mesh_obj.matrix_world
    edges, coords_np = build_mesh_adjacency(mesh, mesh_mat)

    bones = {}
    for name in bone_names:
        pbone = arm_obj.pose.bones.get(name)
        if pbone is None:
            continue
        head = arm_obj.matrix_world @ pbone.head
        tail = arm_obj.matrix_world @ pbone.tail
        path_length = max((tail - head).length, 1e-5)
        bones[name] = {"path_length": path_length, "head_pos": head, "tail_pos": tail,
                       "thickness": estimate_limb_thickness(mesh_obj, head, tail),
                       "tip_thickness": estimate_tip_thickness(mesh_obj, head, tail)}
        dprint(f"  {name}: measured limb radius={bones[name]['thickness']:.4f}, "
               f"radius at tail={bones[name]['tip_thickness']:.4f}")
    return {"coords_np": coords_np, "edges": edges, "bones": bones}


def _median_ray_distance(mesh_obj, origins_and_dirs):
    """Median world-space distance at which rays [(origin_world, dir_world),
    ...] first hit mesh_obj, or 0.0 if none hit. The median shrugs off the
    few rays that escape sideways into a torso or out through an open
    edge."""
    mat = mesh_obj.matrix_world
    mat_inv = mat.inverted()
    hits = []
    for origin_world, dir_world in origins_and_dirs:
        origin_local = mat_inv @ origin_world
        dir_local = (mat_inv.to_3x3() @ dir_world).normalized()
        try:
            ok, loc, _normal, _index = mesh_obj.ray_cast(origin_local, dir_local)
        except RuntimeError:
            ok = False
        if ok:
            hits.append(((mat @ loc) - origin_world).length)
    if not hits:
        return 0.0
    hits.sort()
    return hits[len(hits) // 2]


def estimate_limb_thickness(mesh_obj, head_world, tail_world, samples=(0.3, 0.5, 0.7),
                            directions=12):
    """Radius of the mesh around a bone, in world units, from rays cast
    perpendicular to it at a few points along it. Points sit at the CENTER
    of the volume (see raycast_object), so the rays start inside the limb
    and hit its own surface first. Measured once per build and cached with
    the bone positions, so the live sliders never re-cast."""
    axis = tail_world - head_world
    if axis.length < 1e-9:
        return 0.0
    axis = axis.normalized()
    side = axis.orthogonal().normalized()
    up = axis.cross(side).normalized()
    rays = []
    for t in samples:
        origin = head_world.lerp(tail_world, t)
        for k in range(directions):
            a = 2 * math.pi * k / directions
            rays.append((origin, side * math.cos(a) + up * math.sin(a)))
    return _median_ray_distance(mesh_obj, rays)


def estimate_tip_thickness(mesh_obj, head_world, tail_world, directions=32):
    """Radius of the mesh around a bone's TAIL point, from rays cast in every
    direction except back down the bone -- sizes whatever is at the end of a
    limb (a hand, a foot, a head), which is often thicker than the limb
    itself."""
    axis = tail_world - head_world
    if axis.length < 1e-9:
        return 0.0
    axis = axis.normalized()
    rays = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for k in range(directions):
        z = 1.0 - 2.0 * (k + 0.5) / directions
        r = math.sqrt(max(0.0, 1.0 - z * z))
        d = Vector((r * math.cos(golden * k), r * math.sin(golden * k), z))
        if d.dot(axis) < -0.5:
            continue
        rays.append((tail_world, d))
    return _median_ray_distance(mesh_obj, rays)


def _smoothstep(x):
    import numpy as np
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _write_group(mesh_obj, name, indices, weights):
    """Replace vertex group `name` with exactly these (index, weight) pairs.

    The group is removed and recreated rather than edited -- the Armature
    modifier finds groups by NAME, so a fresh group is indistinguishable to
    it, and one remove() is far cheaper than scanning every vertex for old
    membership. Weights are quantized to 255 levels so vg.add() runs once
    per distinct level instead of once per vertex."""
    import numpy as np
    QUANT_LEVELS = 255
    vg = mesh_obj.vertex_groups.get(name)
    if vg is not None:
        mesh_obj.vertex_groups.remove(vg)
    vg = mesh_obj.vertex_groups.new(name=name)
    if len(indices) == 0:
        return vg
    quantized = np.clip(np.round(weights * QUANT_LEVELS).astype(np.int32), 1, QUANT_LEVELS)
    order = np.argsort(quantized)
    sorted_idx = indices[order]
    sorted_q = quantized[order]
    unique_q, group_start = np.unique(sorted_q, return_index=True)
    group_start = np.append(group_start, len(sorted_q))
    for gi in range(len(unique_q)):
        s, e = group_start[gi], group_start[gi + 1]
        vg.add(sorted_idx[s:e].tolist(), float(unique_q[gi]) / QUANT_LEVELS, 'REPLACE')
    return vg


def apply_uniform_falloff_weights(mesh_obj, arm_obj, bone_names, parent_indices, positions,
                                   falloff_multiplier=0.4, falloff_softness=0.6, joint_blend=0.75,
                                   auto_thickness=True):
    """Weights the mesh in two independent layers, with numpy:

    1. WHICH bone(s) each vertex follows -> the Bone_i vertex groups.
       Each vertex follows the bone whose segment is closest. Between a bone
       and its DIRECT parent, the hand-off is a smooth blend centered on the
       joint plane (halfway between the two bone directions), spanning
       joint_blend * the shorter bone's length; joint_blend=0 is a hard
       seam. Only direct parent/child pairs ever share a vertex -- siblings
       and other branches never do, so posing one branch can't drag
       another. Normalized per vertex (Blender normalizes them anyway).

    2. HOW MUCH each vertex deforms at all -> MASK_GROUP_NAME, the Armature
       modifier's mask (see MASK_GROUP_NAME for why this can't live in the
       bone weights). 1 inside a bone's reach, fading smoothly to 0 over
       falloff_softness:
         - radially, past bone length * falloff_multiplier -- never less
           than the measured limb thickness when auto_thickness is on;
         - axially, behind the head of a bone that hangs straight off the
           anchor, so the body around the anchor bends a little instead of
           tearing.
       A leaf bone (no children) does NOT fade past its tail: whatever lies
       beyond the last point (a hand, a head) moves with it fully, out to
       the measured size of that end.

    Returns (adjusted_count, valid_by_bone, weight_by_bone): which vertices
    each bone actually moves, and each bone's effective influence (follow
    share * mask), both reused by clean_up_weight_islands.
    """
    import numpy as np

    coords_np = positions["coords_np"]
    bone_positions = positions["bones"]
    n_verts = coords_np.shape[0]

    falloff_multiplier = max(0.01, falloff_multiplier)
    falloff_softness = max(0.05, falloff_softness)
    joint_blend = max(0.0, min(1.0, joint_blend))

    point_idx_of = {}
    for name in bone_names:
        try:
            point_idx_of[name] = int(name.split("_", 1)[1])
        except ValueError:
            pass

    active_names = [name for name in bone_names if name in bone_positions
                    and point_idx_of.get(name) is not None]
    B = len(active_names)
    if B == 0 or n_verts == 0:
        return 0, {}, {}

    # parent_bi[b] = index of bone b's direct parent BONE, or -1 when b hangs
    # straight off the anchor (Bone_Root holds no weight, so it never counts).
    point_to_bi = {point_idx_of[name]: i for i, name in enumerate(active_names)}
    parent_bi = [-1] * B
    children_bi = [[] for _ in range(B)]
    for bi, name in enumerate(active_names):
        pt = point_idx_of[name]
        parent_pt = parent_indices[pt] if 0 <= pt < len(parent_indices) else -1
        pb = point_to_bi.get(parent_pt, -1) if parent_pt > 0 else -1
        parent_bi[bi] = pb
        if pb >= 0:
            children_bi[pb].append(bi)

    heads = np.array([list(bone_positions[n]["head_pos"]) for n in active_names], dtype=np.float64)
    tails = np.array([list(bone_positions[n]["tail_pos"]) for n in active_names], dtype=np.float64)
    lengths = np.array([bone_positions[n]["path_length"] for n in active_names], dtype=np.float64)
    axes = (tails - heads) / lengths[:, None]

    dist = np.empty((B, n_verts), dtype=np.float64)
    mask = np.zeros(n_verts, dtype=np.float64)
    for bi, name in enumerate(active_names):
        bdata = bone_positions[name]
        L = lengths[bi]
        t_raw = (coords_np - heads[bi]) @ axes[bi] / L          # 0 at head, 1 at tail
        t_clamped = np.clip(t_raw, 0.0, 1.0)
        closest = heads[bi] + (t_clamped * L)[:, None] * axes[bi]
        d = np.linalg.norm(coords_np - closest, axis=1)
        dist[bi] = d

        reach = L * falloff_multiplier
        if auto_thickness:
            reach = max(reach, bdata.get("thickness", 0.0) * THICKNESS_MARGIN)
        reach_v = np.full(n_verts, reach)
        if not children_bi[bi] and auto_thickness:
            # Leaf: grow the reach toward the tail so the whole end of the
            # limb (sized by rays cast from the tail) is covered, starting
            # one end-radius before the tail so there's no step at the tail.
            tip_reach = max(reach, bdata.get("tip_thickness", 0.0) * THICKNESS_MARGIN)
            if tip_reach > reach:
                t0 = max(0.0, 1.0 - tip_reach / L)
                ramp = _smoothstep((t_raw - t0) / max(1.0 - t0, 1e-6))
                reach_v = reach + (tip_reach - reach) * ramp
        radial = 1.0 - _smoothstep((d - reach_v) / np.maximum(reach_v * falloff_softness, 1e-6))

        if parent_bi[bi] < 0:
            behind = -t_raw * L                                   # > 0 behind the head
            axial = 1.0 - _smoothstep(behind / max(L * falloff_softness, 1e-6))
        else:
            axial = 1.0
        np.maximum(mask, radial * axial, out=mask)

        dprint(f"  {name}: path_length={L:.4f}, reach={reach:.4f} "
               f"(falloff {L * falloff_multiplier:.4f}, measured radius "
               f"{bdata.get('thickness', 0.0):.4f}), leaf={not children_bi[bi]}")

    # Downstream share at each bone's HEAD joint (bones with a parent bone):
    # 0.5 exactly on the joint plane, 1 a half-width down this bone, 0 a
    # half-width up the parent.
    joint_share = [None] * B
    for bi in range(B):
        pb = parent_bi[bi]
        if pb < 0:
            continue
        normal = axes[pb] + axes[bi]
        nl = np.linalg.norm(normal)
        normal = normal / nl if nl > 1e-6 else axes[bi]
        half_width = max(joint_blend * 0.5 * min(lengths[pb], lengths[bi]), 1e-6)
        s = (coords_np - heads[bi]) @ normal
        joint_share[bi] = _smoothstep(0.5 + s / (2.0 * half_width))

    primary = np.argmin(dist, axis=0)
    share = np.zeros((B, n_verts), dtype=np.float64)
    for bi in range(B):
        sel = primary == bi
        if not sel.any():
            continue
        own = np.ones(int(sel.sum()))
        pb = parent_bi[bi]
        if pb >= 0:
            js = joint_share[bi][sel]
            own *= js
            share[pb, sel] += 1.0 - js
        for ci in children_bi[bi]:
            cs = joint_share[ci][sel]
            share[ci, sel] += cs
            own *= 1.0 - cs
        share[bi, sel] += own
    share /= np.maximum(share.sum(axis=0), 1e-12)[None, :]

    effective = share * mask[None, :]
    final_valid = effective > 1e-4

    adjusted = 0
    for bi, name in enumerate(active_names):
        idx = np.nonzero(final_valid[bi])[0]
        _write_group(mesh_obj, name, idx, share[bi, idx])
        adjusted += len(idx)

    mask_idx = np.nonzero(mask > 0.5 / 255)[0]
    _write_group(mesh_obj, MASK_GROUP_NAME, mask_idx, mask[mask_idx])
    for mod in mesh_obj.modifiers:
        if mod.type == 'ARMATURE' and mod.object == arm_obj:
            mod.vertex_group = MASK_GROUP_NAME

    dprint(f"apply_uniform_falloff_weights: {adjusted} influence(s) written, "
           f"{len(mask_idx)} vertex(es) deform")
    valid_by_bone = {name: final_valid[bi] for bi, name in enumerate(active_names)}
    weight_by_bone = {name: effective[bi] for bi, name in enumerate(active_names)}
    return adjusted, valid_by_bone, weight_by_bone


def clean_up_weight_islands(mesh_obj, bone_names, positions, valid_by_bone, weight_by_bone,
                             falloff_softness, min_island_fraction=0.15):
    """Post-process pass, run AFTER apply_uniform_falloff_weights: for each bone,
    find every separate connected patch of vertices that bone currently
    has weight on, keep the largest one, and either remove or REASSIGN any
    other patch that's smaller than min_island_fraction of that largest
    patch's size.

    REASSIGN, NOT JUST DELETE: confirmed directly (via the per-island
    fallback-weight diagnostic) that on some meshes, EVERY stray island
    encountered has a genuine 0.0 fallback weight from every other bone  
    not just a low one, an exact zero. Simply deleting that weight would
    guarantee orphaning every time, making the feature a no-op on those
    meshes (safely declining to help, but not helping either). Instead, a
    vertex that would otherwise be orphaned gets reassigned to whichever
    OTHER bone is spatially closest to it (straight-line distance to that
    bone's own segment), with a freshly computed weight for that bone at
    that position   using the same falloff formula apply_uniform_falloff_weights
    itself uses, just without that bone's own radius gate, since this is a
    deliberate rescue for a vertex outside every bone's normal reach. This
    fixes the visual disconnect (the vertex now belongs to some real bone
    instead of a wrong one) without ever creating a true zero-weight vertex.

    This is a genuinely different technique from everything tried before
    it   every earlier fix (seed strategy, radius tuning, fairness
    normalization, the heat-diffusion hybrid) tried to get the weight
    COMPUTATION right in the first place, by changing how distance itself
    is measured. This doesn't touch that computation at all   it cleans
    up the RESULT afterward, the same way "remove islands" / "smooth
    weights" tools work in Maya, Blender's own weight paint, and other
    production rigging pipelines: a small, disconnected patch of weight
    sitting apart from a bone's main region is a well-known, common
    artifact of ANY automatic weighting method (not specific to this
    tool's geodesic approach), and the standard fix in the industry is
    exactly this   clean it up structurally after the fact, rather than
    trying to prevent every possible cause of it during computation.

    "Connected" here means connected THROUGH VERTICES THIS BONE ALREADY
    HAS WEIGHT ON, via real mesh edges   not connected through the whole
    mesh. A bone's main region and a small stray splotch elsewhere can
    each be internally contiguous while having no edge path between them
    THAT STAYS WITHIN this bone's own weighted vertices, which is exactly
    what makes them separate islands here even though the whole mesh
    itself is one connected piece.

    min_island_fraction is deliberately proportional (a fraction of the
    largest island's own size), not an absolute vertex count, so it scales
    sensibly whether a bone's main region has 50 vertices or 5000.

    positions is the same dict compute_bone_positions returns (needs
    coords_np and each bone's head/tail for the reassignment distance
    calculation). valid_by_bone/weight_by_bone are {bone_name: numpy
    array} from apply_uniform_falloff_weights, reused directly here
    rather than rediscovered by scanning every vertex's groups again.
    Island detection and reassignment are pure numpy (see
    connected_component_labels)   scipy isn't part of Blender's bundled
    Python, only numpy is. This used to be a per-vertex Python BFS plus a
    per-vertex mathutils loop for reassignment (benchmarked ~0.9-1.7 s at
    1M vertices, on every slider change).

    Returns the number of vertices that had weight removed or reassigned.
    """
    import numpy as np

    vgs = mesh_obj.vertex_groups
    coords_np = positions["coords_np"]
    bone_positions = positions["bones"]
    edge_a, edge_b = positions["edges"]
    n_verts = coords_np.shape[0]
    removed_total = 0
    dprint(f"clean_up_weight_islands: min_island_fraction={min_island_fraction}")

    # A vertex needs at least this much weight from some OTHER bone to
    # count as having a genuine fallback   NOT just any nonzero overlap.
    # A sibling bone sharing a branch point can legitimately have some
    # small, technically-nonzero weight reaching the same region (that's
    # normal, expected corridor overlap near a joint), which isn't the
    # same thing as that vertex actually being safe to hand off to it.
    FALLBACK_WEIGHT_THRESHOLD = 0.05

    bone_order = [n for n in bone_names if n in weight_by_bone]
    weight_stack = np.stack([weight_by_bone[n] for n in bone_order]) if bone_order else None

    # Per-bone segment data for reassignment, in bone_positions order (the
    # order the old per-vertex search walked them in, so ties resolve the
    # same way: first bone at the minimum distance wins).
    seg_names = list(bone_positions.keys())
    seg_head = np.array([tuple(bone_positions[n]["head_pos"]) for n in seg_names],
                        dtype=np.float64).reshape(-1, 3)
    seg_tail = np.array([tuple(bone_positions[n]["tail_pos"]) for n in seg_names],
                        dtype=np.float64).reshape(-1, 3)
    seg_len = np.array([bone_positions[n]["path_length"] for n in seg_names],
                       dtype=np.float64)

    def best_reassignment(vert_idx, exclude_name):
        """For each vertex in vert_idx: the closest OTHER bone (straight-line
        distance to its segment) and that bone's falloff weight there.
        Returns (target_seg_index array, weight array); index -1 = none."""
        cand = [i for i, n in enumerate(seg_names) if n != exclude_name]
        k = len(vert_idx)
        if not cand or k == 0:
            return np.full(k, -1, dtype=np.int64), np.zeros(k)
        cand = np.array(cand, dtype=np.int64)
        p = coords_np[vert_idx]                                   # (k, 3)
        head = seg_head[cand]                                     # (c, 3)
        ab = seg_tail[cand] - head                                # (c, 3)
        ab_len_sq = np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-12)
        rel = p[None, :, :] - head[:, None, :]                    # (c, k, 3)
        t_raw = np.einsum('ckj,cj->ck', rel, ab) / ab_len_sq[:, None]
        t_cl = np.clip(t_raw, 0.0, 1.0)
        closest = head[:, None, :] + t_cl[:, :, None] * ab[:, None, :]
        dist = np.linalg.norm(p[None, :, :] - closest, axis=2)    # (c, k)
        best = np.argmin(dist, axis=0)                            # (k,)
        cols = np.arange(k)
        t_raw_b = t_raw[best, cols]
        t_cl_b = t_cl[best, cols]
        path_len = seg_len[cand][best]
        band = np.maximum(path_len * falloff_softness, 1e-6)
        overshoot = np.where(t_raw_b < 0.0, -t_raw_b * path_len,
                             np.where(t_raw_b > 1.0, (t_raw_b - 1.0) * path_len, 0.0))
        tt = np.clip(overshoot / band, 0.0, 1.0)
        dir_factor = np.where(overshoot <= 0.0, 1.0, 1.0 - tt * tt * (3.0 - 2.0 * tt))
        weight = t_cl_b * t_cl_b * (3.0 - 2.0 * t_cl_b) * dir_factor
        return cand[best], weight

    # Reassignments are collected and applied all at once at the end so a
    # vertex being reassigned OUT of one bone in this same pass doesn't
    # interfere with another bone's island detection still in progress.
    reassign_verts = []    # list of numpy arrays, in the order found
    reassign_target = []   # matching arrays of seg_names indices
    reassign_weight = []   # matching arrays of weights

    for name in bone_names:
        vg = vgs.get(name)
        if vg is None:
            continue
        valid_mask = valid_by_bone.get(name)
        if valid_mask is None or not valid_mask.any():
            continue

        weighted_idx = np.nonzero(valid_mask)[0]
        # Islands = connected components of the subgraph made of this
        # bone's weighted vertices and the mesh edges between them,
        # computed on a compacted index space (only weighted vertices).
        keep = valid_mask[edge_a] & valid_mask[edge_b]
        remap = np.full(n_verts, -1, dtype=np.int64)
        remap[weighted_idx] = np.arange(len(weighted_idx))
        labels = connected_component_labels(len(weighted_idx),
                                            remap[edge_a[keep]], remap[edge_b[keep]])
        _roots, inverse, counts = np.unique(labels, return_inverse=True,
                                            return_counts=True)
        if len(counts) <= 1:
            continue

        largest_size = int(counts.max())
        min_size = max(1, int(largest_size * min_island_fraction))
        island_size_of_vert = counts[inverse]
        small_verts = weighted_idx[island_size_of_vert < min_size]
        dprint(f"  {name}: {len(counts)} island(s), "
               f"sizes={sorted(counts.tolist(), reverse=True)[:20]}, "
               f"min_size={min_size} (largest * {min_island_fraction})")
        if len(small_verts) == 0:
            continue

        if weight_stack is not None and name in bone_order and weight_stack.shape[0] > 1:
            idx = bone_order.index(name)
            other_max = np.max(np.delete(weight_stack[:, small_verts], idx, axis=0), axis=0)
            safe = small_verts[other_max >= FALLBACK_WEIGHT_THRESHOLD]
            to_reassign = small_verts[other_max < FALLBACK_WEIGHT_THRESHOLD]
        else:
            safe = small_verts
            to_reassign = small_verts[:0]

        to_remove = [safe]
        if len(to_reassign):
            target_idx, target_weight = best_reassignment(to_reassign, name)
            ok = (target_idx >= 0) & (target_weight > 1e-6)
            if ok.any():
                reassign_verts.append(to_reassign[ok])
                reassign_target.append(target_idx[ok])
                reassign_weight.append(target_weight[ok])
                to_remove.append(to_reassign[ok])
            dprint(f"  {name}: reassigning {int(ok.sum())} vertex(es) that would "
                   f"otherwise be orphaned to whichever other bone is spatially closest.")

        to_remove = np.concatenate(to_remove)
        if len(to_remove):
            vg.remove(to_remove.tolist())
            removed_total += len(to_remove)
            dprint(f"  {name}: removed/reassigned {len(to_remove)} vertex(es) "
                   f"(largest region kept: {largest_size} vertices)")

    # Apply all reassignments in one pass at the end. 'REPLACE' means that
    # if the same vertex was reassigned to the same bone more than once,
    # the LAST assignment wins   keep only that one, then batch vg.add()
    # calls by identical weight value.
    n_reassigned = 0
    if reassign_verts:
        all_v = np.concatenate(reassign_verts)
        all_t = np.concatenate(reassign_target)
        all_w = np.concatenate(reassign_weight)
        for ti in np.unique(all_t):
            vg = vgs.get(seg_names[ti])
            if vg is None:
                continue
            sel = all_t == ti
            v = all_v[sel]
            w = all_w[sel]
            rev_unique, rev_first = np.unique(v[::-1], return_index=True)
            last = len(v) - 1 - rev_first
            v = v[last]
            w = w[last]
            n_reassigned += len(v)
            order = np.argsort(w, kind='stable')
            v = v[order]
            w = w[order]
            uw, starts = np.unique(w, return_index=True)
            starts = np.append(starts, len(w))
            for gi in range(len(uw)):
                vg.add(v[starts[gi]:starts[gi + 1]].tolist(), float(uw[gi]), 'REPLACE')

    if removed_total:
        dprint(f"clean_up_weight_islands: {removed_total} vertex(es) cleaned up total "
               f"({n_reassigned} reassigned)")
    return removed_total


def build_chain(context, mesh_obj):
    """Create a hidden Armature matching the placed point TREE, parent the
    mesh to it with Automatic Weights, and record it on the tool properties.
    Returns True on success. Leaves the mesh back in Sculpt Mode either way."""
    props = mesh_obj.pose_chain_tool
    points_world = [mesh_obj.matrix_world @ Vector(p.co) for p in props.points]
    parent_indices = [p.parent_index for p in props.points]
    if len(points_world) < 2:
        return False

    dprint(f"build_chain: {len(points_world)} points, "
           f"parent_indices={parent_indices}")

    arm_obj = None
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except RuntimeError as e:
        print(f"Pose Chain Tool: could not leave Sculpt Mode to build the rig: {e}")
        return False

    try:
        arm_data = bpy.data.armatures.new(f"{mesh_obj.name}_PoseChainData")
        arm_obj = bpy.data.objects.new(f"{mesh_obj.name}_PoseChainRig", arm_data)
        context.collection.objects.link(arm_obj)

        context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='EDIT')
        ebones = arm_data.edit_bones

        # Synthetic root bone at the anchor   every point attached directly
        # to the anchor (parent_index == 0) parents under THIS bone, so
        # translating it moves the entire tree uniformly regardless of how
        # many branches start right at the anchor.
        root_bone = ebones.new("Bone_Root")
        root_bone.head = points_world[0]
        direction = Vector((0.0, 0.0, 1.0))
        if len(points_world) > 1:
            d = points_world[1] - points_world[0]
            if d.length > 1e-6:
                direction = d.normalized()
        epsilon = max(mesh_obj.dimensions.length, 0.01) * 0.01
        root_bone.tail = points_world[0] + direction * epsilon

        bones_by_point = {0: root_bone}
        # parent_index always refers to an earlier point (you can only pick an
        # already-placed point as a branch parent), so processing in index
        # order guarantees each point's parent bone already exists.
        for i in range(1, len(points_world)):
            parent_point_idx = parent_indices[i]
            parent_bone = bones_by_point.get(parent_point_idx, root_bone)
            b = ebones.new(f"Bone_{i}")
            b.head = points_world[parent_point_idx]
            b.tail = points_world[i]
            b.parent = parent_bone
            # Only really "connected" (head == parent's own tail) when the
            # parent isn't the synthetic root, whose tail is a tiny offset
            # from the anchor rather than the anchor itself.
            b.use_connect = (parent_point_idx != 0)
            bones_by_point[i] = b
        bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        arm_obj.select_set(True)
        context.view_layer.objects.active = arm_obj

        # Empty groups, not Automatic Weights: apply_uniform_falloff_weights
        # below recreates every Bone_i group from scratch, so heat weights
        # were computed (slowly, and sometimes failing outright on sculpt
        # topology) only to be thrown away.
        bpy.ops.object.parent_set(type='ARMATURE_NAME')

        # Bone_Root is a REAL bone in the armature (it has to be, for FK
        # parenting to work), but it's a pure structural pivot: the anchor
        # should move the CHAIN, with its children handling all actual
        # deformation. Bone_Root must never hold any direct mesh weight;
        # only its TRANSFORM should ever reach the mesh, cascaded through
        # normal FK parenting to its child bones.
        root_vg = mesh_obj.vertex_groups.get("Bone_Root")
        if root_vg is not None:
            mesh_obj.vertex_groups.remove(root_vg)
            dprint("build_chain: removed Bone_Root's own vertex group   "
                   "the anchor should never hold direct mesh weight, only "
                   "cascade its transform to child bones via FK")

        bone_names = [f"Bone_{i}" for i in range(1, len(points_world))]

        # Bone positions, mesh coordinates, edge adjacency (for island
        # cleanup) and measured limb thickness, cached so the live sliders
        # can re-weight without re-extracting any of it.
        positions = compute_bone_positions(mesh_obj, arm_obj, bone_names)
        _geodesic_cache[mesh_obj.name] = positions

        pruned, valid_by_bone, weight_by_bone = apply_uniform_falloff_weights(
            mesh_obj, arm_obj, bone_names, parent_indices, positions,
            props.weight_falloff, props.falloff_softness, props.joint_blend,
            props.auto_thickness)
        if pruned:
            print(f"Pose Chain Tool: weighted {pruned} vertex influence(s) to this chain.")

        props.created_vertex_groups.clear()
        for name in bone_names + [MASK_GROUP_NAME]:
            if mesh_obj.vertex_groups.get(name) is not None:
                entry = props.created_vertex_groups.add()
                entry.name = name

        if ISLAND_CLEANUP_THRESHOLD > 0:
            cleaned = clean_up_weight_islands(mesh_obj, bone_names, positions,
                                               valid_by_bone, weight_by_bone,
                                               props.falloff_softness,
                                               ISLAND_CLEANUP_THRESHOLD)
            if cleaned:
                print(f"Pose Chain Tool: cleaned up {cleaned} vertex(es) of stray, "
                      f"disconnected weight patches separated from each bone's main region.")

        arm_obj.hide_set(True)
        arm_obj.hide_select = True

        mod = next((m for m in mesh_obj.modifiers
                    if m.type == 'ARMATURE' and m.object == arm_obj), None)
        if mod is not None:
            mod.use_deform_preserve_volume = props.preserve_volume
        props.modifier_name = mod.name if mod else ""
        props.armature_obj = arm_obj

        context.view_layer.objects.active = mesh_obj
        mesh_obj.select_set(True)
        bpy.ops.object.mode_set(mode='SCULPT')
        return True

    except Exception as e:
        print(f"Pose Chain Tool: failed to build chain: {e}")
        if arm_obj is not None and arm_obj.name in bpy.data.objects:
            try:
                bpy.data.objects.remove(arm_obj, do_unlink=True)
            except Exception:
                pass
        try:
            context.view_layer.objects.active = mesh_obj
            mesh_obj.select_set(True)
            bpy.ops.object.mode_set(mode='SCULPT')
        except Exception:
            pass
        return False


def cleanup_chain(context, mesh_obj):
    """Remove the temp rig, its vertex groups, and reset tool state. Does not
    touch mesh geometry   call this AFTER modifier_apply if you want to keep
    the deformation, or directly if you want to discard it (the modifier
    just loses its target and stops deforming once the armature is gone)."""
    props = mesh_obj.pose_chain_tool
    arm_obj = props.armature_obj
    if arm_obj and arm_obj.name in bpy.data.objects:
        try:
            bpy.data.objects.remove(arm_obj, do_unlink=True)
        except Exception as e:
            print(f"Pose Chain Tool: failed to remove rig: {e}")
    if props.modifier_name and props.modifier_name in mesh_obj.modifiers:
        try:
            mesh_obj.modifiers.remove(mesh_obj.modifiers[props.modifier_name])
        except Exception:
            pass
    removed_groups = 0
    for entry in props.created_vertex_groups:
        vg = mesh_obj.vertex_groups.get(entry.name)
        if vg is not None:
            try:
                mesh_obj.vertex_groups.remove(vg)
                removed_groups += 1
            except Exception:
                pass
    if removed_groups:
        dprint(f"cleanup_chain: removed {removed_groups} temporary vertex group(s)")
    _geodesic_cache.pop(mesh_obj.name, None)
    props.created_vertex_groups.clear()
    props.points.clear()
    props.armature_obj = None
    props.modifier_name = ""
    props.state = 'IDLE'


#                                      -
# The single continuous session operator: place points (with branching),
# then pose them, all in one modal that never lets the on-screen points
# disappear.
#                                      -

class PCT_OT_activate(Operator):
    bl_idname = "pct.activate"
    bl_label = "Activate Pose Chain"
    bl_description = ("Click the mesh to place chain points (anchor first). "
                       "Click an EXISTING point to branch a new chain off of "
                       "it. Once built, click any point anytime to pose it "
                       "(clicking elsewhere still sculpts normally). "
                       "Use Clear Points / Apply in the N-panel to finish.")

    @classmethod
    def poll(cls, context):
        obj, props = get_props(context)
        return context.mode == 'SCULPT' and props is not None

    def invoke(self, context, event):
        obj, props = get_props(context)
        if props.state != 'IDLE':
            self.report({'WARNING'}, "A pose chain is already active")
            return {'CANCELLED'}

        self.obj = obj
        props.points.clear()
        props.armature_obj = None
        props.modifier_name = ""
        props.state = 'PLACING'

        self.active_parent_index = 0  # new points attach here until changed
        self.dragging = False
        self.drag_mode = None
        self.drag_bone = None
        self.drag_index = None
        self.pose_undo_stack = []  # snapshots for our own in-tool pose undo

        # Blender's native Ctrl+Z reloads scene data from a snapshot, which
        # invalidates any Python object references a running script is still
        # holding (self.obj, self.drag_bone, etc.)   that's exactly the
        # "StructRNA has been removed" crash. Disabling global undo for the
        # session prevents that at the root; our own Ctrl+Z handling below
        # provides pose undo instead, restored when the session ends.
        self._prev_global_undo = context.preferences.edit.use_global_undo
        context.preferences.edit.use_global_undo = False

        self._handle_view = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_callback_view, (), 'WINDOW', 'POST_VIEW')
        self._handle_pixel = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_callback_pixel, (context,), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # Let normal UI (N-panel buttons) handle its own clicks. context.region
        # stays pinned to the region we were invoked in (the 3D viewport) for
        # the whole life of the modal   it does NOT update as the mouse moves
        # over other regions like the N-panel   so checking context.region.type
        # here never actually detects an N-panel click. Instead, check the
        # mouse's real screen position against the viewport's own rectangle.
        #
        # Exception: a drag must always end on mouse release, even if the
        # mouse was released over the N-panel or another editor   otherwise
        # the drag (and Fast Dragging's Object Mode) would get stuck on.
        if (self.dragging and event.type == 'LEFTMOUSE'
                and event.value == 'RELEASE'):
            self.end_drag(context)
            return {'RUNNING_MODAL'}
        if not mouse_in_viewport_window(context, event):
            return {'PASS_THROUGH'}

        props = self.obj.pose_chain_tool

        if props.state == 'IDLE':
            # Cleared or Applied via the N-panel   stop quietly.
            self.finish(context)
            return {'CANCELLED'}

        context.area.tag_redraw()

        if props.state == 'PLACING':
            return self.modal_placing(context, event, props)
        return self.modal_active(context, event, props)

    #   placement sub-state (with branching)                 -

    def modal_placing(self, context, event, props):
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            mouse = Vector((event.mouse_region_x, event.mouse_region_y))
            existing = {i: self.obj.matrix_world @ Vector(p.co) for i, p in enumerate(props.points)}
            clicked_idx = self.point_near_click(context, mouse, existing)

            if clicked_idx is not None:
                self.active_parent_index = clicked_idx
                self.report({'INFO'}, f"Branching from point {clicked_idx + 1}")
                return {'RUNNING_MODAL'}

            hit = raycast_object(context, self.obj, (event.mouse_region_x, event.mouse_region_y))
            if hit is None:
                self.report({'WARNING'}, "Click didn't hit the mesh")
            else:
                p = props.points.add()
                p.co = self.obj.matrix_world.inverted() @ hit
                p.parent_index = self.active_parent_index if len(props.points) > 1 else -1
                new_idx = len(props.points) - 1
                self.active_parent_index = new_idx  # continue this branch by default
                dprint(f"placed point {new_idx} (parent={p.parent_index})")
                if len(props.points) >= props.max_points:
                    self.finalize_placement(context, props)
            return {'RUNNING_MODAL'}

        if event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            if len(props.points) >= 2:
                self.finalize_placement(context, props)
            else:
                self.report({'WARNING'}, "Place at least 2 points (anchor + 1) before finishing")
            return {'RUNNING_MODAL'}

        if event.value == 'PRESS' and (
                event.type == 'BACK_SPACE' or (event.type == 'Z' and event.ctrl)):
            if len(props.points) > 0:
                removed = props.points[len(props.points) - 1]
                self.active_parent_index = removed.parent_index if removed.parent_index >= 0 else 0
                props.points.remove(len(props.points) - 1)
                self.report({'INFO'}, f"Undid last point ({len(props.points)} remaining)")
            else:
                self.report({'INFO'}, "No points to undo")
            return {'RUNNING_MODAL'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            props.points.clear()
            props.state = 'IDLE'
            self.finish(context)
            return {'CANCELLED'}

        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}

        return {'RUNNING_MODAL'}

    def finalize_placement(self, context, props):
        if build_chain(context, self.obj):
            props.state = 'READY'
        else:
            self.report({'ERROR'}, "Failed to build the pose chain rig   see System Console")
            props.points.clear()
            props.state = 'IDLE'

    def point_near_click(self, context, mouse, existing_points_by_index, threshold=8.0):
        region, rv3d = context.region, context.region_data
        best_i, best_d = None, threshold
        for i, p in existing_points_by_index.items():
            p2d = view3d_utils.location_3d_to_region_2d(region, rv3d, p)
            if p2d is None:
                continue
            d = (p2d - mouse).length
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    #   active/posing sub-state                       -

    def modal_active(self, context, event, props):
        mouse = Vector((event.mouse_region_x, event.mouse_region_y))

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            points = get_live_chain_points_tree(props.armature_obj)
            idx = self.pick_point(context, mouse, points)
            if idx is not None:
                self.start_drag(context, idx, points, mouse, props)
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}  # not near a point   let the sculpt brush handle it

        if event.type == 'MOUSEMOVE':
            if self.dragging:
                self.update_drag(context, mouse, props)
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if self.dragging:
                self.end_drag(context)
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        if event.type == 'Z' and (event.ctrl or event.oskey) and event.value == 'PRESS' and not self.dragging:
            self.undo_pose(context, props)
            return {'RUNNING_MODAL'}  # consume it   never let this reach Blender's own undo

        return {'PASS_THROUGH'}

    def undo_pose(self, context, props):
        if not self.pose_undo_stack:
            self.report({'INFO'}, "Nothing left to undo")
            return
        snapshot = self.pose_undo_stack.pop()
        restore_pose_snapshot(props.armature_obj, snapshot)
        context.view_layer.update()
        dprint(f"undo_pose: restored snapshot, {len(self.pose_undo_stack)} more available")
        self.report({'INFO'}, f"Pose change undone ({len(self.pose_undo_stack)} more available)")

    def pick_point(self, context, mouse, points_by_index):
        region, rv3d = context.region, context.region_data
        best_i, best_d = None, 20.0  # px threshold
        for i, p in points_by_index.items():
            p2d = view3d_utils.location_3d_to_region_2d(region, rv3d, p)
            if p2d is None:
                continue
            d = (p2d - mouse).length
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def start_drag(self, context, idx, points, mouse, props):
        self.dragging = True
        self.drag_index = idx
        pbones = props.armature_obj.pose.bones
        region, rv3d = context.region, context.region_data

        MAX_UNDO_STEPS = 50
        self.pose_undo_stack.append(capture_pose_snapshot(props.armature_obj))
        if len(self.pose_undo_stack) > MAX_UNDO_STEPS:
            self.pose_undo_stack.pop(0)

        arm_scale = props.armature_obj.matrix_world.to_scale()
        mesh_scale = self.obj.matrix_world.to_scale()
        dprint(f"start_drag: point index={idx}, armature scale={tuple(round(s, 3) for s in arm_scale)}, "
               f"mesh scale={tuple(round(s, 3) for s in mesh_scale)}"
               + ("  <  NEGATIVE SCALE component present, can invert motion" if
                  any(s < 0 for s in list(arm_scale) + list(mesh_scale)) else ""))

        if idx == 0:
            self.drag_mode = 'TRANSLATE'
            self.drag_bone = pbones.get("Bone_Root")
            self.drag_start_matrix = self.drag_bone.matrix.copy()
            self.drag_start_world_pos = points[0].copy()
            self.drag_start_mouse = mouse.copy()
            dprint("  mode=TRANSLATE bone=Bone_Root")
        else:
            self.drag_mode = 'ROTATE'
            bone_name = f"Bone_{idx}"
            self.drag_bone = pbones.get(bone_name)
            self.drag_start_matrix = self.drag_bone.matrix.copy()
            pivot = props.armature_obj.matrix_world @ self.drag_bone.head
            self.drag_pivot_world = pivot
            pivot2d = view3d_utils.location_3d_to_region_2d(region, rv3d, pivot)
            self.drag_pivot_2d = pivot2d if pivot2d else mouse.copy()
            v = mouse - self.drag_pivot_2d
            self.drag_start_angle = math.atan2(v.y, v.x)
            # Point the axis OUT of the screen (toward the viewer), not into
            # it   otherwise a counterclockwise mouse drag produces a
            # clockwise rotation on screen (right-hand rule around an
            # into-the-screen axis is visually reversed).
            self.drag_view_axis = (rv3d.view_rotation @ Vector((0.0, 0.0, 1.0))).normalized()
            dprint(f"  mode=ROTATE bone={bone_name} pivot={tuple(round(c, 4) for c in pivot)}")

            if POSE_CHAIN_DEBUG:
                vg = self.obj.vertex_groups.get(bone_name)
                if vg is not None:
                    mesh = self.obj.data
                    bound_verts = [v.index for v in mesh.vertices
                                   if any(g.group == vg.index and g.weight > 1e-6 for g in v.groups)]
                    dprint(f"  {len(bound_verts)} vertices bound to {bone_name}")
                    if bound_verts:
                        cached = _geodesic_cache.get(self.obj.name)
                        if cached is not None:
                            import numpy as np
                            n = len(mesh.vertices)
                            islands = get_mesh_islands(n, cached["edges"])
                            bound = np.zeros(n, dtype=bool)
                            bound[bound_verts] = True
                            for i_idx, island in enumerate(islands):
                                overlap = int(bound[island].sum())
                                if overlap:
                                    dprint(f"    -> island {i_idx} ({len(island)} verts total): "
                                           f"{overlap} of them bound here")

        # Last, after all drag state above is captured from Sculpt Mode.
        self.enter_fast_drag_mode(context, props)

    def update_drag(self, context, mouse, props):
        arm_obj = props.armature_obj
        region, rv3d = context.region, context.region_data
        try:
            arm_world = arm_obj.matrix_world
            arm_world_inv = arm_world.inverted()
            start_world = arm_world @ self.drag_start_matrix

            if self.drag_mode == 'TRANSLATE':
                new_pos = view3d_utils.region_2d_to_location_3d(
                    region, rv3d, mouse, self.drag_start_world_pos)
                old_pos = view3d_utils.region_2d_to_location_3d(
                    region, rv3d, self.drag_start_mouse, self.drag_start_world_pos)
                delta = new_pos - old_pos
                new_world = Matrix.Translation(delta) @ start_world
            else:
                v = mouse - self.drag_pivot_2d
                cur_angle = math.atan2(v.y, v.x)
                delta_angle = cur_angle - self.drag_start_angle
                pivot = self.drag_pivot_world
                rot = Matrix.Rotation(delta_angle, 4, self.drag_view_axis)
                new_world = Matrix.Translation(pivot) @ rot @ Matrix.Translation(-pivot) @ start_world

            # No context.view_layer.update() here: setting the matrix already
            # tags the rig for re-evaluation, and Blender evaluates it once
            # before the next viewport redraw. Forcing it on every mouse-move
            # event re-deformed the whole mesh per event (~60 ms each at 700k
            # vertices), even when several events arrive between redraws.
            self.drag_bone.matrix = arm_world_inv @ new_world
        except Exception as e:
            print(f"Pose Chain Tool: drag update failed: {e}")

    def end_drag(self, context):
        self.dragging = False
        if self.drag_bone is not None:
            m = self.drag_bone.matrix
            dprint(f"end_drag: bone={self.drag_bone.name} "
                   f"final head(armature-space)={tuple(round(c, 4) for c in m.translation)}")
        self.leave_fast_drag_mode(context)

    #   Fast Dragging: Object Mode while a point is dragged      -
    #
    # Benchmarked on a 714k-vertex sculpt: after each pose change, Blender
    # re-evaluates the mesh in ~70 ms in Sculpt Mode but ~4 ms in Object
    # Mode (Sculpt Mode keeps extra per-vertex data in sync with every
    # deformation). Switching modes costs ~4 ms into Object Mode and ~70 ms
    # back into Sculpt Mode, paid once per drag instead of on every update.

    def enter_fast_drag_mode(self, context, props):
        self._fast_drag_active = False
        if not props.fast_drag or context.mode != 'SCULPT':
            return
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError as e:
            print(f"Pose Chain Tool: Fast Dragging could not leave Sculpt Mode ({e}); "
                  f"dragging in Sculpt Mode instead.")
            return
        self._fast_drag_active = True
        _object_mode_drag.add(self.obj.name)
        dprint("fast drag: switched to Object Mode")

    def leave_fast_drag_mode(self, context):
        if not getattr(self, '_fast_drag_active', False):
            return
        self._fast_drag_active = False
        _object_mode_drag.discard(self.obj.name)
        try:
            context.view_layer.objects.active = self.obj
            bpy.ops.object.mode_set(mode='SCULPT')
            dprint("fast drag: back in Sculpt Mode")
        except RuntimeError as e:
            print(f"Pose Chain Tool: could not return to Sculpt Mode after dragging ({e})")

    #   lifecycle                               -

    def finish(self, context):
        if getattr(self, 'dragging', False):
            self.end_drag(context)
        else:
            self.leave_fast_drag_mode(context)
        if getattr(self, '_handle_view', None):
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_view, 'WINDOW')
            self._handle_view = None
        if getattr(self, '_handle_pixel', None):
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_pixel, 'WINDOW')
            self._handle_pixel = None
        if hasattr(self, '_prev_global_undo'):
            try:
                context.preferences.edit.use_global_undo = self._prev_global_undo
            except Exception:
                pass
        if context.area:
            context.area.tag_redraw()

    def current_points(self):
        """{point_index: world position}."""
        props = self.obj.pose_chain_tool
        if props.state == 'PLACING':
            return {i: self.obj.matrix_world @ Vector(p.co) for i, p in enumerate(props.points)}
        return get_live_chain_points_tree(props.armature_obj)

    def current_edges(self, points_by_index):
        """[(pos_a, pos_b), ...] for every parent-child pair, using the
        placed points' parent_index (the tree shape doesn't change once
        built, only the positions do)."""
        props = self.obj.pose_chain_tool
        edges = []
        edge_point_indices = []
        for i, p in enumerate(props.points):
            if i == 0:
                continue
            parent_idx = p.parent_index
            if parent_idx in points_by_index and i in points_by_index:
                edges.append((points_by_index[parent_idx], points_by_index[i]))
                edge_point_indices.append((parent_idx, i))
        return edges, edge_point_indices

    def marker_colors(self, points_by_index):
        """{point_index: color}. Orange while still placing. Once ready: blue
        by default, except while dragging, where the dragged point AND every
        point that branches off of it (transitively, possibly across
        multiple branches) turn green   points elsewhere in the tree are
        unaffected and stay blue. Dragging the anchor (index 0) affects the
        whole tree."""
        props = self.obj.pose_chain_tool
        placing_color = (1.0, 0.3, 0.05, 1.0)
        ready_color = (0.15, 0.55, 1.0, 1.0)
        drag_color = (0.2, 1.0, 0.3, 1.0)

        if props.state == 'PLACING':
            return {i: placing_color for i in points_by_index}

        if self.dragging and self.drag_index is not None:
            if self.drag_index == 0:
                affected = set(points_by_index.keys())
            else:
                affected = get_descendants(self.drag_index, get_children_map(props))
            return {i: (drag_color if i in affected else ready_color) for i in points_by_index}

        return {i: ready_color for i in points_by_index}

    def draw_callback_view(self):
        # If Blender invalidated this operator instance (most commonly: the
        # add-on file got hot-reloaded while this modal was still actively
        # running), EVERY attribute access on self raises ReferenceError,
        # including our own plain instance attributes   there's no state we
        # can reach to clean up the handler from inside a dead instance.
        # Suppressing here just stops the console from flooding; the handler
        # itself becomes a harmless permanent no-op until Blender restarts.
        try:
            points = self.current_points()
            colors = self.marker_colors(points)
            edges, edge_point_indices = self.current_edges(points)
            edge_colors = [(colors.get(a, (1, 1, 1, 1)), colors.get(b, (1, 1, 1, 1)))
                           for a, b in edge_point_indices]
            draw_chain_edges_3d(edges, edge_colors)
        except ReferenceError:
            pass

    def draw_callback_pixel(self, context):
        try:
            props = self.obj.pose_chain_tool
            points = self.current_points()
            colors = self.marker_colors(points)
            highlight = self.active_parent_index if props.state == 'PLACING' else None
            draw_chain_markers_2d(context, points, colors, show_labels=props.show_labels,
                                   highlight_index=highlight)
        except ReferenceError:
            pass


class PCT_OT_cancel_placement(Operator):
    bl_idname = "pct.cancel_placement"
    bl_label = "Cancel Placement"
    bl_description = "Cancel point placement without building a chain"

    def execute(self, context):
        obj, props = get_props(context)
        if props:
            props.points.clear()
            props.state = 'IDLE'
        return {'FINISHED'}


#                                      -
# Clear / Apply
#                                      -

class PCT_OT_clear_points(Operator):
    bl_idname = "pct.clear_points"
    bl_label = "Clear Points"
    bl_description = "Remove chain points and the temporary rig WITHOUT keeping any deformation"

    @classmethod
    def poll(cls, context):
        obj, props = get_props(context)
        return props is not None and props.state != 'IDLE'

    def execute(self, context):
        obj, props = get_props(context)
        cleanup_chain(context, obj)
        self.report({'INFO'}, "Points cleared, mesh restored")
        return {'FINISHED'}


class PCT_OT_apply_deform(Operator):
    bl_idname = "pct.apply_deform"
    bl_label = "Apply Deformations"
    bl_description = "Bake the current pose permanently into the mesh and remove the temporary rig"

    @classmethod
    def poll(cls, context):
        obj, props = get_props(context)
        return (props is not None and props.state == 'READY'
                and props.armature_obj is not None)

    def execute(self, context):
        obj, props = get_props(context)
        mod_name = props.modifier_name

        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError as e:
            self.report({'ERROR'}, f"Could not switch to Object Mode to apply: {e}")
            return {'CANCELLED'}

        context.view_layer.objects.active = obj
        obj.select_set(True)

        applied = False
        try:
            if mod_name and mod_name in obj.modifiers:
                bpy.ops.object.modifier_apply(modifier=mod_name)
                applied = True
            else:
                self.report({'WARNING'}, "No armature modifier found to apply")
        except RuntimeError as e:
            self.report({'ERROR'}, f"Could not apply deformation: {e}")
        finally:
            cleanup_chain(context, obj)
            try:
                context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.object.mode_set(mode='SCULPT')
            except RuntimeError as e:
                self.report({'ERROR'}, f"Applied, but could not return to Sculpt Mode: {e}")
                return {'CANCELLED'}

        if applied:
            self.report({'INFO'}, "Pose applied to mesh")
        return {'FINISHED'}


class PCT_OT_preview_weights(Operator):
    bl_idname = "pct.preview_weights"
    bl_label = "Preview Bone Influence"
    bl_description = ("Switch to Weight Paint mode showing exactly which vertices "
                       "the selected bone currently influences and how strongly   "
                       "lets you check Weight Falloff/Falloff Softness directly on "
                       "the mesh instead of guessing")

    @classmethod
    def poll(cls, context):
        obj, props = get_props(context)
        return props is not None and props.state == 'READY' and props.preview_bone != "NONE"

    def execute(self, context):
        obj, props = get_props(context)
        bone_name = props.preview_bone
        vg = obj.vertex_groups.get(bone_name)
        if vg is None:
            self.report({'WARNING'}, f"No vertex group found for {bone_name}   "
                        f"try rebuilding the chain")
            return {'CANCELLED'}

        obj.vertex_groups.active_index = vg.index
        try:
            bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        except RuntimeError as e:
            self.report({'ERROR'}, f"Could not switch to Weight Paint mode: {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Showing {bone_name}'s influence   switch back to "
                    f"Sculpt Mode (mode dropdown, top-left of the viewport) when done.")
        return {'FINISHED'}


#                                      -
# N-panel
#                                      -

class PCT_PT_panel(Panel):
    bl_label = "Pose Chain Tool"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Pose Chain"

    @classmethod
    def poll(cls, context):
        # Sculpt Mode only (this used to be bl_context = "sculpt_mode"),
        # plus Object Mode while Fast Dragging, so the panel doesn't vanish
        # and reappear on every drag.
        if context.mode == 'SCULPT':
            return True
        obj = context.active_object
        return obj is not None and obj.name in _object_mode_drag

    def draw(self, context):
        layout = self.layout
        obj, props = get_props(context)
        if props is None:
            layout.label(text="Select a mesh object", icon='INFO')
            return

        # Keep the module-level debug flag in sync with the property on
        # every draw   the property's own update= callback only fires on
        # an actual checkbox click, so a .blend saved with debug_logging
        # already True would otherwise show a checked box that doesn't
        # actually produce output until toggled off and back on.
        global POSE_CHAIN_DEBUG
        POSE_CHAIN_DEBUG = props.debug_logging

        #   collapsible how-to-use section, always available         -
        help_box = layout.box()
        row = help_box.row()
        row.prop(props, "show_help",
                 icon='TRIA_DOWN' if props.show_help else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="How This Works")
        if props.show_help:
            col = help_box.column(align=True)
            col.label(text="1. Set Max Points, click Activate.")
            col.label(text="2. Click mesh to place points.")
            col.label(text="   Click a point to branch off it.")
            col.label(text="   Backspace/Ctrl+Z: undo point")
            col.label(text="   Enter: finish  Esc: cancel")
            col.separator()
            col.label(text="   Blue points = chain is ready.")
            col.separator()
            col.label(text="3. Drag anchor to move tree.")
            col.label(text="   Drag a point to rotate its branch.")
            col.label(text="   Click empty space to sculpt.")
            col.label(text="   Ctrl+Z undoes a pose change.")
            col.separator()
            col.label(text="4. Clear Points resets. Apply bakes it.")

        layout.prop(props, "show_labels")
        layout.prop(props, "fast_drag")
        layout.prop(props, "debug_logging")
        layout.separator()

        if props.state == 'IDLE':
            layout.prop(props, "max_points")
            layout.prop(props, "weight_falloff")
            layout.prop(props, "falloff_softness")
            layout.prop(props, "joint_blend")
            layout.prop(props, "auto_thickness")
            layout.prop(props, "preserve_volume")
            layout.operator("pct.activate", icon='PLAY', text="Activate")

        elif props.state == 'PLACING':
            layout.label(text=f"Placing point {len(props.points) + 1} / {props.max_points}",
                         icon='RADIOBUT_ON')
            layout.label(text="Click an existing point to branch from it",
                         icon='UV_SYNC_SELECT')
            box = layout.box()
            box.label(text="Backspace or Ctrl+Z:", icon='LOOP_BACK')
            box.label(text="undo the last point")
            layout.operator("pct.cancel_placement", icon='X', text="Cancel")

        else:  # READY   chain built, points are live and clickable
            box = layout.box()
            box.label(text=f"Chain active: {len(props.points)} points", icon='CHECKMARK')
            box.label(text="Ctrl+Z undoes the last pose change", icon='LOOP_BACK')

            layout.separator()
            col = layout.column(align=True)
            col.label(text="Live-adjust weighting (updates the mesh directly):")
            col.prop(props, "weight_falloff")
            col.prop(props, "falloff_softness")
            col.prop(props, "joint_blend")
            col.prop(props, "auto_thickness")
            col.prop(props, "preserve_volume")

            layout.separator()
            col = layout.column(align=True)
            col.label(text="Preview a bone's weight influence:")
            col.prop(props, "preview_bone", text="")
            col.operator("pct.preview_weights", icon='GROUP_VERTEX', text="Preview Bone Influence")

            layout.separator()
            row = layout.row(align=True)
            row.operator("pct.clear_points", icon='TRASH', text="Clear Points")
            row.operator("pct.apply_deform", icon='CHECKMARK', text="Apply")


#                                      -
# Registration
#                                      -

classes = (
    PCT_Point,
    PCT_GroupNameEntry,
    PCT_Properties,
    PCT_OT_activate,
    PCT_OT_cancel_placement,
    PCT_OT_clear_points,
    PCT_OT_apply_deform,
    PCT_OT_preview_weights,
    PCT_PT_panel,
)


def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            try:
                bpy.utils.unregister_class(cls)
            except Exception:
                pass
            bpy.utils.register_class(cls)

    bpy.types.Object.pose_chain_tool = PointerProperty(type=PCT_Properties)


def unregister():
    _geodesic_cache.clear()
    _refresh_pending.clear()
    _object_mode_drag.clear()
    if hasattr(bpy.types.Object, "pose_chain_tool"):
        del bpy.types.Object.pose_chain_tool

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
