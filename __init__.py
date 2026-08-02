# Add-on metadata lives in blender_manifest.toml next to this file.
#
# ARCHITECTURE NOTE (read this before touching the deformation code):
# This tool places points in Sculpt Mode and treats them as an FK bone TREE
# (not just a straight chain -- points can branch, e.g. a palm point with
# several independent finger chains coming off it). Rather than hand-rolling
# vertex skinning, it builds a real (hidden) Armature object with Automatic
# Weights under the hood -- the same engine Blender uses for rigging -- and
# then drives that armature's pose bones directly via Python (PoseBone.matrix)
# from a custom modal operator running in Sculpt Mode. The user never sees
# Pose Mode; dragging a point in the sculpt viewport is what moves the
# underlying bone.
#
# SESSION MODEL: a single continuous modal operator (PCT_OT_activate) spans
# the entire lifetime from "Activate" through to Clear/Apply. It never stops
# and restarts in between. Its internal sub-state (props.state) moves
# PLACING -> READY, and while READY, clicking near a point starts a drag
# while clicks elsewhere pass through to the normal sculpt brush so sculpting
# still works normally around it. The only way out is Clear Points / Apply
# Deformations in the N-panel, which set state back to IDLE; the running
# modal notices and exits on its next event.
#
# TREE / BRANCHING MODEL:
#   Each point (except point 0, the anchor) stores parent_index -- the index
#   of the point it branches off of. By default that's "the previous point",
#   so a straight chain is just the simplest possible tree. To branch,
#   clicking an EXISTING point during placement selects it as the new
#   "active parent" -- subsequent new points attach there instead, letting
#   you build e.g. one palm point with five independent finger chains.
#
#   Bone_Root is a small synthetic bone created at point 0 (the anchor) that
#   every point with parent_index == 0 attaches to. This guarantees dragging
#   the anchor moves the WHOLE tree uniformly, no matter how many branches
#   start directly at it -- without it, each anchor-attached branch would be
#   an independent root bone and we'd have to move them all separately.
#   For every other point i (i >= 1): Bone_i goes from its parent point's
#   current position to point i's position, parented under whichever bone
#   ends at the parent point (Bone_Root if parent_index == 0, else
#   Bone_{parent_index}).
#
# Dragging point 0 (anchor) translates Bone_Root -> the whole tree follows.
# Dragging point i (i >= 1) rotates Bone_i around its head (the parent
# point's position) -- and everything downstream (i's descendants in the
# tree, which may span multiple branches) cascades automatically since those
# bones are parented under it, directly or indirectly.

'''Copyright (C) <2026> <the_grass_trainer>
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.'''


import bpy
import blf
import gpu
import math
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix
from bpy_extras import view3d_utils
from bpy.props import (
    FloatVectorProperty, CollectionProperty, IntProperty,
    EnumProperty, PointerProperty, StringProperty, BoolProperty,
)
from bpy.types import PropertyGroup, Operator, Panel

# Set to True to print island detection, bone-binding, and branching
# decisions to Blender's System Console (Window > Toggle System Console on
# Windows; on Mac/Linux, launch Blender from a terminal to see stdout there).
POSE_CHAIN_DEBUG = True


def dprint(*args):
    if POSE_CHAIN_DEBUG:
        print("[PoseChainTool]", *args)


# ---------------------------------------------------------------------------
# Property groups
# ---------------------------------------------------------------------------

class PCT_Point(PropertyGroup):
    co: FloatVectorProperty(size=3, subtype='XYZ')
    # Index of the point this one branches off of. Unused/ignored for point 0
    # (the anchor). Defaults to -1 so an uninitialized value is obviously
    # wrong rather than silently pointing at point 0.
    parent_index: IntProperty(default=-1)


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
    max_points: IntProperty(name="Max Points", default=3, min=2, max=100, soft_max=20,
                             description="Total points across the whole tree (including "
                                         "branches). Increase for longer/more branching rigs.")
    armature_obj: PointerProperty(type=bpy.types.Object)
    modifier_name: StringProperty(default="")
    show_labels: BoolProperty(
        name="Show Point Numbers", default=True,
        description="Show the 1/2/3... order label on each point marker")
    show_help: BoolProperty(
        name="Show Help", default=True,
        description="Show the how-to-use instructions in this panel")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def mouse_in_viewport_window(context, event):
    """True if the mouse is currently over the 3D viewport's own drawing
    area (as opposed to the N-panel, header, or another editor).

    context.region stays pinned to the region the operator was invoked in
    (the viewport's WINDOW region) for the entire life of a modal operator --
    it never updates to reflect wherever the mouse currently is, which is
    exactly why checking context.region.type doesn't work here. What DOES
    change per-event is event.mouse_region_x/y, which Blender always reports
    relative to that same fixed region -- so once the mouse leaves that
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

    Raycasts against the EVALUATED mesh (what's actually displayed) first --
    the original (base) mesh data can momentarily lag behind what's on
    screen during/just after a sculpt stroke -- and falls back to the
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


def point_segment_distance(p, a, b):
    """Distance from point p to the line segment a-b (all Vectors, same space)."""
    ab = b - a
    ab_len_sq = ab.length_squared
    if ab_len_sq <= 1e-12:
        return (p - a).length
    t = max(0.0, min(1.0, (p - a).dot(ab) / ab_len_sq))
    return (p - (a + ab * t)).length


def get_mesh_islands(mesh):
    """Connected components of the mesh by vertex/edge adjacency -- each
    disconnected piece (e.g. a separate object before it was Ctrl+J joined
    into this one) comes back as its own list of vertex indices."""
    n = len(mesh.vertices)
    adjacency = [[] for _ in range(n)]
    for e in mesh.edges:
        v0, v1 = e.vertices[0], e.vertices[1]
        adjacency[v0].append(v1)
        adjacency[v1].append(v0)

    visited = [False] * n
    islands = []
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        island = []
        while stack:
            vi = stack.pop()
            island.append(vi)
            for nb in adjacency[vi]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        islands.append(island)
    return islands


def fill_unweighted_vertices(mesh_obj, arm_obj, bone_names):
    """Guarantee every disconnected mesh island is influenced by some bone.

    Automatic (heat-map) Weights propagates weight through CONNECTED mesh
    topology from each bone -- if the mesh is actually several separate
    pieces joined into one object (Ctrl+J), heat can't cross a seam that
    isn't really welded, so any island it can't reach silently gets no
    weight at all and never moves. This doesn't raise an error (so the
    envelope-weights fallback never triggers), it just quietly under-covers
    the mesh.

    Works per ISLAND rather than per vertex: deciding vertex-by-vertex can
    split a single disconnected piece across two different bones
    inconsistently -- deciding once per island, using its overall centroid,
    keeps each disconnected piece moving as one consistent rigid unit.
    Bone_Root is intentionally excluded from the candidates (it's just a
    structural pivot at the anchor, not meant to hold mesh weight). Returns
    the number of vertices it had to fix.
    """
    mesh = mesh_obj.data
    vgs = mesh_obj.vertex_groups
    bone_vgs = [vgs.get(name) for name in bone_names]
    group_indices = {vg.index for vg in bone_vgs if vg}
    if not group_indices:
        return 0

    segments = []
    for name in bone_names:
        pbone = arm_obj.pose.bones.get(name)
        if pbone is None:
            segments.append(None)
            continue
        segments.append((arm_obj.matrix_world @ pbone.head, arm_obj.matrix_world @ pbone.tail))

    mesh_mat = mesh_obj.matrix_world

    def vertex_weight(vi):
        return sum(g.weight for g in mesh.vertices[vi].groups if g.group in group_indices)

    islands = get_mesh_islands(mesh)
    dprint(f"build_chain: mesh has {len(islands)} disconnected island(s), "
           f"{len(bone_names)} bone(s): {bone_names}")
    for i, seg in enumerate(segments):
        if seg is None:
            dprint(f"  bone {bone_names[i]}: MISSING (no pose bone found)")
        else:
            dprint(f"  bone {bone_names[i]}: head={tuple(round(c, 4) for c in seg[0])} "
                   f"tail={tuple(round(c, 4) for c in seg[1])}")

    fixed = 0
    for island_idx, island in enumerate(islands):
        trusted = any(vertex_weight(vi) > 1e-6 for vi in island)

        centroid = Vector((0.0, 0.0, 0.0))
        for vi in island:
            centroid += mesh_mat @ mesh.vertices[vi].co
        centroid /= len(island)

        if trusted:
            dprint(f"  island {island_idx}: {len(island)} verts, "
                   f"centroid={tuple(round(c, 4) for c in centroid)} -- "
                   f"TRUSTED (Automatic Weights already reached it), left as-is")
            continue

        dists = []
        for i, seg in enumerate(segments):
            if seg is None:
                dists.append(None)
                continue
            dists.append(point_segment_distance(centroid, seg[0], seg[1]))
        dist_report = ", ".join(
            f"{bone_names[i]}={d:.4f}" if d is not None else f"{bone_names[i]}=n/a"
            for i, d in enumerate(dists))

        best_idx, best_dist = None, None
        for i, d in enumerate(dists):
            if d is None:
                continue
            if best_dist is None or d < best_dist:
                best_dist, best_idx = d, i

        chosen = bone_names[best_idx] if best_idx is not None else "NONE"
        if best_idx is not None:
            dprint(f"  island {island_idx}: {len(island)} verts, "
                   f"centroid={tuple(round(c, 4) for c in centroid)} -- UNREACHED, "
                   f"distances=[{dist_report}] -> REBOUND to {chosen} (dist={best_dist:.4f})")
        else:
            dprint(f"  island {island_idx}: {len(island)} verts -- UNREACHED, no valid bone found")

        if best_idx is not None and bone_vgs[best_idx] is not None:
            bone_vgs[best_idx].add(island, 1.0, 'REPLACE')
            fixed += len(island)

    dprint(f"build_chain: fill_unweighted_vertices fixed {fixed} vertex(es) total")
    return fixed


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

        # Synthetic root bone at the anchor -- every point attached directly
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

        try:
            bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        except RuntimeError as e:
            # Automatic (heat-map) weighting can fail on tricky topology --
            # fall back to a simpler weighting scheme rather than aborting.
            print(f"Pose Chain Tool: Automatic Weights failed ({e}), "
                  f"falling back to envelope weights.")
            bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')

        bone_names = [f"Bone_{i}" for i in range(1, len(points_world))]
        fixed = fill_unweighted_vertices(mesh_obj, arm_obj, bone_names)
        if fixed:
            print(f"Pose Chain Tool: {fixed} vertex(es) had no weight from Automatic "
                  f"Weights (likely a disconnected mesh island from a Join) -- "
                  f"assigned them to their nearest bone directly.")

        arm_obj.hide_set(True)
        arm_obj.hide_select = True

        mod = next((m for m in mesh_obj.modifiers
                    if m.type == 'ARMATURE' and m.object == arm_obj), None)
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
    """Remove the temp rig and reset tool state. Does not touch mesh geometry --
    call this AFTER modifier_apply if you want to keep the deformation, or
    directly if you want to discard it (the modifier just loses its target and
    stops deforming once the armature is gone)."""
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
    props.points.clear()
    props.armature_obj = None
    props.modifier_name = ""
    props.state = 'IDLE'


# ---------------------------------------------------------------------------
# The single continuous session operator: place points (with branching),
# then pose them, all in one modal that never lets the on-screen points
# disappear.
# ---------------------------------------------------------------------------

class PCT_OT_activate(Operator):
    bl_idname = "pct.activate"
    bl_label = "Activate Pose Chain"
    bl_description = ("Click the mesh to place chain points (anchor first). "
                       "Click an EXISTING point to branch a new chain off of "
                       "it. Once built, click any point anytime to pose it -- "
                       "clicking elsewhere still sculpts normally. "
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
        # the whole life of the modal -- it does NOT update as the mouse moves
        # over other regions like the N-panel -- so checking context.region.type
        # here never actually detects an N-panel click. Instead, check the
        # mouse's real screen position against the viewport's own rectangle.
        if not mouse_in_viewport_window(context, event):
            return {'PASS_THROUGH'}

        props = self.obj.pose_chain_tool

        if props.state == 'IDLE':
            # Cleared or Applied via the N-panel -- stop quietly.
            self.finish(context)
            return {'CANCELLED'}

        context.area.tag_redraw()

        if props.state == 'PLACING':
            return self.modal_placing(context, event, props)
        return self.modal_active(context, event, props)

    # -- placement sub-state (with branching) ---------------------------------

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
            self.report({'ERROR'}, "Failed to build the pose chain rig -- see System Console")
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

    # -- active/posing sub-state ---------------------------------------------

    def modal_active(self, context, event, props):
        mouse = Vector((event.mouse_region_x, event.mouse_region_y))

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            points = get_live_chain_points_tree(props.armature_obj)
            idx = self.pick_point(context, mouse, points)
            if idx is not None:
                self.start_drag(context, idx, points, mouse, props)
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}  # not near a point -- let the sculpt brush handle it

        if event.type == 'MOUSEMOVE':
            if self.dragging:
                self.update_drag(context, mouse, props)
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if self.dragging:
                self.dragging = False
                if self.drag_bone is not None:
                    m = self.drag_bone.matrix
                    dprint(f"end_drag: bone={self.drag_bone.name} "
                           f"final head(armature-space)={tuple(round(c, 4) for c in m.translation)}")
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}

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

        arm_scale = props.armature_obj.matrix_world.to_scale()
        mesh_scale = self.obj.matrix_world.to_scale()
        dprint(f"start_drag: point index={idx}, armature scale={tuple(round(s, 3) for s in arm_scale)}, "
               f"mesh scale={tuple(round(s, 3) for s in mesh_scale)}"
               + ("  <-- NEGATIVE SCALE component present, can invert motion" if
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
            # it -- otherwise a counterclockwise mouse drag produces a
            # clockwise rotation on screen (right-hand rule around an
            # into-the-screen axis is visually reversed).
            self.drag_view_axis = (rv3d.view_rotation @ Vector((0.0, 0.0, 1.0))).normalized()
            dprint(f"  mode=ROTATE bone={bone_name} pivot={tuple(round(c, 4) for c in pivot)}")

            vg = self.obj.vertex_groups.get(bone_name)
            if vg is not None:
                mesh = self.obj.data
                bound_verts = [v.index for v in mesh.vertices
                               if any(g.group == vg.index and g.weight > 1e-6 for g in v.groups)]
                dprint(f"  {len(bound_verts)} vertices bound to {bone_name}")
                if bound_verts:
                    islands = get_mesh_islands(mesh)
                    bound_set = set(bound_verts)
                    for i_idx, island in enumerate(islands):
                        overlap = bound_set.intersection(island)
                        if overlap:
                            dprint(f"    -> island {i_idx} ({len(island)} verts total): "
                                   f"{len(overlap)} of them bound here")

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

            self.drag_bone.matrix = arm_world_inv @ new_world
            context.view_layer.update()
        except Exception as e:
            print(f"Pose Chain Tool: drag update failed: {e}")

    # -- lifecycle -------------------------------------------------------------

    def finish(self, context):
        if getattr(self, '_handle_view', None):
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_view, 'WINDOW')
            self._handle_view = None
        if getattr(self, '_handle_pixel', None):
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_pixel, 'WINDOW')
            self._handle_pixel = None
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
        multiple branches) turn green -- points elsewhere in the tree are
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
        points = self.current_points()
        colors = self.marker_colors(points)
        edges, edge_point_indices = self.current_edges(points)
        edge_colors = [(colors.get(a, (1, 1, 1, 1)), colors.get(b, (1, 1, 1, 1)))
                       for a, b in edge_point_indices]
        draw_chain_edges_3d(edges, edge_colors)

    def draw_callback_pixel(self, context):
        props = self.obj.pose_chain_tool
        points = self.current_points()
        colors = self.marker_colors(points)
        highlight = self.active_parent_index if props.state == 'PLACING' else None
        draw_chain_markers_2d(context, points, colors, show_labels=props.show_labels,
                               highlight_index=highlight)


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


# ---------------------------------------------------------------------------
# Clear / Apply
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# N-panel
# ---------------------------------------------------------------------------

class PCT_PT_panel(Panel):
    bl_label = "Pose Chain Tool"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Pose Chain"
    bl_context = "sculpt_mode"

    def draw(self, context):
        layout = self.layout
        obj, props = get_props(context)
        if props is None:
            layout.label(text="Select a mesh object", icon='INFO')
            return

        # -- collapsible how-to-use section, always available -----------------
        help_box = layout.box()
        row = help_box.row()
        row.prop(props, "show_help",
                 icon='TRIA_DOWN' if props.show_help else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="How This Works")
        if props.show_help:
            col = help_box.column(align=True)
            col.label(text="1. Set Max Points, click Activate.")
            col.label(text="2. Click the mesh to place points,")
            col.label(text="   anchor first.")
            col.label(text="   Click an EXISTING point to")
            col.label(text="   branch a new chain off it")
            col.label(text="   (e.g. one finger per branch).")
            col.label(text="   Backspace / Ctrl+Z: undo a point")
            col.label(text="   Enter: finish early (2+ points)")
            col.label(text="   Esc / Right-click: cancel")
            col.separator()
            col.label(text="   Points turn blue once the chain")
            col.label(text="   is built and ready to pose.")
            col.separator()
            col.label(text="3. Drag the anchor to move the")
            col.label(text="   whole tree.")
            col.label(text="   Drag any other point to rotate")
            col.label(text="   it (and its own branch) around")
            col.label(text="   the point it branches from.")
            col.label(text="   Click empty space to sculpt")
            col.label(text="   normally.")
            col.separator()
            col.label(text="4. Clear Points discards it all.")
            col.label(text="   Apply bakes the pose in.")

        layout.prop(props, "show_labels")
        layout.separator()

        if props.state == 'IDLE':
            layout.prop(props, "max_points")
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

        else:  # READY -- chain built, points are live and clickable
            box = layout.box()
            box.label(text=f"Chain active — {len(props.points)} points", icon='CHECKMARK')

            layout.separator()
            row = layout.row(align=True)
            row.operator("pct.clear_points", icon='TRASH', text="Clear Points")
            row.operator("pct.apply_deform", icon='CHECKMARK', text="Apply")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    PCT_Point,
    PCT_Properties,
    PCT_OT_activate,
    PCT_OT_cancel_placement,
    PCT_OT_clear_points,
    PCT_OT_apply_deform,
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
    if hasattr(bpy.types.Object, "pose_chain_tool"):
        del bpy.types.Object.pose_chain_tool

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
