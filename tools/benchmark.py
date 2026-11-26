"""Headless performance benchmark for the Sculpt Pose Chain Tool.

Run with Blender in background mode, from the repo root:

    blender --background --factory-startup --python tools/benchmark.py -- 25000 100000 400000

Each number is an approximate target vertex count. For each size it builds an
elongated "limb" mesh, places a 5-point chain with one branch, then times:

  * build_chain() total, broken down into its main steps
    (adjacency, apply_uniform_falloff_weights, clean_up_weight_islands)
  * a live slider refresh (re-weighting with cached data)
  * one rig re-evaluation after moving a bone (Blender's Armature modifier
    re-deforming the mesh; happens once per viewport redraw while dragging)

Optional flags:
  --profile       print a cProfile summary of build_chain for the largest size
  --auto-weights  also time Blender's Automatic Weights alone (what <= 1.5.2
                  ran inside build_chain; very slow on big meshes)

This file is not part of the extension zip (excluded in blender_manifest.toml).
"""

import bpy
import sys
import os
import math
import time
import importlib.util
import cProfile
import pstats
import io
from mathutils import Vector, Matrix

ADDON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_addon(path=None):
    """Load the add-on from this repo, or from `path` (--addon=<file>) to
    benchmark another version, e.g. one extracted with git show."""
    spec = importlib.util.spec_from_file_location(
        "sculpt_pose_chain_tool", path or os.path.join(ADDON_DIR, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sculpt_pose_chain_tool"] = mod
    spec.loader.exec_module(mod)
    mod.register()
    return mod


def reset_scene():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for data in (bpy.data.meshes, bpy.data.armatures):
        for item in list(data):
            data.remove(item)


def make_limb(target_verts):
    """UV sphere stretched along Z: roughly limb-shaped, one connected island.
    UV sphere vertex count = segments * (rings - 1) + 2."""
    side = max(8, int(math.sqrt(target_verts)))
    bpy.ops.mesh.primitive_uv_sphere_add(segments=side, ring_count=side, radius=1.0)
    obj = bpy.context.active_object
    obj.scale = (0.6, 0.6, 3.0)
    bpy.ops.object.transform_apply(scale=True)
    return obj


def place_points(obj):
    """Anchor at the bottom, a 3-segment main chain up the middle, plus one
    branch off point 2 (index 1). Coordinates are object-local."""
    props = obj.pose_chain_tool
    props.points.clear()
    layout = [
        ((0.0, 0.0, -2.6), -1),
        ((0.0, 0.0, -1.0), 0),
        ((0.0, 0.0, 0.6), 1),
        ((0.0, 0.0, 2.2), 2),
        ((0.35, 0.0, 0.0), 1),   # branch off point index 1
    ]
    for co, parent in layout:
        p = props.points.add()
        p.co = co
        p.parent_index = parent
    props.max_points = len(layout)


class Timers:
    """Wraps module-level functions so build_chain's internal calls are timed."""

    def __init__(self):
        self.totals = {}

    def wrap(self, mod, name):
        original = getattr(mod, name, None)
        if original is None:
            return None

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - t0

        setattr(mod, name, timed)
        return original


def time_parent_set(obj, arm_obj):
    """Automatic Weights in isolation, on a throwaway copy of the same rig."""
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh_copy = obj.copy()
    mesh_copy.data = obj.data.copy()
    for m in list(mesh_copy.modifiers):
        mesh_copy.modifiers.remove(m)
    mesh_copy.vertex_groups.clear()
    mesh_copy.parent = None
    bpy.context.collection.objects.link(mesh_copy)
    arm_copy = arm_obj.copy()
    arm_copy.data = arm_obj.data.copy()
    bpy.context.collection.objects.link(arm_copy)
    arm_copy.hide_set(False)
    arm_copy.hide_select = False

    bpy.ops.object.select_all(action='DESELECT')
    mesh_copy.select_set(True)
    arm_copy.select_set(True)
    bpy.context.view_layer.objects.active = arm_copy
    t0 = time.perf_counter()
    try:
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        ok = True
    except RuntimeError:
        ok = False
    elapsed = time.perf_counter() - t0

    bpy.data.objects.remove(mesh_copy, do_unlink=True)
    bpy.data.objects.remove(arm_copy, do_unlink=True)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='SCULPT')
    return elapsed, ok


def bench_size(mod, target, profile=False, auto_weights=False):
    reset_scene()
    obj = make_limb(target)
    n_verts = len(obj.data.vertices)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='SCULPT')
    place_points(obj)
    props = obj.pose_chain_tool

    timers = Timers()
    originals = {}
    for name in ("build_mesh_adjacency", "fill_unweighted_vertices",
                 "apply_uniform_falloff_weights", "clean_up_weight_islands",
                 "get_mesh_islands"):
        originals[name] = timers.wrap(mod, name)

    profiler = cProfile.Profile() if profile else None
    t0 = time.perf_counter()
    if profiler:
        profiler.enable()
    ok = mod.build_chain(bpy.context, obj)
    if profiler:
        profiler.disable()
    build_total = time.perf_counter() - t0
    build_steps = dict(timers.totals)

    for name, fn in originals.items():
        if fn is not None:
            setattr(mod, name, fn)

    result = {"target": target, "verts": n_verts, "build_ok": ok,
              "build_total": build_total, "build_steps": build_steps}
    if not ok:
        return result, None
    props.state = 'READY'

    # Live slider refresh: same work _schedule_corridor_refresh's timer does.
    positions = mod._geodesic_cache[obj.name]
    bone_names = [f"Bone_{i}" for i in range(1, len(props.points))]
    parent_indices = [p.parent_index for p in props.points]
    t0 = time.perf_counter()
    _p, valid_by_bone, weight_by_bone = mod.apply_uniform_falloff_weights(
        obj, props.armature_obj, bone_names, parent_indices, positions,
        0.5, props.falloff_softness, props.joint_blend)
    t_apply = time.perf_counter() - t0
    t0 = time.perf_counter()
    mod.clean_up_weight_islands(obj, bone_names, positions, valid_by_bone,
                                weight_by_bone, props.falloff_softness,
                                mod.ISLAND_CLEANUP_THRESHOLD)
    t_clean = time.perf_counter() - t0
    t0 = time.perf_counter()
    obj.data.update()
    bpy.context.view_layer.update()
    t_eval = time.perf_counter() - t0
    result["refresh"] = {"apply_weights": t_apply, "island_cleanup": t_clean,
                         "mesh_update_eval": t_eval}

    # Pose updates: replicate update_drag (set bone matrix + view_layer.update()).
    arm = props.armature_obj
    pbone = arm.pose.bones["Bone_2"]
    start = pbone.matrix.copy()
    pivot = arm.matrix_world @ pbone.head
    axis = Vector((1.0, 0.0, 0.0))
    steps = 30
    times = []
    for i in range(1, steps + 1):
        rot = Matrix.Rotation(math.radians(i), 4, axis)
        new_world = Matrix.Translation(pivot) @ rot @ Matrix.Translation(-pivot) @ (arm.matrix_world @ start)
        t0 = time.perf_counter()
        pbone.matrix = arm.matrix_world.inverted() @ new_world
        bpy.context.view_layer.update()
        times.append(time.perf_counter() - t0)
    times.sort()
    result["pose_update"] = {"median": times[len(times) // 2], "max": times[-1]}

    # Automatic Weights alone, for comparison with versions <= 1.5.2 that ran
    # it inside build_chain. Slow on big meshes, so only with --auto-weights.
    if auto_weights:
        t_auto, auto_ok = time_parent_set(obj, arm)
        result["auto_weights"] = t_auto
        result["auto_weights_ok"] = auto_ok

    mod.cleanup_chain(bpy.context, obj)
    return result, profiler


def fmt(s):
    return f"{s * 1000:8.1f} ms" if s < 10 else f"{s:8.2f} s "


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    profile = "--profile" in argv
    auto_weights = "--auto-weights" in argv
    sizes = [int(a) for a in argv if a.isdigit()] or [25000, 100000, 400000]

    print(f"Blender {bpy.app.version_string}, sizes={sizes}")
    addon = next((a.split("=", 1)[1] for a in argv if a.startswith("--addon=")), None)
    if addon:
        print(f"add-on file: {addon}")
    mod = load_addon(addon)
    results = []
    last_profiler = None
    for i, target in enumerate(sizes):
        print(f"\n=== target ~{target:,} verts ===", flush=True)
        do_profile = profile and i == len(sizes) - 1
        res, prof = bench_size(mod, target, profile=do_profile, auto_weights=auto_weights)
        results.append(res)
        if prof:
            last_profiler = prof
        print(f"verts: {res['verts']:,}   build ok: {res['build_ok']}")
        print(f"  build_chain total            {fmt(res['build_total'])}")
        if "auto_weights" in res:
            print(f"    Automatic Weights (alone)  {fmt(res['auto_weights'])}  ok={res['auto_weights_ok']}")
        for k, v in res["build_steps"].items():
            print(f"    {k:<28} {fmt(v)}")
        if "refresh" in res:
            r = res["refresh"]
            print(f"  live slider refresh")
            print(f"    apply_uniform_falloff      {fmt(r['apply_weights'])}")
            print(f"    clean_up_weight_islands    {fmt(r['island_cleanup'])}")
            print(f"    mesh update + eval         {fmt(r['mesh_update_eval'])}")
        if "pose_update" in res:
            p = res["pose_update"]
            print(f"  rig re-evaluation (per redraw while dragging)  median {fmt(p['median'])}  max {fmt(p['max'])}")

    if last_profiler:
        s = io.StringIO()
        pstats.Stats(last_profiler, stream=s).sort_stats("cumulative").print_stats(30)
        print("\n=== cProfile: build_chain (largest size), top 30 by cumulative ===")
        print(s.getvalue())


if __name__ == "__main__":
    main()
