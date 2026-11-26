"""Check that the current __init__.py produces the same vertex weights as a
reference version of the add-on (e.g. the last release).

    git show <tag-or-commit>:__init__.py > reference.py
    blender --background --factory-startup --python tools/verify_weights.py -- reference.py

Builds a chain on several test meshes (plain limb, noisy limb, joined
pieces, branching "hand") with different slider settings, then runs a live
slider refresh, and compares every vertex group weight from both versions.
Exit code is 1 if any weight differs by more than the tolerance.

This file is not part of the extension zip (excluded in blender_manifest.toml).
"""
import bpy
import sys
import os
import random
import importlib.util
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-4  # vertex group weights are stored as 32-bit floats


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reset_scene():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.armatures):
        for item in list(coll):
            coll.remove(item)


def limb(side, scale=(0.6, 0.6, 3.0), location=(0, 0, 0)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=side, ring_count=side, radius=1.0,
                                         location=location)
    obj = bpy.context.active_object
    obj.scale = scale
    bpy.ops.object.transform_apply(location=True, scale=True)
    return obj


def add_noise(obj, amount, seed=1):
    rng = np.random.default_rng(seed)
    n = len(obj.data.vertices)
    co = np.empty(n * 3)
    obj.data.vertices.foreach_get('co', co)
    co += rng.normal(0.0, amount, n * 3)
    obj.data.vertices.foreach_set('co', co)
    obj.data.update()


def join(objs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    return bpy.context.active_object


CHAIN = [((0, 0, -2.6), -1), ((0, 0, -1.0), 0), ((0, 0, 0.6), 1), ((0, 0, 2.2), 2),
         ((0.35, 0, 0.0), 1)]
HAND = [((0, 0, -2.0), -1), ((0, 0, 0.0), 0),
        ((-0.9, 0, 1.4), 1), ((-0.9, 0, 2.6), 2),
        ((0.0, 0, 1.6), 1), ((0.0, 0, 2.9), 4),
        ((0.9, 0, 1.4), 1), ((0.9, 0, 2.6), 6),
        ((1.3, 0, 0.2), 1)]


def scenario_mesh(kind):
    if kind == "limb":
        return limb(140)
    if kind == "noisy":
        obj = limb(140)
        add_noise(obj, 0.04)
        return obj
    if kind == "joined":
        a = limb(120)
        b = limb(40, scale=(0.3, 0.3, 0.3), location=(0.9, 0.0, 1.5))
        c = limb(40, scale=(0.25, 0.25, 0.6), location=(0.0, 0.0, 3.4))
        return join([a, b, c])
    if kind == "hand":
        obj = limb(160, scale=(1.4, 0.4, 3.0))
        add_noise(obj, 0.02, seed=7)
        return obj
    raise ValueError(kind)


SCENARIOS = [
    # (mesh, points, weight_falloff, falloff_softness, joint_blend, refresh settings,
    #  auto_thickness -- only used by versions that have that option)
    ("limb", CHAIN, 0.4, 0.3, 0.0, (0.8, 0.5, 0.0), True),
    ("limb", CHAIN, 1.0, 0.6, 0.5, (0.2, 0.3, 1.0), False),
    ("noisy", CHAIN, 0.3, 0.3, 0.0, (1.5, 0.9, 0.3), True),
    ("noisy", CHAIN, 0.2, 0.1, 0.0, (0.1, 0.05, 0.0), False),
    ("joined", CHAIN, 0.6, 0.3, 0.0, (1.2, 0.4, 0.7), True),
    ("hand", HAND, 0.8, 0.3, 0.0, (1.6, 1.0, 0.4), True),
    ("hand", HAND, 1.6, 0.5, 0.8, (0.5, 0.3, 0.0), False),
    ("hand", HAND, 0.15, 0.05, 0.0, (0.3, 0.1, 0.2), False),
]


def read_weights(obj):
    n = len(obj.data.vertices)
    names = [vg.name for vg in obj.vertex_groups]
    w = np.zeros((len(names), n))
    for v in obj.data.vertices:
        for g in v.groups:
            w[g.group, v.index] = g.weight
    return dict(zip(names, w))


def run(mod, scenario):
    kind, points, wf, fs, jb, refresh, auto_thickness = scenario
    mod.register()
    reset_scene()
    obj = scenario_mesh(kind)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='SCULPT')
    props = obj.pose_chain_tool
    props.points.clear()
    for co, parent in points:
        p = props.points.add()
        p.co = co
        p.parent_index = parent
    props.max_points = len(points)
    props.weight_falloff, props.falloff_softness, props.joint_blend = wf, fs, jb
    extra = {}
    if hasattr(props, "auto_thickness"):
        props.auto_thickness = auto_thickness
        extra["auto_thickness"] = auto_thickness

    cleaned = []
    original_cleanup = mod.clean_up_weight_islands

    def counting_cleanup(*args, **kwargs):
        r = original_cleanup(*args, **kwargs)
        cleaned.append(r)
        return r

    mod.clean_up_weight_islands = counting_cleanup
    try:
        ok = mod.build_chain(bpy.context, obj)
        assert ok, "build_chain failed"
        after_build = read_weights(obj)

        # Live slider refresh, same steps as _schedule_corridor_refresh.
        positions = mod._geodesic_cache[obj.name]
        bone_names = [f"Bone_{i}" for i in range(1, len(props.points))]
        parent_indices = [p.parent_index for p in props.points]
        rwf, rfs, rjb = refresh
        _p, valid, weight = mod.apply_uniform_falloff_weights(
            obj, props.armature_obj, bone_names, parent_indices, positions, rwf, rfs, rjb,
            **extra)
        mod.clean_up_weight_islands(obj, bone_names, positions, valid, weight, rfs,
                                    mod.ISLAND_CLEANUP_THRESHOLD)
        after_refresh = read_weights(obj)
    finally:
        mod.clean_up_weight_islands = original_cleanup
    n_verts = len(obj.data.vertices)
    mod.cleanup_chain(bpy.context, obj)
    mod.unregister()
    return n_verts, after_build, after_refresh, cleaned


def compare(a, b):
    if sorted(a) != sorted(b):
        return None, f"group names differ: {sorted(a)} vs {sorted(b)}"
    worst = max((float(np.abs(a[k] - b[k]).max()) for k in a), default=0.0)
    return worst, ""


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not argv:
        print("usage: ... --python tools/verify_weights.py -- <reference __init__.py>")
        sys.exit(2)
    ref = load(os.path.abspath(argv[0]), "pct_reference")
    cur = load(os.path.join(REPO, "__init__.py"), "pct_current")

    failures = 0
    for i, sc in enumerate(SCENARIOS, 1):
        n, ref_build, ref_refresh, ref_clean = run(ref, sc)
        _n, cur_build, cur_refresh, cur_clean = run(cur, sc)
        d_build, msg1 = compare(ref_build, cur_build)
        d_refresh, msg2 = compare(ref_refresh, cur_refresh)
        good = (d_build is not None and d_refresh is not None
                and d_build <= TOL and d_refresh <= TOL and ref_clean == cur_clean)
        failures += 0 if good else 1
        print(f"[{'PASS' if good else 'FAIL'}] {i}. {sc[0]:<6} {n:>6,} verts, "
              f"bones={len(sc[1]) - 1}, settings={sc[2:5]} -> refresh {sc[5]}")
        print(f"      max weight diff: build={d_build} refresh={d_refresh} {msg1}{msg2}")
        print(f"      island cleanup touched (ref / new): {ref_clean} / {cur_clean}")
    print(f"\n{len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios match")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
