"""Blender script: re-import an exported FBX and check it against expectations from the decoder.

Run: blender -b --factory-startup --python validate_fbx.py -- --fbx X.fbx --expect X.expect.json --out X.validation.json
"""

import argparse
import json
import math
import sys

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--fbx", required=True)
    p.add_argument("--expect", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def close(a, b, tol):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def main():
    args = parse_args()
    expect = json.load(open(args.expect))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=args.fbx, use_image_search=False)

    checks = []

    def check(name, ok, detail=None):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    meshes = {ob.name: ob for ob in bpy.data.objects if ob.type == "MESH"}
    armatures = [ob for ob in bpy.data.objects if ob.type == "ARMATURE"]
    result = {"fbx": args.fbx, "objects": []}

    for lod in expect["lods"]:
        ob = meshes.get(lod["object"])
        if ob is None:
            # Blender may suffix duplicate names; match by prefix
            ob = next((o for n, o in meshes.items() if n.startswith(lod["object"])), None)
        check(f"{lod['object']}: object present", ob is not None)
        if ob is None:
            continue
        me = ob.data
        me.calc_loop_triangles()
        nv, nf = len(me.vertices), len(me.polygons)
        info = {"name": ob.name, "vertices": nv, "polygons": nf, "uv_layers": len(me.uv_layers),
                "materials": [m.name if m else None for m in me.materials], "vertex_groups": len(ob.vertex_groups)}
        result["objects"].append(info)
        check(f"{ob.name}: nonzero vertices and triangles", nv > 0 and nf > 0, {"vertices": nv, "polygons": nf})
        bad = [p.index for p in me.polygons if any(v >= nv for v in p.vertices)]
        check(f"{ob.name}: face indices valid", not bad, {"invalid_faces": len(bad)})
        non_tri = sum(1 for p in me.polygons if len(p.vertices) != 3)
        check(f"{ob.name}: all faces are triangles", non_tri == 0, {"non_triangles": non_tri})
        check(f"{ob.name}: vertex count matches decoded (used) vertices", nv == lod["vertices_used"],
              {"blender": nv, "expected": lod["vertices_used"]})
        check(f"{ob.name}: triangle count matches decoded (non-degenerate) triangles", nf == lod["triangles_nondegenerate"],
              {"blender": nf, "expected": lod["triangles_nondegenerate"]})

        # bounding box in Blender world space
        mw = ob.matrix_world
        coords = [mw @ v.co for v in me.vertices]
        if coords:
            bmin = [min(c[i] for c in coords) for i in range(3)]
            bmax = [max(c[i] for c in coords) for i in range(3)]
            info["bbox_min"], info["bbox_max"] = bmin, bmax
            tol = lod.get("bbox_tolerance", 2e-3)
            ok = close(bmin, lod["bbox_min"], tol) and close(bmax, lod["bbox_max"], tol)
            check(f"{ob.name}: scale and orientation (bbox matches decoded data after axis conversion)", ok,
                  {"blender_min": bmin, "blender_max": bmax, "expected_min": lod["bbox_min"],
                   "expected_max": lod["bbox_max"]})
            finite = all(math.isfinite(x) for c in coords for x in c)
            check(f"{ob.name}: coordinates finite", finite)

        # UVs
        if lod.get("uv_bounds"):
            check(f"{ob.name}: UV layer present", len(me.uv_layers) > 0, {"uv_layers": len(me.uv_layers)})
            if me.uv_layers:
                uv = me.uv_layers[0].data
                us = [d.uv[0] for d in uv]
                vs = [d.uv[1] for d in uv]
                umin, umax, vmin, vmax = min(us), max(us), min(vs), max(vs)
                e = lod["uv_bounds"]  # already converted to Blender's V-up convention
                ok = close([umin, vmin, umax, vmax], [e[0][0], e[0][1], e[1][0], e[1][1]], 2e-3)
                check(f"{ob.name}: UV bounds match decoded UVs (V flipped for Blender)", ok,
                      {"blender": [[umin, vmin], [umax, vmax]], "expected": e})

        # per-section triangle counts through material slots
        per_slot = {}
        for p in me.polygons:
            per_slot[p.material_index] = per_slot.get(p.material_index, 0) + 1
        got = sorted(per_slot.values())
        want = sorted(x for x in lod["section_triangles"] if x > 0)
        check(f"{ob.name}: no missing sections (triangles per material slot)", got == want or
              sum(got) == sum(want) and len(me.materials) >= lod["distinct_materials"],
              {"blender": got, "expected": want, "material_slots": len(me.materials)})

        # textures
        tex = {}
        for m in me.materials:
            if m is None or not m.use_nodes:
                continue
            imgs = [n.image for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image]
            tex[m.name] = [{"image": i.name, "size": list(i.size), "has_data": i.has_data or bool(i.packed_file)}
                           for i in imgs]
        info["textures"] = tex
        for mname, want_count in lod.get("expected_textures", {}).items():
            got_imgs = next((v for k, v in tex.items() if k.startswith(mname)), [])
            ok = len(got_imgs) >= want_count and all(i["size"][0] > 0 for i in got_imgs)
            check(f"{ob.name}: material '{mname}' has its {want_count} texture(s) with pixel data", ok,
                  {"images": got_imgs})

    sk = expect.get("skeleton")
    if sk:
        check("armature present", len(armatures) == 1, {"armatures": len(armatures)})
        if armatures:
            arm = armatures[0]
            bones = arm.data.bones
            names = [b.name for b in bones]
            check("armature bone names match skeleton", sorted(names) == sorted(sk["names"]),
                  {"blender": len(names), "expected": len(sk["names"])})
            errs = []
            for name, head in zip(sk["names"], sk["heads"]):
                b = bones.get(name)
                if b is None:
                    continue
                w = arm.matrix_world @ b.head_local
                errs.append(max(abs(w[i] - head[i]) for i in range(3)))
            check("bone bind positions match skeleton", errs and max(errs) < 2e-3,
                  {"max_error": max(errs) if errs else None})
    result["checks"] = checks
    result["passed"] = all(c["passed"] for c in checks)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


main()
