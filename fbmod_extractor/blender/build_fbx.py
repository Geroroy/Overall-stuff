"""Blender script: GLB produced by fbmod-extractor -> .blend + .fbx.

Run: blender -b --factory-startup --python build_fbx.py -- --glb X.glb --fbx X.fbx --blend X.blend --summary X.json
"""

import argparse
import json
import sys

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--fbx", required=True)
    p.add_argument("--blend")
    p.add_argument("--summary", required=True)
    return p.parse_args(argv)


def bone_attrs_to_vertex_groups(ob):
    """Turn _BONE_INDICES_n/_BONE_WEIGHTS_n point attributes into vertex groups named bone_<index>."""
    me = ob.data
    created = 0
    k = 0
    while f"_BONE_INDICES_{k}" in me.attributes and f"_BONE_WEIGHTS_{k}" in me.attributes:
        idx_attr = me.attributes[f"_BONE_INDICES_{k}"]
        w_attr = me.attributes[f"_BONE_WEIGHTS_{k}"]
        if idx_attr.domain != "POINT" or w_attr.domain != "POINT":
            break
        for v_i, (ia, wa) in enumerate(zip(idx_attr.data, w_attr.data)):
            idx = ia.color if hasattr(ia, "color") else ia.vector
            wts = wa.color if hasattr(wa, "color") else wa.vector
            for c in range(4):
                w = float(wts[c])
                if w <= 0.0:
                    continue
                name = f"bone_{int(round(idx[c])):03d}"
                vg = ob.vertex_groups.get(name) or ob.vertex_groups.new(name=name)
                vg.add([v_i], w, "ADD")
                created += 1
        k += 1
    return created


def add_channel_textures(mat):
    """Wire roughness / metallic images listed in the material's fbmod_channels extras."""
    if not mat.use_nodes or "fbmod_channels" not in mat.keys():
        return []
    channels = mat["fbmod_channels"].to_dict() if hasattr(mat["fbmod_channels"], "to_dict") else dict(mat["fbmod_channels"])
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return []
    wired = []
    import os
    base = os.path.dirname(bpy.path.abspath(ARGS.glb))
    for key, socket in (("roughness", "Roughness"), ("metallic", "Metallic")):
        info = channels.get(key)
        if not info or bsdf.inputs[socket].is_linked:
            continue
        path = os.path.normpath(os.path.join(base, info["file"]))
        if not os.path.isfile(path):
            continue
        img = bpy.data.images.load(path, check_existing=True)
        img.colorspace_settings.name = "Non-Color"
        img.pack()
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = img
        node.label = f"{key} (fbmod)"
        nt.links.new(node.outputs["Color"], bsdf.inputs[socket])
        if key == "metallic":
            bsdf.inputs["Metallic"].default_value = 0.0
        wired.append(key)
    return wired


ARGS = parse_args()


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    bpy.ops.import_scene.gltf(filepath=ARGS.glb, disable_bone_shape=True, merge_vertices=False,
                              import_shading="NORMALS", guess_original_bind_pose=True)

    summary = {"objects": [], "materials": [], "images": []}
    for ob in list(bpy.data.objects):
        entry = {"name": ob.name, "type": ob.type}
        if ob.type == "MESH":
            has_armature = any(m.type == "ARMATURE" for m in ob.modifiers)
            entry["vertex_groups_from_attributes"] = 0 if has_armature else bone_attrs_to_vertex_groups(ob)
            me = ob.data
            entry.update(vertices=len(me.vertices), polygons=len(me.polygons), uv_layers=len(me.uv_layers),
                         vertex_groups=len(ob.vertex_groups), materials=[m.name if m else None for m in me.materials],
                         armature=has_armature)
        if ob.type == "ARMATURE":
            entry["bones"] = len(ob.data.bones)
        summary["objects"].append(entry)
    for mat in bpy.data.materials:
        summary["materials"].append({"name": mat.name, "extra_channels_wired": add_channel_textures(mat)})
    for img in bpy.data.images:
        if img.source == "FILE" and not img.packed_file:
            try:
                img.pack()
            except RuntimeError:
                pass
        summary["images"].append({"name": img.name, "size": list(img.size), "packed": bool(img.packed_file)})

    if ARGS.blend:
        bpy.ops.wm.save_as_mainfile(filepath=ARGS.blend, compress=True)
    bpy.ops.export_scene.fbx(
        filepath=ARGS.fbx, use_selection=False, object_types={"MESH", "ARMATURE", "EMPTY"},
        apply_unit_scale=True, apply_scale_options="FBX_SCALE_NONE", axis_forward="-Z", axis_up="Y",
        mesh_smooth_type="OFF", use_tspace=False, add_leaf_bones=False, bake_anim=False,
        path_mode="COPY", embed_textures=True,
    )
    summary["fbx"] = ARGS.fbx
    summary["blend"] = ARGS.blend
    with open(ARGS.summary, "w") as f:
        json.dump(summary, f, indent=1)


main()
