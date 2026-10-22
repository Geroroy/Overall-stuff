"""Minimal glTF 2.0 (GLB) writer for decoded Frostbite meshes.

Coordinate conversion: Frostbite data is left-handed (X right, Y up, Z forward;
Frosty's own viewport renders it with LookAtLH/PerspectiveFovLH). glTF is
right-handed with +Y up, so the default ``axis="gltf"`` mode maps
(x, y, z) -> (-x, y, z): the model keeps its real chirality and still faces +Z.
Because that map is a reflection, triangle winding is reversed and tangent
handedness (w) is negated. ``axis="frosty"`` writes raw coordinates instead,
reproducing Frosty Editor's FBX export (mirrored along X when viewed in a
right-handed application), which is what Frosty expects on re-import.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mesh import DecodedSection

FLOAT, USHORT, UINT, UBYTE = 5126, 5123, 5125, 5121
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
TYPES = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}
AXIS_MODES = ("gltf", "frosty")


@dataclass
class SkeletonData:
    names: list[str]
    parents: list[int]
    local: list[np.ndarray]  # 4x4 local transforms (Frostbite row-vector convention, LH)
    source: str


@dataclass
class GltfMaterial:
    name: str
    base_color_uri: str | None = None
    normal_uri: str | None = None
    extras: dict = field(default_factory=dict)


class _Builder:
    def __init__(self):
        self.bin = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add(self, arr: np.ndarray, component: int, target: int | None = None, minmax: bool = False,
            normalized: bool = False) -> int:
        arr = np.ascontiguousarray(arr)
        while len(self.bin) % 4:
            self.bin.append(0)
        offset = len(self.bin)
        self.bin += arr.tobytes()
        view = {"buffer": 0, "byteOffset": offset, "byteLength": arr.nbytes}
        if target:
            view["target"] = target
        self.views.append(view)
        comps = 1 if arr.ndim == 1 else arr.shape[1]
        acc = {"bufferView": len(self.views) - 1, "componentType": component, "count": int(arr.shape[0]),
               "type": TYPES[comps]}
        if normalized:
            acc["normalized"] = True
        if minmax:
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def convert_positions(p: np.ndarray, axis: str) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32).copy()
    if axis == "gltf":
        p[:, 0] = -p[:, 0]
    return p


def _lt_to_matrix(m: np.ndarray, axis: str) -> np.ndarray:
    """Frostbite LinearTransform (rows right/up/forward/trans, row vectors) -> column-major glTF matrix."""
    col = m.T.copy()  # column-vector convention
    if axis == "gltf":
        s = np.diag([-1.0, 1.0, 1.0, 1.0])
        col = s @ col @ s
    return col


def _decompose(m: np.ndarray):
    t = m[:3, 3].copy()
    basis = m[:3, :3].copy()
    sx, sy, sz = (np.linalg.norm(basis[:, i]) for i in range(3))
    scale = np.array([sx, sy, sz])
    if np.linalg.det(basis) < 0:
        scale[0] = -scale[0]
    r = basis / np.where(scale == 0, 1, scale)
    # rotation matrix -> quaternion (x, y, z, w)
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return t, q, scale


def bind_pose_world(skeleton: SkeletonData, axis: str) -> list[np.ndarray]:
    """Model-space bind matrices (column-vector convention) from the local pose chain."""
    n = len(skeleton.names)
    world: list[np.ndarray | None] = [None] * n

    def resolve(i: int, depth: int = 0) -> np.ndarray:
        if world[i] is None:
            if depth > n:
                raise ValueError("cycle in skeleton hierarchy")
            m = _lt_to_matrix(skeleton.local[i], axis)
            p = skeleton.parents[i]
            world[i] = resolve(p, depth + 1) @ m if 0 <= p < n and p != i else m
        return world[i]

    return [resolve(i) for i in range(n)]


def write_glb(path: Path, mesh_name: str, lods: list[list[DecodedSection]], materials: list[GltfMaterial],
              section_material: dict[tuple[int, int], int], axis: str = "gltf",
              skeleton: SkeletonData | None = None, extras: dict | None = None,
              keep_normals: dict[tuple[int, int], bool] | None = None,
              flip_winding: dict[tuple[int, int], bool] | None = None) -> dict:
    """Write one GLB holding every LOD in ``lods`` as a separate mesh/node.

    ``flip_winding`` (per (lod, section)) is applied on top of the reversal
    implied by the axis reflection; the caller decides it from the
    normal/winding check. Returns a summary dict.
    """
    if axis not in AXIS_MODES:
        raise ValueError(axis)
    b = _Builder()
    gl_meshes, nodes, scene_nodes = [], [], []
    images, textures, gl_materials = [], [], []
    image_index: dict[str, int] = {}

    def tex_index(uri: str) -> int:
        if uri not in image_index:
            images.append({"uri": uri})
            textures.append({"source": len(images) - 1})
            image_index[uri] = len(textures) - 1
        return image_index[uri]

    for m in materials:
        mat: dict = {"name": m.name, "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0},
                     "doubleSided": False}
        if m.base_color_uri:
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex_index(m.base_color_uri)}
        if m.normal_uri:
            mat["normalTexture"] = {"index": tex_index(m.normal_uri)}
        if m.extras:
            mat["extras"] = m.extras
        gl_materials.append(mat)

    summary = {"lods": []}
    joint_nodes: list[int] = []
    skin_index = None

    if skeleton is not None:
        first = len(nodes)
        for i, (name, local) in enumerate(zip(skeleton.names, skeleton.local)):
            t, q, s = _decompose(_lt_to_matrix(local, axis))
            nodes.append({"name": name, "translation": t.tolist(), "rotation": q.tolist(), "scale": s.tolist()})
        for i, parent in enumerate(skeleton.parents):
            if 0 <= parent < len(skeleton.names) and parent != i:
                nodes[first + parent].setdefault("children", []).append(first + i)
            else:
                scene_nodes.append(first + i)
        joint_nodes = list(range(first, first + len(skeleton.names)))
        # inverse bind matrices from the bind pose (accumulated local transforms)
        world = bind_pose_world(skeleton, axis)
        ibm = np.stack([np.linalg.inv(w).T.reshape(-1) for w in world]).astype(np.float32)  # column-major
        ibm_acc = b.add(ibm.reshape(-1, 16), FLOAT)
        skin_index = 0

    for lod_sections in lods:
        if not lod_sections:
            continue
        lod_idx = lod_sections[0].lod_index
        prims = []
        for ds in lod_sections:
            attrs = {}
            pos = convert_positions(ds.positions, axis)
            attrs["POSITION"] = b.add(pos, FLOAT, ARRAY_BUFFER, minmax=True)
            keep = keep_normals.get((ds.lod_index, ds.section_index), True) if keep_normals else True
            if ds.normals is not None and keep:
                attrs["NORMAL"] = b.add(convert_positions(ds.normals, axis), FLOAT, ARRAY_BUFFER)
                if ds.tangents is not None:
                    t = ds.tangents.copy()
                    t[:, :3] = convert_positions(t[:, :3], axis)
                    if axis == "gltf":
                        t[:, 3] = -t[:, 3]
                    attrs["TANGENT"] = b.add(t.astype(np.float32), FLOAT, ARRAY_BUFFER)
            for k, name in enumerate(sorted(ds.uvs, key=lambda n: int(n[-1]))):
                attrs[f"TEXCOORD_{k}"] = b.add(ds.uvs[name].astype(np.float32), FLOAT, ARRAY_BUFFER)
            for k, name in enumerate(sorted(ds.colors)):
                attrs[f"COLOR_{k}"] = b.add(ds.colors[name].astype(np.float32), FLOAT, ARRAY_BUFFER)
            if ds.joints is not None and ds.weights is not None:
                for set_i in range(ds.joints.shape[1] // 4):
                    j = ds.joints[:, set_i * 4:(set_i + 1) * 4].astype(np.uint16)
                    w = ds.weights[:, set_i * 4:(set_i + 1) * 4].astype(np.float32)
                    if skeleton is not None:
                        attrs[f"JOINTS_{set_i}"] = b.add(j, USHORT, ARRAY_BUFFER)
                        attrs[f"WEIGHTS_{set_i}"] = b.add(w, FLOAT, ARRAY_BUFFER)
                    else:
                        # FLOAT keeps indices exact (< 2**24) through Blender's importer, which maps
                        # integer custom attributes to lossy byte colours.
                        attrs[f"_BONE_INDICES_{set_i}"] = b.add(j.astype(np.float32), FLOAT, ARRAY_BUFFER)
                        attrs[f"_BONE_WEIGHTS_{set_i}"] = b.add(w, FLOAT, ARRAY_BUFFER)
            tris = ds.indices.astype(np.uint32)
            flip = bool(flip_winding.get((ds.lod_index, ds.section_index), False)) if flip_winding else False
            if (axis == "gltf") ^ flip:
                tris = tris[:, [0, 2, 1]]
            idx_acc = b.add(tris.reshape(-1), UINT, ELEMENT_ARRAY_BUFFER)
            prim = {"attributes": attrs, "indices": idx_acc, "mode": 4,
                    "extras": {"section_index": ds.section_index, "material_name": ds.material_name,
                               "categories": ds.categories}}
            mat = section_material.get((ds.lod_index, ds.section_index))
            if mat is not None:
                prim["material"] = mat
            prims.append(prim)
        gl_meshes.append({"name": f"{mesh_name}_lod{lod_idx}", "primitives": prims})
        node = {"name": f"{mesh_name}_lod{lod_idx}", "mesh": len(gl_meshes) - 1,
                "extras": {"lod": lod_idx}}
        if skin_index is not None and any(ds.joints is not None for ds in lod_sections):
            node["skin"] = skin_index
        nodes.append(node)
        scene_nodes.append(len(nodes) - 1)
        summary["lods"].append({"lod": lod_idx, "primitives": len(prims),
                                "vertices": int(sum(len(ds.positions) for ds in lod_sections)),
                                "triangles": int(sum(len(ds.indices) for ds in lod_sections))})

    doc = {
        "asset": {"version": "2.0", "generator": "fbmod-extractor"},
        "scene": 0,
        "scenes": [{"name": mesh_name, "nodes": scene_nodes}],
        "nodes": nodes,
        "meshes": gl_meshes,
        "accessors": b.accessors,
        "bufferViews": b.views,
        "buffers": [{"byteLength": len(b.bin)}],
        "extras": dict(extras or {}, axis_mode=axis),
    }
    if gl_materials:
        doc["materials"] = gl_materials
    if images:
        doc["images"] = images
        doc["textures"] = textures
        doc["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
        for t in textures:
            t["sampler"] = 0
    if skin_index is not None:
        doc["skins"] = [{"name": skeleton.source.replace("\\", "/").rsplit("/", 1)[-1], "joints": joint_nodes,
                         "inverseBindMatrices": ibm_acc}]

    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    while len(b.bin) % 4:
        b.bin.append(0)
    total = 12 + 8 + len(js) + 8 + len(b.bin)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", len(b.bin), 0x004E4942))
        f.write(bytes(b.bin))
    summary["file"] = str(path)
    summary["bytes"] = total
    return summary


def read_glb(path: Path) -> tuple[dict, bytes]:
    data = Path(path).read_bytes()
    magic, version, total = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2 or total != len(data):
        raise ValueError("not a valid GLB")
    jlen, jtype = struct.unpack_from("<II", data, 12)
    doc = json.loads(data[20:20 + jlen])
    blen, _ = struct.unpack_from("<II", data, 20 + jlen)
    return doc, data[28 + jlen:28 + jlen + blen]


def accessor_array(doc: dict, binary: bytes, index: int) -> np.ndarray:
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    dtype = {FLOAT: "<f4", USHORT: "<u2", UINT: "<u4", UBYTE: "u1"}[acc["componentType"]]
    comps = {v: k for k, v in TYPES.items()}[acc["type"]]
    arr = np.frombuffer(binary, dtype=dtype, count=acc["count"] * comps,
                        offset=view["byteOffset"] + acc.get("byteOffset", 0))
    return arr.reshape(acc["count"], comps) if comps > 1 else arr
