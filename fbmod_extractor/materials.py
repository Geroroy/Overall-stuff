"""Material, texture-parameter and texture-role resolution from decoded EBX."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ebx import EbxPartition


@dataclass
class TextureParam:
    parameter: str
    partition_guid: str
    instance_guid: str

    def describe(self) -> dict:
        return {"parameter": self.parameter, "texture_partition": self.partition_guid,
                "texture_instance": self.instance_guid}


@dataclass
class MaterialInfo:
    index: int
    instance_guid: str | None
    shader: dict | None
    textures: list[TextureParam] = field(default_factory=list)
    vectors: list[dict] = field(default_factory=list)

    def describe(self) -> dict:
        return {
            "index": self.index,
            "instance_guid": self.instance_guid,
            "shader": self.shader,
            "texture_parameters": [t.describe() for t in self.textures],
            "vector_parameters": self.vectors,
        }


def _walk(value, out_tex: list, out_vec: list, depth: int = 0) -> None:
    if depth > 12:
        return
    if isinstance(value, dict):
        name = value.get("ParameterName")
        v = value.get("Value")
        if isinstance(name, str):
            if isinstance(v, dict) and "$import" in v:
                imp = v["$import"]
                out_tex.append(TextureParam(name, imp["partition"], imp["instance"]))
                return
            if isinstance(v, dict) and {"x", "y", "z", "w"} <= set(v):
                out_vec.append({"parameter": name, "value": [v["x"], v["y"], v["z"], v["w"]]})
                return
        for child in value.values():
            _walk(child, out_tex, out_vec, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _walk(child, out_tex, out_vec, depth + 1)


def collect_parameters(value) -> tuple[list[TextureParam], list[dict]]:
    tex: list[TextureParam] = []
    vec: list[dict] = []
    _walk(value, tex, vec)
    return tex, vec


def _shader_ref(fields: dict) -> dict | None:
    shader = fields.get("Shader")
    if isinstance(shader, dict):
        inner = shader.get("Shader")
        if isinstance(inner, dict) and "$import" in inner:
            return inner["$import"]
    return None


def mesh_materials(part: EbxPartition) -> list[MaterialInfo]:
    """Materials of a MeshAsset partition, in the order sections index them (MaterialId)."""
    root = part.root
    if root is None:
        return []
    out = []
    for i, ptr in enumerate(root.fields.get("Materials") or []):
        obj = part.deref(ptr)
        if obj is None:
            out.append(MaterialInfo(i, None, None))
            continue
        tex, vec = collect_parameters(obj.fields)
        out.append(MaterialInfo(i, obj.instance_guid, _shader_ref(obj.fields), tex, vec))
    return out


def variation_entries(part: EbxPartition) -> list[dict]:
    """Texture overrides from a MeshVariationDatabase (or similar) partition.

    Every decoded dict that names a ``Mesh`` import and carries ``Materials`` is
    reported with the texture parameters found under each material entry.
    """
    found: list[dict] = []

    def visit(value, depth=0):
        if depth > 10:
            return
        if isinstance(value, dict):
            mesh = value.get("Mesh")
            mats = value.get("Materials")
            if isinstance(mesh, dict) and "$import" in mesh and isinstance(mats, list):
                entry = {"mesh": mesh["$import"], "variation_name_hash": value.get("VariationAssetNameHash"),
                         "materials": []}
                for m in mats:
                    if not isinstance(m, dict):
                        continue
                    target = m.get("Material")
                    tex, vec = collect_parameters({k: v for k, v in m.items() if k != "Material"})
                    entry["materials"].append({
                        "material": target.get("$import") if isinstance(target, dict) else None,
                        "texture_parameters": [t.describe() for t in tex],
                        "vector_parameters": vec,
                    })
                found.append(entry)
                return
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                visit(child, depth + 1)

    for obj in part.objects:
        visit(obj.fields)
    return found


# -- texture roles -------------------------------------------------------------

# Channel packing used by SWBF2 character textures, per the community
# documentation cited in docs/FORMAT_NOTES.md. Treated as a convention, not as
# decoded fact: the report marks every role derived from it as "convention".
SUFFIX_CONVENTIONS = [
    (re.compile(r"(?:^|_)cs$"), "color_smoothness",
     {"base_color": "RGB", "smoothness": "A"}),
    (re.compile(r"(?:^|_)(?:nm|nam|normal|nrm)$"), "normal_metallic",
     {"normal": "RGB", "metallic": "A"}),
    (re.compile(r"(?:^|_)(?:aosl|ao)$"), "occlusion_pack", {"occlusion": "R"}),
    (re.compile(r"(?:^|_)(?:d|diff|diffuse|albedo|basecolor|col|color)$"), "base_color", {"base_color": "RGB"}),
    (re.compile(r"(?:^|_)(?:mask|msk)$"), "mask", {}),
    (re.compile(r"(?:^|_)(?:op|opacity|alpha)$"), "opacity", {"opacity": "R"}),
    (re.compile(r"(?:^|_)(?:em|emissive|glow)$"), "emissive", {"emissive": "RGB"}),
]


def infer_texture_role(texture_name: str | None, parameter: str | None = None) -> dict:
    """Guess a texture's role from its asset name suffix (and shader parameter name)."""
    candidates = []
    if texture_name:
        candidates.append(("texture name suffix", texture_name.rsplit("/", 1)[-1].lower()))
    if parameter:
        candidates.append(("shader parameter name", parameter.lower().lstrip("_")))
    for basis, key in candidates:
        for pattern, role, channels in SUFFIX_CONVENTIONS:
            if pattern.search(key):
                return {"role": role, "channels": channels, "basis": basis, "confidence": "convention"}
    return {"role": "unknown", "channels": {}, "basis": None, "confidence": "none"}
