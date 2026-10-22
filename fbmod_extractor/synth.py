"""Synthetic SWBF2 mod builder: the writer side of every format this package reads.

Used by the automated tests and the ``selftest`` CLI command to exercise the
whole pipeline against data with known ground truth. Nothing here is used when
extracting a real mod.
"""

from __future__ import annotations

import hashlib
import io
import struct
import uuid
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .binary import guid_to_bytes
from .compression import compress
from .ebx import (EBX_V4, T_ARRAY, T_CSTRING, T_F32, T_I32, T_INHERITED, T_POINTER, T_RESREF, T_STRUCT,
                  name_hash)
from .fbmod import MAGIC
from .meshset import RES_TYPE_MESHSET
from .sdk_tables import load_render_formats
from .texture import RES_TYPE_TEXTURE
from .vertex import USAGE

RENDER = {v: k for k, v in load_render_formats().items()}
SQRT2 = np.sqrt(2.0)


# -- tangent frame encoding (inverse of vertex.unpack_tangent_space_quaternion) ----

def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def encode_tangent_space(tangent: np.ndarray, binormal: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Pack orthonormal frames into the 32-bit quaternion format the game decodes."""
    out = np.empty(len(normal), dtype=np.uint32)
    for i in range(len(normal)):
        n, t, b = normal[i], tangent[i], binormal[i]
        cx, cy = n, -t
        cz = np.cross(cx, cy)
        m = np.stack([cx, cy, cz], axis=1)  # columns
        q = _mat_to_quat(m)
        idx = int(np.argmax(np.abs(q)))
        q = q * (1.0 if q[idx] >= 0 else -1.0)
        others = [k for k in range(4) if k != idx]
        a, c, bb = q[others[0]], q[others[1]], q[others[2]]

        def quant(v, bits):
            return int(np.clip(np.rint((v + 1 / SQRT2) * ((1 << bits) - 1) / SQRT2), 0, (1 << bits) - 1))

        code = (quant(a, 10) << 22) | (quant(c, 9) << 13) | (quant(bb, 10) << 3) | (idx << 1)
        code |= 1 if np.dot(b, cz) > 0 else 0
        out[i] = code
    return out


# -- geometry -------------------------------------------------------------------

@dataclass
class SynthMesh:
    positions: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    binormals: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray  # (M, 3)
    joints: np.ndarray | None = None  # (N, 4)
    weights: np.ndarray | None = None  # (N, 4) uint8


def make_capsule(radius=0.3, height=1.6, rings=12, segs=16, y0=0.0, bones=(0, 1, 2)) -> SynthMesh:
    """A closed capsule-like body (cylinder + caps) in Frostbite's LH space, CW front faces."""
    ys, rs = [], []
    for i in range(rings + 1):
        t = i / rings
        ang = np.pi * (t - 0.5)
        ys.append(y0 + height / 2 + np.sin(ang) * height / 2)
        rs.append(max(radius * np.cos(ang) * 1.0, 1e-4) if i not in (0, rings) else 0.0)
    pos, nrm, tan, uv = [], [], [], []
    for i, (y, r) in enumerate(zip(ys, rs)):
        for j in range(segs + 1):
            phi = 2 * np.pi * j / segs
            p = np.array([np.cos(phi) * r, y, np.sin(phi) * r])
            ang = np.pi * (i / rings - 0.5)
            n = np.array([np.cos(phi) * np.cos(ang), np.sin(ang), np.sin(phi) * np.cos(ang)])
            pos.append(p)
            nrm.append(n / np.linalg.norm(n))
            tan.append(np.array([-np.sin(phi), 0.0, np.cos(phi)]))
            uv.append([j / segs, 1.0 - i / rings])
    pos, nrm, tan, uv = map(np.array, (pos, nrm, tan, uv))
    tri = []
    w = segs + 1
    for i in range(rings):
        for j in range(segs):
            a, b, c, d = i * w + j, i * w + j + 1, (i + 1) * w + j, (i + 1) * w + j + 1
            tri += [(a, b, c), (b, d, c)]
    tri = np.array(tri, dtype=np.uint32)
    # orient every triangle so that cross(e1, e2) points along the outward normal (LH + CW front)
    p = pos
    c = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    nn = nrm[tri].sum(axis=1)
    bad = np.einsum("ij,ij->i", c, nn) < 0
    tri[bad] = tri[bad][:, [0, 2, 1]]
    keep = np.linalg.norm(c, axis=1) > 1e-10
    tri = tri[keep]
    tan = tan - nrm * np.einsum("ij,ij->i", tan, nrm)[:, None]
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    # Game convention: bitangent follows +dP/dv with V growing downward in the texture.
    bino = np.cross(nrm, tan)
    joints = np.zeros((len(pos), 4), dtype=np.uint8)
    weights = np.zeros((len(pos), 4), dtype=np.uint8)
    hmin, hmax = pos[:, 1].min(), pos[:, 1].max()
    for k in range(len(pos)):
        f = (pos[k, 1] - hmin) / max(hmax - hmin, 1e-6)
        if f < 0.5:
            joints[k, :2] = (bones[0], bones[1])
            w1 = int(round(f * 2 * 255))
        else:
            joints[k, :2] = (bones[1], bones[2])
            w1 = int(round((f - 0.5) * 2 * 255))
        weights[k, :2] = (255 - w1, w1)
    return SynthMesh(pos, nrm, tan, bino, uv, tri, joints, weights)


# -- vertex/index buffers ----------------------------------------------------------------

DECL_SKINNED = [  # (usage, format, offset, stream)
    (USAGE["Pos"], 0x08, 0, 0),            # Half4
    (USAGE["BoneIndices"], 0x0C, 8, 0),    # UByte4
    (USAGE["BoneWeights"], 0x0D, 12, 0),   # UByte4N
    (USAGE["TangentSpace"], 0x20, 0, 1),   # UInt
    (USAGE["TexCoord0"], 0x06, 4, 1),      # Half2
]
STRIDES_SKINNED = [16, 8]
DECL_RIGID = [
    (USAGE["Pos"], 0x03, 0, 0),            # Float3
    (USAGE["TangentSpace"], 0x20, 12, 0),
    (USAGE["TexCoord0"], 0x06, 16, 0),
]
STRIDES_RIGID = [20]


def encode_vertices(m: SynthMesh, decl, strides) -> bytes:
    n = len(m.positions)
    ts = encode_tangent_space(m.tangents, m.binormals, m.normals)
    blocks = []
    for s, stride in enumerate(strides):
        buf = np.zeros((n, stride), dtype=np.uint8)
        for usage, fmt, off, stream in decl:
            if stream != s:
                continue
            if usage == USAGE["Pos"]:
                if fmt == 0x08:
                    v = np.concatenate([m.positions, np.ones((n, 1))], axis=1).astype("<f2")
                else:
                    v = m.positions.astype("<f4")
            elif usage == USAGE["TangentSpace"]:
                v = ts.astype("<u4")[:, None]
            elif usage == USAGE["TexCoord0"]:
                v = m.uvs.astype("<f2")
            elif usage == USAGE["BoneIndices"]:
                v = m.joints.astype("u1")
            elif usage == USAGE["BoneWeights"]:
                v = m.weights.astype("u1")
            else:
                raise ValueError(usage)
            raw = np.ascontiguousarray(v).view(np.uint8).reshape(n, -1)
            buf[:, off:off + raw.shape[1]] = raw
        blocks.append(buf.tobytes())
    return b"".join(blocks)


# -- MeshSet resource writer --------------------------------------------------------

@dataclass
class SynthSection:
    material_name: str
    material_id: int
    mesh: SynthMesh
    decl: list
    strides: list
    categories: tuple = ("opaque",)
    bone_list: tuple = ()


@dataclass
class SynthLod:
    sections: list[SynthSection]
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    index_size: int = 2
    inline: bool = False


def build_lod_buffers(lod: SynthLod) -> tuple[bytes, bytes, list[tuple[int, int]]]:
    vb, ib = bytearray(), bytearray()
    placements = []
    start_index = 0
    for sec in lod.sections:
        voff = len(vb)
        vb += encode_vertices(sec.mesh, sec.decl, sec.strides)
        idx = sec.mesh.indices.reshape(-1).astype("<u2" if lod.index_size == 2 else "<u4")
        ib += idx.tobytes()
        placements.append((voff, start_index))
        start_index += len(idx)
    return bytes(vb), bytes(ib), placements


def _decl_bytes(decl, strides) -> bytes:
    out = bytearray()
    elems = list(decl) + [(0, 0, 0xFF, 0)] * (16 - len(decl))
    for u, f, o, s in elems:
        out += bytes((u, f, o, s))
    st = list(strides) + [0] * (16 - len(strides))
    for s in st:
        out += bytes((s, 0))
    out += bytes((len(decl), len([s for s in strides if s]))) + b"\x00\x00"
    return bytes(out)


def build_meshset(full_name: str, mesh_type: int, lods: list[SynthLod], bbox, bone_count: int = 0) -> tuple[bytes, bytes, dict]:
    """Serialise a SWBF2 MeshSet. Returns (resource bytes, res meta, {chunk_id: chunk bytes})."""
    name = full_name.rsplit("/", 1)[-1]
    w = io.BytesIO()
    fix: list[tuple[int, str]] = []  # (position of i64 pointer, label)
    labels: dict[str, int] = {}

    def ptr(label):
        fix.append((w.tell(), label))
        w.write(b"\x00" * 8)

    def pad16():
        w.write(b"\x00" * ((-w.tell()) % 16))

    def vec3(v):
        w.write(struct.pack("<4f", v[0], v[1], v[2], 0.0))

    vec3(bbox[0])
    vec3(bbox[1])
    for i in range(6):
        if i < len(lods):
            ptr(f"lod{i}")
        else:
            w.write(b"\x00" * 8)
    w.write(b"\x00" * 8)
    ptr("fullname")
    ptr("name")
    w.write(struct.pack("<II", 0x1234 & 0xFFFFFFFF, mesh_type))
    w.write(b"\x00" * 24)
    w.write(struct.pack("<I", 0))
    w.write(b"\x00" * 4)
    total_sections = sum(len(lod.sections) for lod in lods)
    w.write(struct.pack("<HH", len(lods), total_sections))
    if mesh_type == 1:
        w.write(struct.pack("<HH", bone_count, bone_count))
        if bone_count:
            ptr("boneindices")
            w.write(b"\x00" * 8)
    pad16()

    chunks: dict[str, bytes] = {}
    lod_bufs = []
    for li, lod in enumerate(lods):
        vb, ib, placements = build_lod_buffers(lod)
        lod_bufs.append((vb, ib, placements))
        labels[f"lod{li}"] = w.tell()
        w.write(struct.pack("<II", mesh_type, 0))
        w.write(struct.pack("<i", len(lod.sections)))
        ptr(f"sections{li}")
        for cat in ("opaque", "transparent", "transparent_decal", "z_only", "shadow"):
            members = [i for i, s in enumerate(lod.sections) if cat in s.categories]
            w.write(struct.pack("<i", len(members)))
            ptr(f"cat{li}_{cat}")
        fmt = RENDER["R16_UINT"] if lod.index_size == 2 else RENDER["R32_UINT"]
        w.write(struct.pack("<IiII", 1, fmt, len(ib), len(vb)))
        w.write(struct.pack("<i", 0))
        w.write(guid_to_bytes("00000000-0000-0000-0000-000000000000" if lod.inline else lod.chunk_id))
        w.write(struct.pack("<I", 0 if lod.inline else 0xFFFFFFFF))
        w.write(b"\x00" * 8)
        ptr(f"lod{li}_debug")
        ptr(f"lod{li}_name")
        ptr(f"lod{li}_short")
        w.write(struct.pack("<I", 0))
        w.write(b"\x00" * 8)
        if mesh_type == 1:
            w.write(struct.pack("<I", bone_count))
            ptr(f"lod{li}_bones")
        pad16()
        if not lod.inline:
            chunks[lod.chunk_id] = vb + ib

    for li, lod in enumerate(lods):
        labels[f"sections{li}"] = w.tell()
        vb, ib, placements = lod_bufs[li]
        for si, sec in enumerate(lod.sections):
            start = w.tell()
            voff, sidx = placements[si]
            w.write(b"\x00" * 16)
            ptr(f"sec{li}_{si}_name")
            w.write(struct.pack("<iI", sec.material_id, 0))
            w.write(struct.pack("<IIII", len(sec.mesh.indices), sidx, voff, len(sec.mesh.positions)))
            w.write(bytes((sum(sec.strides), 3, 0, 0)))
            w.write(b"\x00" * 12)
            w.write(struct.pack("<BBH", 4 if sec.mesh.joints is not None else 0, 0, len(sec.bone_list)))
            if sec.bone_list:
                ptr(f"sec{li}_{si}_bones")
            else:
                w.write(b"\x00" * 8)
            w.write(b"\x00" * 8)
            w.write(_decl_bytes(sec.decl, sec.strides))
            w.write(_decl_bytes(sec.decl, sec.strides))
            w.write(struct.pack("<6f", 1, 1, 1, 1, 1, 1))
            w.write(b"\x00" * 44)
            assert w.tell() - start == 352
    pad16()
    for li, lod in enumerate(lods):
        for si, sec in enumerate(lod.sections):
            if sec.bone_list:
                labels[f"sec{li}_{si}_bones"] = w.tell()
                w.write(struct.pack(f"<{len(sec.bone_list)}H", *sec.bone_list))
    pad16()
    # strings
    for li, lod in enumerate(lods):
        for si, sec in enumerate(lod.sections):
            labels[f"sec{li}_{si}_name"] = w.tell()
            w.write(sec.material_name.encode() + b"\x00")
        labels[f"lod{li}_debug"] = w.tell()
        w.write(f"Mesh:{full_name}_lod{li}".encode() + b"\x00")
        labels[f"lod{li}_name"] = w.tell()
        w.write(f"{full_name}_lod{li}".encode() + b"\x00")
        labels[f"lod{li}_short"] = w.tell()
        w.write(f"{name}_lod{li}".encode() + b"\x00")
    labels["fullname"] = w.tell()
    w.write(full_name.encode() + b"\x00")
    labels["name"] = w.tell()
    w.write(name.encode() + b"\x00")
    pad16()
    for li, lod in enumerate(lods):
        for cat in ("opaque", "transparent", "transparent_decal", "z_only", "shadow"):
            labels[f"cat{li}_{cat}"] = w.tell()
            w.write(bytes(i for i, s in enumerate(lod.sections) if cat in s.categories))
    pad16()
    for li in range(len(lods)):
        labels[f"lod{li}_bones"] = w.tell()
        w.write(struct.pack(f"<{bone_count}I", *range(bone_count)))
    pad16()
    if mesh_type == 1 and bone_count:
        labels["boneindices"] = w.tell()
        w.write(struct.pack(f"<{bone_count}H", *range(bone_count)))
        pad16()
    header_end = w.tell()
    inline_size = 0
    for li, lod in enumerate(lods):
        if lod.inline:
            vb, ib, _ = lod_bufs[li]
            w.write(vb + ib)
            pad16()
    inline_size = w.tell() - header_end
    data = bytearray(w.getvalue())
    for pos, label in fix:
        struct.pack_into("<q", data, pos, labels.get(label, 0))
    meta = struct.pack("<IIIHH", header_end if inline_size else 0, inline_size, 0, 0, 0)
    return bytes(data), meta, chunks


# -- texture resource writer ----------------------------------------------------------

def make_texture_image(kind: str, size: int = 64) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    if kind == "cs":
        rgb = np.dstack([xx * 255 // size, yy * 255 // size, np.full_like(xx, 128)])
        a = ((xx // 8 + yy // 8) % 2) * 200 + 30
        return Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), "RGBA")
    if kind == "nm":
        n = np.dstack([np.full_like(xx, 128), np.full_like(xx, 128), np.full_like(xx, 255),
                       ((xx * 4) % 256)])
        return Image.fromarray(n.astype(np.uint8), "RGBA")
    raise ValueError(kind)


def encode_texture(img: Image.Image, fmt: str) -> bytes:
    """Encode mip 0 with Pillow (BC1/BC3/BC5) or raw RGBA."""
    if fmt == "R8G8B8A8_UNORM":
        return img.convert("RGBA").tobytes()
    pf = {"BC1_UNORM": "DXT1", "BC3_UNORM": "DXT5", "BC5_UNORM": "BC5"}[fmt]
    b = io.BytesIO()
    (img.convert("RGB") if fmt == "BC5_UNORM" else img).save(b, "DDS", pixel_format=pf)
    raw = b.getvalue()
    data = raw[148:] if raw[84:88] == b"DX10" else raw[128:]
    block = 8 if fmt == "BC1_UNORM" else 16
    if len(data) != (img.width // 4) * (img.height // 4) * block:
        raise RuntimeError(f"this Pillow ({Image.__version__}) cannot encode {fmt}; Pillow >= 11.2 is needed "
                           "for the synthetic test fixtures")
    return data


def build_texture(fmt: str, width: int, height: int, mip0: bytes, chunk_id: str, srgb: bool = False) -> bytes:
    sizes = [len(mip0)] + [0] * 14
    return (struct.pack("<IIIiIH", 0, 0, 0, RENDER[fmt], 0, 0x2 if srgb else 0)
            + struct.pack("<HHHHBB", width, height, 1, 1, 1, 0)
            + guid_to_bytes(chunk_id)
            + struct.pack("<15I", *sizes)
            + struct.pack("<II", len(mip0), 0)
            + b"group".ljust(16, b"\x00"))


# -- EBX writer -----------------------------------------------------------------------

@dataclass
class EbxClassDef:
    name: str
    fields: list  # (name, kind, ref, offset)  ref = class name for struct/array element/inherited
    size: int
    alignment: int = 4
    kind: int = T_POINTER  # 3 = class, 2 = struct


SCHEMA = {
    "DataContainer": EbxClassDef("DataContainer", [], 8),
    "Asset": EbxClassDef("Asset", [("$base", T_INHERITED, "DataContainer", 0), ("Name", T_CSTRING, None, 8)], 12),
    "Vec3": EbxClassDef("Vec3", [("x", T_F32, None, 0), ("y", T_F32, None, 4), ("z", T_F32, None, 8)], 16, 16, T_STRUCT),
    "LinearTransform": EbxClassDef("LinearTransform", [("right", T_STRUCT, "Vec3", 0), ("up", T_STRUCT, "Vec3", 16),
                                                       ("forward", T_STRUCT, "Vec3", 32), ("trans", T_STRUCT, "Vec3", 48)],
                                   64, 16, T_STRUCT),
    "TextureShaderParameter": EbxClassDef("TextureShaderParameter", [("Value", T_POINTER, None, 0),
                                                                     ("ParameterName", T_CSTRING, None, 4)], 8, 4, T_STRUCT),
    "SurfaceShaderInstanceDataStruct": EbxClassDef("SurfaceShaderInstanceDataStruct", [
        ("Shader", T_POINTER, None, 0), ("TextureParameters", T_ARRAY, "TextureShaderParameter", 16)], 28, 4, T_STRUCT),
    "MeshMaterial": EbxClassDef("MeshMaterial", [("$base", T_INHERITED, "DataContainer", 0),
                                                 ("Shader", T_STRUCT, "SurfaceShaderInstanceDataStruct", 8)], 36),
    "MeshBaseAsset": EbxClassDef("MeshBaseAsset", [("$base", T_INHERITED, "Asset", 0)], 12),
    "MeshAsset": EbxClassDef("MeshAsset", [("$base", T_INHERITED, "MeshBaseAsset", 0),
                                           ("Materials", T_ARRAY, "Pointer", 188),
                                           ("MeshSetResource", T_RESREF, None, 192)], 272, 16),
    "SkinnedMeshAsset": EbxClassDef("SkinnedMeshAsset", [("$base", T_INHERITED, "MeshAsset", 0)], 336, 16),
    "RigidMeshAsset": EbxClassDef("RigidMeshAsset", [("$base", T_INHERITED, "MeshAsset", 0)], 272, 16),
    "TextureBaseAsset": EbxClassDef("TextureBaseAsset", [("$base", T_INHERITED, "Asset", 0),
                                                         ("Resource", T_RESREF, None, 16)], 24, 8),
    "TextureAsset": EbxClassDef("TextureAsset", [("$base", T_INHERITED, "TextureBaseAsset", 0)], 32, 8),
    "SkeletonBaseAsset": EbxClassDef("SkeletonBaseAsset", [("$base", T_INHERITED, "Asset", 0)], 12),
    "SkeletonAsset": EbxClassDef("SkeletonAsset", [("$base", T_INHERITED, "SkeletonBaseAsset", 0),
                                                   ("BoneNames", T_ARRAY, "CString", 12),
                                                   ("Hierarchy", T_ARRAY, "Int32", 20),
                                                   ("LocalPose", T_ARRAY, "LinearTransform", 24)], 56),
    "MeshVariationDatabaseMaterial": EbxClassDef("MeshVariationDatabaseMaterial", [
        ("Material", T_POINTER, None, 0), ("TextureParameters", T_ARRAY, "TextureShaderParameter", 4)], 8, 4, T_STRUCT),
    "MeshVariationDatabaseEntry": EbxClassDef("MeshVariationDatabaseEntry", [
        ("Mesh", T_POINTER, None, 0), ("Materials", T_ARRAY, "MeshVariationDatabaseMaterial", 4),
        ("VariationAssetNameHash", T_I32, None, 8)], 12, 4, T_STRUCT),
    "MeshVariationDatabase": EbxClassDef("MeshVariationDatabase", [
        ("$base", T_INHERITED, "Asset", 0), ("Entries", T_ARRAY, "MeshVariationDatabaseEntry", 12)], 16),
}
_PRIM_ARRAY = {"Pointer": T_POINTER, "CString": T_CSTRING, "Int32": T_I32}


@dataclass
class EbxObj:
    cls: str
    fields: dict
    guid: str = field(default_factory=lambda: str(uuid.uuid4()))


def build_ebx(objects: list[EbxObj], imports: list[tuple[str, str]], partition_guid: str) -> bytes:
    """Serialise a self-describing EBX v4 partition (all objects exported, one record each)."""
    # collect classes (with bases and referenced structs), array pseudo-classes
    classes: list[str] = []

    def need(name):
        if name in classes or name in _PRIM_ARRAY:
            return
        cd = SCHEMA[name]
        for fname, kind, ref, _ in cd.fields:
            if kind in (T_INHERITED, T_STRUCT) or (kind == T_ARRAY and ref not in _PRIM_ARRAY):
                need(ref)
        classes.append(name)

    for o in objects:
        need(o.cls)
    array_classes = []
    for c in classes:
        for fname, kind, ref, _ in SCHEMA[c].fields:
            if kind == T_ARRAY and ref not in array_classes:
                array_classes.append(ref)
    names = classes + [f"array_{a}" for a in array_classes]
    all_names = set(names)
    for c in classes:
        for fname, *_ in SCHEMA[c].fields:
            all_names.add(fname if fname != "$base" else "$")
    all_names.add("member")
    type_names = b"".join(n.encode() + b"\x00" for n in sorted(all_names))
    type_names += b"\x00" * ((-len(type_names)) % 4)

    class_index = {n: i for i, n in enumerate(names)}
    fields_out = []  # (name, kind, classRef, offset)
    class_out = []  # (name, fieldIndex, fieldCount, align, kind, size)
    for c in classes:
        cd = SCHEMA[c]
        fi = len(fields_out)
        for fname, kind, ref, off in cd.fields:
            if kind == T_INHERITED:
                fields_out.append(("$", T_INHERITED, class_index[ref], 0))
            elif kind == T_STRUCT:
                fields_out.append((fname, T_STRUCT, class_index[ref], off))
            elif kind == T_ARRAY:
                fields_out.append((fname, T_ARRAY, class_index[f"array_{ref}"], off))
            else:
                fields_out.append((fname, kind, 0, off))
        class_out.append((c, fi, len(cd.fields), cd.alignment, cd.kind, cd.size))
    for a in array_classes:
        fi = len(fields_out)
        if a in _PRIM_ARRAY:
            fields_out.append(("member", _PRIM_ARRAY[a], 0, 0))
        else:
            fields_out.append(("member", T_STRUCT, class_index[a], 0))
        class_out.append((f"array_{a}", fi, 1, 4, T_ARRAY, 4))

    strings = bytearray()
    string_off: dict[str, int] = {}

    def s_off(text: str) -> int:
        if text not in string_off:
            string_off[text] = len(strings)
            strings.extend(text.encode() + b"\x00")
        return string_off[text]

    arrays = bytearray()
    array_table = []  # (offset, count, classRef)
    import_index = {imp: i for i, imp in enumerate(imports)}
    obj_index = {id(o): i + 1 for i, o in enumerate(objects)}

    def pack_value(kind, ref, value) -> bytes:
        if kind == T_CSTRING:
            return struct.pack("<I", s_off(value or ""))
        if kind == T_F32:
            return struct.pack("<f", value)
        if kind == T_I32:
            return struct.pack("<i", value)
        if kind == T_RESREF:
            return struct.pack("<Q", value)
        if kind == T_POINTER:
            if value is None:
                return struct.pack("<I", 0)
            if isinstance(value, tuple):
                return struct.pack("<I", 0x80000000 | import_index[value])
            return struct.pack("<I", obj_index[id(value)])
        if kind == T_STRUCT:
            return pack_class(ref, value)
        if kind == T_ARRAY:
            elem_kind = _PRIM_ARRAY.get(ref, T_STRUCT)
            # pack elements first: nested arrays append their own payloads, which must not
            # land between this array's elements
            items = [pack_value(elem_kind, ref, item) for item in value]
            if elem_kind == T_STRUCT:
                al = SCHEMA[ref].alignment
                arrays.extend(b"\x00" * ((-len(arrays)) % al))
            off = len(arrays)
            for item in items:
                arrays.extend(item)
            arrays.extend(b"\x00" * ((-len(arrays)) % 4))
            array_table.append((off, len(value), class_index[f"array_{ref}"]))
            return struct.pack("<i", len(array_table) - 1)
        raise ValueError(kind)

    def pack_class(cname, values: dict, base_offset: int = 0) -> bytes:
        cd = SCHEMA[cname]
        buf = bytearray(cd.size)
        for fname, kind, ref, off in cd.fields:
            if kind == T_INHERITED:
                sub = pack_class(ref, values)
                buf[:len(sub)] = bytes(a | b for a, b in zip(sub, buf[:len(sub)]))
                continue
            if fname not in values:
                continue
            raw = pack_value(kind, ref, values[fname])
            buf[off - base_offset:off - base_offset + len(raw)] = raw
        return bytes(buf)

    data = bytearray()
    instances = []
    for o in objects:
        cd = SCHEMA[o.cls]
        data.extend(b"\x00" * ((-len(data)) % cd.alignment))
        data.extend(guid_to_bytes(o.guid))
        body = pack_class(o.cls, o.fields)
        # alignment-4 classes have no 8-byte header: field offsets are relative to (guid end - 8)
        data.extend(body[8:] if cd.alignment == 4 else body)
        instances.append((class_index[o.cls], 1))
    data.extend(b"\x00" * ((-len(data)) % 16))
    strings.extend(b"\x00" * ((-len(strings)) % 16))

    header = bytearray()
    n_imports = len(imports)
    head = struct.Struct("<IIIIHHHHHHIII")
    header_len = head.size + 16 + 8 + n_imports * 32 + len(type_names) + len(fields_out) * 16 + len(class_out) * 16 \
        + len(instances) * 4
    header_len += (-header_len) % 16
    header_len += len(array_table) * 12
    header_len += (-header_len) % 16
    strings_offset = header_len
    header += head.pack(EBX_V4, strings_offset, len(strings) + len(data), n_imports, len(instances), len(instances),
                        len(classes), len(class_out), len(fields_out), len(type_names), len(strings),
                        len(array_table), len(data))
    header += guid_to_bytes(partition_guid)
    header += struct.pack("<II", 0, 0)
    for p, i in imports:
        header += guid_to_bytes(p) + guid_to_bytes(i)
    header += type_names
    for fname, kind, cref, off in fields_out:
        header += struct.pack("<IHHII", name_hash(fname), (kind << 4) << 1, cref, off, 0)
    for cname, fi, fc, al, kind, size in class_out:
        header += struct.pack("<IiBBHHH", name_hash(cname), fi, fc, al, (kind << 4) << 1, size, 0)
    for cref, count in instances:
        header += struct.pack("<HH", cref, count)
    header += b"\x00" * ((-len(header)) % 16)
    for off, count, cref in array_table:
        header += struct.pack("<IIi", off, count, cref)
    header += b"\x00" * ((-len(header)) % 16)
    assert len(header) == strings_offset, (len(header), strings_offset)
    return bytes(header + strings + data + arrays)


# -- container writer ------------------------------------------------------------------

@dataclass
class ModEntry:
    type: str
    name: str
    data: bytes | None = None
    compress_payload: bool = True
    res_type: int = 0
    res_rid: int = 0
    res_meta: bytes = b""
    original_size: int | None = None


def build_fbmod(entries: list[ModEntry], version: int = 5, title: str = "Synthetic", profile: str = "StarWarsBattlefrontII") -> bytes:
    payloads: list[bytes] = []
    table = bytearray()
    type_ids = {"embedded": 0, "ebx": 1, "res": 2, "chunk": 3, "bundle": 4}
    table += struct.pack("<i", len(entries))
    for e in entries:
        stored = None
        if e.data is not None:
            stored = compress(e.data) if e.compress_payload else e.data
            payloads.append(stored)
        idx = len(payloads) - 1 if stored is not None else -1
        table += struct.pack("<Bi", type_ids[e.type], idx)
        if version > 3 or idx != -1:
            table += e.name.encode() + b"\x00"
        if idx != -1:
            table += hashlib.sha1(stored).digest()
            table += struct.pack("<qBi", e.original_size if e.original_size is not None else len(e.data), 0, 0)
            if version >= 3:
                table += b"\x00"
        if version <= 3:
            if idx != -1:
                table += struct.pack("<ii", 0, 0)
        else:
            table += struct.pack("<i", 0)
        if e.type == "res":
            table += struct.pack("<IQi", e.res_type, e.res_rid, len(e.res_meta)) + e.res_meta
        elif e.type == "chunk":
            table += struct.pack("<IIIIii", 0, 0, 0, len(e.data or b""), 0, -1)
        elif e.type == "bundle":
            table += e.name.encode() + b"\x00" + struct.pack("<i", 0)
    details = [title, "fbmod-extractor", "Test", "1.0", "synthetic mod for automated tests"]
    if version >= 5:
        details.append("")
    head = bytearray()
    head += struct.pack("<QIqi", MAGIC, version, 0, len(payloads))
    head += bytes((len(profile),)) + profile.encode()
    head += struct.pack("<i", 1234)
    head += b"".join(d.encode() + b"\x00" for d in details)
    body = head + table
    data_offset = len(body)
    struct.pack_into("<q", body, 12, data_offset)
    index = bytearray()
    blob = bytearray()
    for p in payloads:
        index += struct.pack("<qq", len(blob), len(p))
        blob += p
    return bytes(body + index + blob)


# -- full synthetic hero ------------------------------------------------------------------

def synthetic_hero_mod(version: int = 5, with_mesh_ebx: bool = True, drop_chunk: bool = False,
                       with_skeleton: bool = True) -> tuple[bytes, dict]:
    """A skinned 'body' (2 sections, 2 LODs), a rigid 'head', textures and a skeleton."""
    truth: dict = {}
    body = make_capsule(radius=0.25, height=1.2, y0=0.0, bones=(0, 1, 2))
    belt = make_capsule(radius=0.28, height=0.2, rings=4, segs=12, y0=0.55, bones=(1, 1, 1))
    body_lo = make_capsule(radius=0.25, height=1.2, rings=6, segs=8, y0=0.0, bones=(0, 1, 2))
    head = make_capsule(radius=0.12, height=0.3, rings=8, segs=12, y0=1.3, bones=(2, 2, 2))
    head.joints = head.weights = None

    body_lods = [
        SynthLod([SynthSection("body_mat", 0, body, DECL_SKINNED, STRIDES_SKINNED),
                  SynthSection("belt_mat", 1, belt, DECL_SKINNED, STRIDES_SKINNED)]),
        SynthLod([SynthSection("body_mat", 0, body_lo, DECL_SKINNED, STRIDES_SKINNED)], index_size=4),
    ]
    body_name = "characters/heroes/synth/synth_body_mesh"
    ms_body, meta_body, chunks_body = build_meshset(body_name, 1, body_lods,
                                                    ((-0.3, 0.0, -0.3), (0.3, 1.3, 0.3)), bone_count=3)
    head_lods = [SynthLod([SynthSection("head_mat", 0, head, DECL_RIGID, STRIDES_RIGID)], inline=True)]
    head_name = "characters/heroes/synth/synth_head_mesh"
    ms_head, meta_head, chunks_head = build_meshset(head_name, 0, head_lods,
                                                    ((-0.15, 1.25, -0.15), (0.15, 1.65, 0.15)))
    truth["body"] = {"lod0_vertices": len(body.positions) + len(belt.positions),
                     "lod0_triangles": len(body.indices) + len(belt.indices),
                     "lod1_vertices": len(body_lo.positions), "lod1_triangles": len(body_lo.indices),
                     "positions": [body.positions, belt.positions], "normals": [body.normals, belt.normals],
                     "uvs": [body.uvs, belt.uvs], "indices": [body.indices, belt.indices]}
    truth["head"] = {"lod0_vertices": len(head.positions), "lod0_triangles": len(head.indices),
                     "positions": [head.positions]}

    rid_body, rid_head = 0x1111000000000001, 0x1111000000000002
    rid_cs, rid_nm = 0x2222000000000001, 0x2222000000000002
    cs_img, nm_img = make_texture_image("cs"), make_texture_image("nm")
    cs_data, nm_data = encode_texture(cs_img, "BC3_UNORM"), encode_texture(nm_img, "R8G8B8A8_UNORM")
    cs_chunk, nm_chunk = str(uuid.uuid4()), str(uuid.uuid4())
    tex_cs = build_texture("BC3_UNORM", 64, 64, cs_data, cs_chunk, srgb=True)
    tex_nm = build_texture("R8G8B8A8_UNORM", 64, 64, nm_data, nm_chunk)
    truth["textures"] = {"synth_body_cs": cs_img, "synth_body_nm": nm_img}

    cs_part, nm_part = str(uuid.uuid4()), str(uuid.uuid4())
    cs_asset = EbxObj("TextureAsset", {"Name": "characters/heroes/synth/synth_body_cs", "Resource": rid_cs})
    nm_asset = EbxObj("TextureAsset", {"Name": "characters/heroes/synth/synth_body_nm", "Resource": rid_nm})
    ebx_cs = build_ebx([cs_asset], [], cs_part)
    ebx_nm = build_ebx([nm_asset], [], nm_part)

    shader_import = (str(uuid.uuid4()), str(uuid.uuid4()))
    imports = [shader_import, (cs_part, cs_asset.guid), (nm_part, nm_asset.guid)]
    mats = [EbxObj("MeshMaterial", {"Shader": {"Shader": shader_import, "TextureParameters": [
        {"ParameterName": "_CS", "Value": (cs_part, cs_asset.guid)},
        {"ParameterName": "_NM", "Value": (nm_part, nm_asset.guid)}]}}),
        EbxObj("MeshMaterial", {"Shader": {"Shader": shader_import, "TextureParameters": [
            {"ParameterName": "_CS", "Value": (cs_part, cs_asset.guid)}]}})]
    body_asset = EbxObj("SkinnedMeshAsset", {"Name": body_name, "Materials": mats, "MeshSetResource": rid_body})
    ebx_body = build_ebx([body_asset] + mats, imports, str(uuid.uuid4()))

    def lt(t):
        return {"right": {"x": 1, "y": 0, "z": 0}, "up": {"x": 0, "y": 1, "z": 0},
                "forward": {"x": 0, "y": 0, "z": 1}, "trans": {"x": t[0], "y": t[1], "z": t[2]}}
    skel = EbxObj("SkeletonAsset", {"Name": "animations/synth/synth_skeleton", "BoneNames": ["Hips", "Spine", "Head"],
                                    "Hierarchy": [-1, 0, 1],
                                    "LocalPose": [lt((0, 0.3, 0)), lt((0, 0.4, 0)), lt((0, 0.5, 0))]})
    ebx_skel = build_ebx([skel], [], str(uuid.uuid4()))

    icon = io.BytesIO()
    Image.new("RGB", (8, 8), (40, 80, 160)).save(icon, "PNG")
    entries = [ModEntry("embedded", "Icon", icon.getvalue(), compress_payload=False)]
    entries += [ModEntry("embedded", f"Screenshot{i}", None) for i in range(4)]
    if with_mesh_ebx:
        entries.append(ModEntry("ebx", body_name, ebx_body))
    entries += [ModEntry("ebx", "characters/heroes/synth/synth_body_cs", ebx_cs),
                ModEntry("ebx", "characters/heroes/synth/synth_body_nm", ebx_nm),
                ModEntry("ebx", "characters/heroes/synth/unmodified_reference", None)]
    if with_skeleton:
        entries.append(ModEntry("ebx", "animations/synth/synth_skeleton", ebx_skel))
    truth["skeleton_ebx"] = ebx_skel
    entries += [ModEntry("res", body_name, ms_body, res_type=RES_TYPE_MESHSET, res_rid=rid_body, res_meta=meta_body),
                ModEntry("res", head_name, ms_head, res_type=RES_TYPE_MESHSET, res_rid=rid_head, res_meta=meta_head),
                ModEntry("res", "characters/heroes/synth/synth_body_cs", tex_cs, res_type=RES_TYPE_TEXTURE,
                         res_rid=rid_cs, res_meta=b"\x00" * 16),
                ModEntry("res", "characters/heroes/synth/synth_body_nm", tex_nm, res_type=RES_TYPE_TEXTURE,
                         res_rid=rid_nm, res_meta=b"\x00" * 16)]
    all_chunks = dict(chunks_body)
    all_chunks[cs_chunk] = cs_data
    all_chunks[nm_chunk] = nm_data
    dropped = None
    for guid, blob in all_chunks.items():
        if drop_chunk and guid == body_lods[0].chunk_id:
            dropped = guid
            continue
        entries.append(ModEntry("chunk", guid, blob))
    truth["dropped_chunk"] = dropped
    return build_fbmod(entries, version=version, title="Synthetic Hero"), truth
