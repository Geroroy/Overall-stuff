"""Vertex declarations, element decoding and tangent-space unpacking for SWBF2 meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# VertexElementUsage (Frostbite)
USAGE_NAMES = {
    0x00: "Unknown", 0x01: "Pos", 0x02: "BoneIndices", 0x03: "BoneIndices2", 0x04: "BoneWeights",
    0x05: "BoneWeights2", 0x06: "Normal", 0x07: "Tangent", 0x08: "Binormal", 0x09: "BinormalSign",
    0x0A: "WorldTrans1", 0x0B: "WorldTrans2", 0x0C: "WorldTrans3", 0x0D: "InstanceId",
    0x0E: "InstanceUserData0", 0x0F: "PrevInstanceUserData0", 0x10: "InstanceUserData1",
    0x11: "PrevWorldTrans1", 0x12: "PrevWorldTrans2", 0x13: "PrevWorldTrans3", 0x14: "XenonIndex",
    0x15: "XenonBarycentric", 0x16: "XenonQuadID", 0x17: "Index", 0x18: "ViewIndex",
    0x1E: "Color0", 0x1F: "Color1",
    0x21: "TexCoord0", 0x22: "TexCoord1", 0x23: "TexCoord2", 0x24: "TexCoord3", 0x25: "TexCoord4",
    0x26: "TexCoord5", 0x27: "TexCoord6", 0x28: "TexCoord7", 0x29: "DisplacementMapTexCoord",
    0x2A: "RadiosityTexCoord", 0x2B: "VisInfo", 0x2C: "SpriteSize", 0x2D: "PackedTexCoord0",
    0x2E: "PackedTexCoord1", 0x2F: "PackedTexCoord2", 0x30: "PackedTexCoord3", 0x31: "ClipDistance0",
    0x32: "ClipDistance1", 0x33: "SubMaterialIndex", 0x34: "TangentSpace", 0x3C: "BranchInfo",
    0x3D: "PosAndScale", 0x3E: "Rotation", 0x3F: "SpriteSizeAndUv", 0x5A: "FadePos", 0x5B: "SpawnTime",
    0x64: "RegionIds", 0x65: "BlendWeights", 0x96: "PosAndSoftMul", 0x97: "Alpha", 0x98: "Misc0",
    0x99: "Misc1", 0x9A: "Misc2", 0x9B: "Misc3", 0xCD: "DeformationIndex", 0xCE: "DeformationWeight",
    0xCF: "DeformationPosition", 0xD0: "Delta", 0xD1: "ElementIndex", 0xD2: "Uv01", 0xD3: "WorldPos",
    0xE6: "Custom", 0xF0: "DestructionMaskDistance", 0xF1: "DestructionMaskTexCoord", 0xFA: "VertIndex",
}
USAGE = {v: k for k, v in USAGE_NAMES.items()}

# VertexElementFormat -> (name, numpy dtype, components, normalisation divisor or None)
FORMATS = {
    0x00: ("None", None, 0, None),
    0x01: ("Float", "<f4", 1, None), 0x02: ("Float2", "<f4", 2, None),
    0x03: ("Float3", "<f4", 3, None), 0x04: ("Float4", "<f4", 4, None),
    0x05: ("Half", "<f2", 1, None), 0x06: ("Half2", "<f2", 2, None),
    0x07: ("Half3", "<f2", 3, None), 0x08: ("Half4", "<f2", 4, None),
    0x32: ("UByteN", "u1", 1, 255.0),
    0x0A: ("Byte4", "i1", 4, None), 0x0B: ("Byte4N", "i1", 4, 127.0),
    0x0C: ("UByte4", "u1", 4, None), 0x0D: ("UByte4N", "u1", 4, 255.0),
    0x0E: ("Short", "<i2", 1, None), 0x0F: ("Short2", "<i2", 2, None),
    0x10: ("Short3", "<i2", 3, None), 0x11: ("Short4", "<i2", 4, None),
    0x12: ("ShortN", "<i2", 1, 32767.0), 0x13: ("Short2N", "<i2", 2, 32767.0),
    0x14: ("Short3N", "<i2", 3, 32767.0), 0x15: ("Short4N", "<i2", 4, 32767.0),
    0x16: ("UShort2", "<u2", 2, None), 0x17: ("UShort4", "<u2", 4, None),
    0x18: ("UShort2N", "<u2", 2, 65535.0), 0x19: ("UShort4N", "<u2", 4, 65535.0),
    0x1A: ("Int", "<i4", 1, None), 0x1B: ("Int2", "<i4", 2, None), 0x33: ("Int3", "<i4", 3, None),
    0x1C: ("Int4", "<i4", 4, None), 0x1D: ("IntN", "<i4", 1, 2147483647.0),
    0x1E: ("Int2N", "<i4", 2, 2147483647.0), 0x1F: ("Int4N", "<i4", 4, 2147483647.0),
    0x20: ("UInt", "<u4", 1, None), 0x21: ("UInt2", "<u4", 2, None), 0x34: ("UInt3", "<u4", 3, None),
    0x22: ("UInt4", "<u4", 4, None), 0x23: ("UIntN", "<u4", 1, 4294967295.0),
    0x24: ("UInt2N", "<u4", 2, 4294967295.0), 0x25: ("UInt4N", "<u4", 4, 4294967295.0),
    0x26: ("Comp3_10_10_10", "<u4", 1, None), 0x27: ("Comp3N_10_10_10", "<u4", 1, None),
    0x28: ("UComp3_10_10_10", "<u4", 1, None), 0x29: ("UComp3N_10_10_10", "<u4", 1, None),
    0x2A: ("Comp3_11_11_10", "<u4", 1, None), 0x2B: ("Comp3N_11_11_10", "<u4", 1, None),
    0x2C: ("UComp3_11_11_10", "<u4", 1, None), 0x2D: ("UComp3N_11_11_10", "<u4", 1, None),
    0x2E: ("Comp4_10_10_10_2", "<u4", 1, None), 0x2F: ("Comp4N_10_10_10_2", "<u4", 1, None),
    0x30: ("UComp4_10_10_10_2", "<u4", 1, None), 0x31: ("UComp4N_10_10_10_2", "<u4", 1, None),
}


def format_size(fmt: int) -> int:
    name, dtype, comps, _ = FORMATS.get(fmt, ("?", None, 0, None))
    if dtype is None:
        return 0
    return np.dtype(dtype).itemsize * comps


@dataclass
class VertexElement:
    usage: int
    format: int
    offset: int
    stream: int

    @property
    def usage_name(self) -> str:
        return USAGE_NAMES.get(self.usage, f"Usage_0x{self.usage:02X}")

    @property
    def format_name(self) -> str:
        return FORMATS.get(self.format, (f"Format_0x{self.format:02X}",))[0]

    @property
    def size(self) -> int:
        return format_size(self.format)

    def describe(self) -> dict:
        return {"usage": self.usage_name, "format": self.format_name, "offset": self.offset,
                "stream": self.stream, "size": self.size}


@dataclass
class GeometryDeclaration:
    elements: list[VertexElement]  # only the first ``element_count`` entries
    stream_strides: list[int]
    stream_classifications: list[int]
    element_count: int
    stream_count: int

    def describe(self) -> dict:
        return {
            "elements": [e.describe() for e in self.elements],
            "stream_strides": [s for s in self.stream_strides if s],
            "element_count": self.element_count,
            "stream_count": self.stream_count,
        }


def decode_element(raw: np.ndarray, fmt: int) -> np.ndarray:
    """Decode an (N, size) uint8 array of one element into (N, components) float/int values."""
    name, dtype, comps, norm = FORMATS[fmt]
    vals = raw.copy().view(np.dtype(dtype)).reshape(len(raw), comps)
    if norm is not None:
        return vals.astype(np.float64) / norm
    if np.dtype(dtype).kind == "f":
        return vals.astype(np.float64)
    return vals


# -- tangent space ---------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def _quat_columns(q: np.ndarray):
    """Columns R*x, R*y, R*z of the rotation matrix of unit quaternions q=(x,y,z,w)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    cx = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)
    cy = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=1)
    cz = np.cross(cx, cy)
    return cx, cy, cz


def unpack_tangent_space_quaternion(ts: np.ndarray):
    """Decode SWBF2's 32-bit packed tangent frame.

    Bit layout: [31..22] 10-bit component A, [21..13] 9-bit component C,
    [12..3] 10-bit component B, [2..1] index of the dropped (largest) component,
    [0] handedness. Stored components are q*sqrt(2)/... mapped from
    [-1/sqrt2, 1/sqrt2]. Returns (tangent, binormal, normal), each (N, 3).

    The frame matches the game's vertex shader: with R = rotation(q),
    tangent = -R*y, binormal = +/-(R*x cross R*y) (sign from bit 0), normal = R*x.
    """
    ts = ts.astype(np.uint32).reshape(-1)
    s = np.sqrt(2.0)
    a = (ts >> 22).astype(np.float64) * (s / 1023.0) - 1.0 / s
    b = ((ts >> 3) & 0x3FF).astype(np.float64) * (s / 1023.0) - 1.0 / s
    c = ((ts >> 13) & 0x1FF).astype(np.float64) * (s / 511.0) - 1.0 / s
    idx = (ts >> 1) & 3
    big = np.sqrt(np.clip(1.0 - (a * a + b * b + c * c), 0.0, 1.0))
    # The three stored components fill the quaternion slots in the order A, C, B,
    # skipping slot ``idx``, which receives the reconstructed largest component.
    stored = np.stack([a, c, b], axis=1)
    q = np.empty((len(ts), 4))
    for slot in range(4):
        sel = idx == slot
        if not sel.any():
            continue
        others = [k for k in range(4) if k != slot]
        q[np.ix_(sel, others)] = stored[sel]
        q[sel, slot] = big[sel]
    cx, cy, cz = _quat_columns(q)
    sign = np.where((ts & 1) == 1, 1.0, -1.0)[:, None]
    normal = _normalize(cx)
    tangent = _normalize(-cy)
    binormal = _normalize(cz * sign)
    return tangent, binormal, normal


def unpack_tangent_space_axis_angle(v: np.ndarray):
    """Decode the axis-angle tangent frame (UByte4N / UShort4N, values in [0,1]).

    Mirrors the game shader: w*255 holds flag bits selecting the reconstructed
    axis component and handedness, xy hold two axis components, z the angle.
    Returns (tangent, binormal, normal).
    """
    v = v.astype(np.float64)
    tw = v[:, 3] * 255.0 + 0.25
    tx = v[:, 0] * 0.707107 - 0.707107
    ty = v[:, 1] * 0.707107 - 0.707107
    tz = v[:, 2] * 1.5708
    flags = tw.astype(np.uint32)
    f16 = (flags & 16).astype(np.float64)
    f32 = (flags & 32).astype(np.float64)
    f192 = (flags & 192).astype(np.float64)
    f1 = (flags & 1) != 0
    ay = f16 * 0.0441942 + tx
    az = f32 * 0.0220971 + ty
    angle = f192 * 0.0245437 + tz
    ax = np.sqrt(np.abs(1.0 - (ay * ay + az * az)))
    ax = np.where(f1, -ax, ax)
    axis = np.stack([ax, ay, az], axis=1)
    sel4 = (flags & 4) != 0
    sel2 = (flags & 2) != 0
    sel8 = (flags & 8) != 0
    r1 = np.where(sel4[:, None], axis, axis[:, [1, 0, 2]])
    r1 = np.where(sel2[:, None], r1, r1[:, [1, 2, 0]])
    perp_a = np.stack([-r1[:, 1], r1[:, 0], np.zeros(len(r1))], axis=1)
    perp_b = np.stack([r1[:, 2], np.zeros(len(r1)), -r1[:, 0]], axis=1)
    p = _normalize(np.where(sel2[:, None], perp_a, perp_b))
    cr = np.cross(r1, p)
    p = p * np.cos(np.abs(angle))[:, None] + cr * np.sin(np.abs(angle))[:, None]
    bi = np.cross(r1, p)
    bi = np.where(sel8[:, None], -bi, bi)
    normal = _normalize(r1)
    binormal = _normalize(bi)
    tangent = _normalize(-p)
    return tangent, binormal, normal
