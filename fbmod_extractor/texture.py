"""Texture resource (res type 0x6BDE20BA) parsing and conversion for SWBF2.

SWBF2 texture header (132 bytes)::

    u32 mipOffset0, u32 mipOffset1, u32 type (0 2D, 1 cube, 2 3D, 3 2D array),
    i32 pixelFormat (game RenderFormat enum), u32 unknown, u16 flags (0x2 = sRGB),
    u16 width, u16 height, u16 depth, u16 sliceCount, u8 mipCount, u8 firstMip,
    guid chunkId, u32 mipSizes[15], u32 chunkSize, u32 assetNameHash, char textureGroup[16]

The pixel data lives in the chunk ``chunkId``: mips in order, each mip holding
all slices (``offset(mip) = sum(mipSizes[:mip]) * slices``).
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .binary import NULL_GUID, FormatError, Reader
from .sdk_tables import load_render_formats

log = logging.getLogger(__name__)

RES_TYPE_TEXTURE = 0x6BDE20BA
TEXTURE_TYPES = {0: "2d", 1: "cube", 2: "3d", 3: "2d_array", 4: "1d_array", 5: "1d", 6: "cube_array"}
FLAG_SRGB = 0x2
HEADER_SIZE = 132

# RenderFormat name -> (DXGI format, bytes per 4x4 block or per pixel, is_block, Pillow decode spec)
FORMAT_INFO = {
    "BC1_UNORM": (71, 8, True, ("bcn", 1, "RGBA")), "BC1_SRGB": (72, 8, True, ("bcn", 1, "RGBA")),
    "BC1A_UNORM": (71, 8, True, ("bcn", 1, "RGBA")), "BC1A_SRGB": (72, 8, True, ("bcn", 1, "RGBA")),
    "BC2_UNORM": (74, 16, True, ("bcn", 2, "RGBA")), "BC2_SRGB": (75, 16, True, ("bcn", 2, "RGBA")),
    "BC3_UNORM": (77, 16, True, ("bcn", 3, "RGBA")), "BC3_SRGB": (78, 16, True, ("bcn", 3, "RGBA")),
    "BC4_UNORM": (80, 8, True, ("bcn", 4, "L")), "BC5_UNORM": (83, 16, True, ("bcn", 5, "RGB")),
    "BC6U_FLOAT": (95, 16, True, ("bcn", 6, "RGB")), "BC6S_FLOAT": (96, 16, True, ("bcn", 6, "RGB")),
    "BC7_UNORM": (98, 16, True, ("bcn", 7, "RGBA")), "BC7_SRGB": (99, 16, True, ("bcn", 7, "RGBA")),
    "R8G8B8A8_UNORM": (28, 4, False, ("raw", "RGBA", "RGBA")), "R8G8B8A8_SRGB": (29, 4, False, ("raw", "RGBA", "RGBA")),
    "B8G8R8A8_UNORM": (87, 4, False, ("raw", "BGRA", "RGBA")), "B8G8R8A8_SRGB": (91, 4, False, ("raw", "BGRA", "RGBA")),
    "R8_UNORM": (61, 1, False, ("raw", "L", "L")), "R8G8_UNORM": (49, 2, False, ("rg8", None, "RGB")),
    "R16_UNORM": (56, 2, False, ("r16", None, "I;16")),
    "R16G16B16A16_FLOAT": (10, 8, False, ("half4", None, "RGBA")),
    "R32G32B32A32_FLOAT": (2, 16, False, ("float4", None, "RGBA")),
}


@dataclass
class TextureHeader:
    mip_offsets: tuple[int, int]
    type: int
    pixel_format: int
    unknown: int
    flags: int
    width: int
    height: int
    depth: int
    slice_count: int
    mip_count: int
    first_mip: int
    chunk_id: str
    mip_sizes: list[int]
    chunk_size: int
    asset_name_hash: int
    texture_group: str
    format_name: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def is_srgb(self) -> bool:
        return bool(self.flags & FLAG_SRGB) or self.format_name.endswith("SRGB")

    @property
    def slices(self) -> int:
        if self.type == 1:
            return 6
        if self.type in (2, 3):
            return max(self.depth, 1)
        return max(self.slice_count, 1)

    def describe(self) -> dict:
        return {
            "type": TEXTURE_TYPES.get(self.type, self.type),
            "pixel_format_value": self.pixel_format,
            "pixel_format": self.format_name or None,
            "srgb": self.is_srgb,
            "flags": f"0x{self.flags:04x}",
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "slice_count": self.slice_count,
            "mip_count": self.mip_count,
            "first_mip": self.first_mip,
            "chunk_id": self.chunk_id,
            "chunk_size": self.chunk_size,
            "mip_sizes": self.mip_sizes[:self.mip_count],
            "texture_group": self.texture_group,
            "warnings": self.warnings,
        }


def parse_texture_header(data: bytes) -> TextureHeader:
    if len(data) < HEADER_SIZE:
        raise FormatError(f"texture header is {len(data)} bytes, expected {HEADER_SIZE}")
    r = Reader(data)
    h = TextureHeader(
        mip_offsets=(r.u32(), r.u32()), type=r.u32(), pixel_format=r.i32(), unknown=r.u32(), flags=r.u16(),
        width=r.u16(), height=r.u16(), depth=r.u16(), slice_count=r.u16(), mip_count=r.u8(), first_mip=r.u8(),
        chunk_id=r.guid(), mip_sizes=[r.u32() for _ in range(15)], chunk_size=r.u32(),
        asset_name_hash=r.u32(), texture_group=r.fixed_string(16),
    )
    if not (1 <= h.width <= 16384 and 1 <= h.height <= 16384 and 1 <= h.mip_count <= 15):
        raise FormatError(f"implausible texture header ({h.width}x{h.height}, {h.mip_count} mips)")
    h.format_name = load_render_formats().get(h.pixel_format, "")
    if not h.format_name:
        h.warnings.append(f"unknown RenderFormat value {h.pixel_format}")
    info = FORMAT_INFO.get(h.format_name)
    if info:
        expected = expected_mip_size(h.width, h.height, info[1], info[2])
        if h.mip_sizes[0] != expected:
            h.warnings.append(f"mip0 size {h.mip_sizes[0]} != {expected} expected for {h.format_name} {h.width}x{h.height}")
    return h


def expected_mip_size(w: int, h: int, unit: int, is_block: bool) -> int:
    if is_block:
        return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * unit
    return w * h * unit


@dataclass
class MipData:
    level: int
    width: int
    height: int
    data: bytes  # first slice


def split_mips(h: TextureHeader, chunk: bytes) -> tuple[list[MipData], str]:
    """Locate the mip levels present in a decompressed texture chunk."""
    slices = h.slices
    full = sum(h.mip_sizes[:h.mip_count]) * slices
    if len(chunk) >= full:
        start_mip = 0
        layout = "full mip chain"
    else:
        start_mip = None
        for m in range(1, h.mip_count):
            if len(chunk) >= sum(h.mip_sizes[m:h.mip_count]) * slices:
                start_mip = m
                break
        if start_mip is None:
            raise FormatError(f"chunk of {len(chunk)} bytes is smaller than the smallest mip set")
        layout = f"mip chain starting at mip {start_mip} (higher mips not in this chunk)"
    mips = []
    pos = 0
    for m in range(start_mip, h.mip_count):
        size = h.mip_sizes[m]
        mips.append(MipData(m, max(1, h.width >> m), max(1, h.height >> m), chunk[pos:pos + size]))
        pos += size * slices
    return mips, layout


def write_dds(path: Path, h: TextureHeader, mips: list[MipData]) -> None:
    """Write the original (still block-compressed) pixel data as a DX10 DDS file (first slice)."""
    info = FORMAT_INFO.get(h.format_name)
    if info is None:
        raise FormatError(f"no DDS mapping for {h.format_name or h.pixel_format}")
    dxgi, unit, is_block, _ = info
    top = mips[0]
    # DDSD_CAPS | HEIGHT | WIDTH | PIXELFORMAT | MIPMAPCOUNT | (LINEARSIZE or PITCH)
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | (0x80000 if is_block else 0x8)
    pitch = len(top.data) if is_block else top.width * unit
    caps = 0x1000 | ((0x8 | 0x400000) if len(mips) > 1 else 0)  # TEXTURE | COMPLEX | MIPMAP
    pixel_format = struct.pack("<II4sIIIII", 32, 0x4, b"DX10", 0, 0, 0, 0, 0)
    header = (
        b"DDS "
        + struct.pack("<7I", 124, flags, top.height, top.width, pitch, 0, len(mips))
        + b"\x00" * 44
        + pixel_format
        + struct.pack("<5I", caps, 0, 0, 0, 0)
    )
    dx10 = struct.pack("<5I", dxgi, 3, 0, 1, 0)  # TEXTURE2D, one array slice
    with open(path, "wb") as f:
        f.write(header)
        f.write(dx10)
        for m in mips:
            f.write(m.data)


def decode_to_image(h: TextureHeader, mip: MipData) -> Image.Image:
    info = FORMAT_INFO.get(h.format_name)
    if info is None:
        raise FormatError(f"no decoder for pixel format {h.format_name or h.pixel_format}")
    _, unit, is_block, (kind, arg, mode) = info
    w, hgt = mip.width, mip.height
    need = expected_mip_size(w, hgt, unit, is_block)
    if len(mip.data) < need:
        raise FormatError(f"mip {mip.level} has {len(mip.data)} bytes, needs {need}")
    data = mip.data[:need]
    if kind == "bcn":
        if arg == 6:
            args = (6, "BC6HS" if h.format_name == "BC6S_FLOAT" else "BC6H")
        elif arg == 5:
            args = (5, "BC5U")
        else:
            args = (arg,)
        # Pillow decodes whole blocks; pad odd sizes to the block grid then crop.
        bw, bh = max(4, (w + 3) // 4 * 4), max(4, (hgt + 3) // 4 * 4)
        img = Image.frombytes(mode, (bw, bh), data, "bcn", *args)
        return img.crop((0, 0, w, hgt)) if (bw, bh) != (w, hgt) else img
    if kind == "raw":
        return Image.frombytes(mode, (w, hgt), data, "raw", arg)
    if kind == "rg8":
        rg = np.frombuffer(data, np.uint8).reshape(hgt, w, 2)
        return Image.fromarray(np.dstack([rg, np.zeros((hgt, w), np.uint8)]), "RGB")
    if kind == "r16":
        return Image.fromarray(np.frombuffer(data, "<u2").reshape(hgt, w).astype(np.uint16))
    if kind in ("half4", "float4"):
        arr = np.frombuffer(data, "<f2" if kind == "half4" else "<f4").reshape(hgt, w, 4).astype(np.float64)
        return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")
    raise FormatError(f"unsupported decode kind {kind}")


def is_chunk_empty(h: TextureHeader) -> bool:
    return h.chunk_id == NULL_GUID
