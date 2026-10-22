"""Frostbite block compression (the framing used inside CAS files and .fbmod payloads).

A payload is a sequence of blocks::

    u32 BE  decompressed size   (top byte != 0 -> "use dictionary" flag, FIFA only)
    u8      compression type    (low 7 bits; bit 7 = obfuscation flag, FIFA19 only)
    u8      flags               (low nibble = bits 16..19 of the compressed size)
    u16 BE  compressed size     (bits 0..15)
    ...     compressed bytes

Compression types: 0x00 none, 0x02 zlib, 0x09 LZ4, 0x0F zstd, 0x11/0x15 Oodle.
Star Wars Battlefront II mods written by Frosty use zstd (code 0x0F70) without a
dictionary; a block that would grow under compression is stored with code 0x70.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import zstandard

from .binary import FormatError

COMPRESSION_NAMES = {0x00: "none", 0x02: "zlib", 0x09: "lz4", 0x0F: "zstd", 0x11: "oodle", 0x15: "oodle"}
MAX_BLOCK = 0x10000


class UnsupportedCompression(Exception):
    pass


@dataclass
class BlockInfo:
    offset: int
    decompressed_size: int
    compressed_size: int
    codec: str
    uses_dictionary: bool


def parse_block_headers(data: bytes) -> list[BlockInfo]:
    """Walk the block headers and verify they exactly tile ``data``.

    Raises FormatError if the buffer is not a well-formed block stream.
    """
    blocks: list[BlockInfo] = []
    pos = 0
    n = len(data)
    if n < 8:
        raise FormatError("too short for a compressed block stream")
    while pos < n:
        if pos + 8 > n:
            raise FormatError(f"truncated block header at 0x{pos:x}")
        dsize_raw, ctype, flags, csize = struct.unpack_from(">IBBH", data, pos)
        uses_dict = (dsize_raw & 0xFF000000) != 0
        dsize = dsize_raw & 0x00FFFFFF
        csize |= (flags & 0x0F) << 16
        code = ctype & 0x7F
        if code not in COMPRESSION_NAMES:
            raise FormatError(f"unknown compression type 0x{ctype:02x} at 0x{pos:x}")
        if (flags & 0xF0) != 0x70:
            raise FormatError(f"unexpected block flags 0x{flags:02x} at 0x{pos:x}")
        if dsize == 0 or dsize > 0x400000:
            raise FormatError(f"implausible block size {dsize} at 0x{pos:x}")
        if code == 0x00 and csize != dsize:
            raise FormatError(f"stored block size mismatch at 0x{pos:x}")
        end = pos + 8 + csize
        if end > n:
            raise FormatError(f"block at 0x{pos:x} overruns buffer")
        blocks.append(BlockInfo(pos, dsize, csize, COMPRESSION_NAMES[code], uses_dict))
        pos = end
    return blocks


def looks_block_compressed(data: bytes, expected_size: int | None = None) -> bool:
    try:
        blocks = parse_block_headers(data)
    except FormatError:
        return False
    if expected_size is not None and expected_size > 0:
        return sum(b.decompressed_size for b in blocks) == expected_size
    return True


def _decompress_lz4(payload: bytes, size: int) -> bytes:
    try:
        import lz4.block  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise UnsupportedCompression("LZ4 block found; install the optional 'lz4' package") from exc
    return lz4.block.decompress(payload, uncompressed_size=size)


def decompress(data: bytes, expected_size: int | None = None) -> tuple[bytes, list[BlockInfo]]:
    """Decompress a Frostbite block stream. Returns (payload, block list)."""
    blocks = parse_block_headers(data)
    out = bytearray()
    dctx = zstandard.ZstdDecompressor()
    for b in blocks:
        payload = data[b.offset + 8:b.offset + 8 + b.compressed_size]
        if b.codec == "none":
            chunk = payload
        elif b.codec == "zstd":
            if b.uses_dictionary:
                raise UnsupportedCompression(
                    "zstd block requires the game's ebx dictionary (FIFA-style); not used by SWBF2 mods"
                )
            chunk = dctx.decompress(payload, max_output_size=b.decompressed_size)
        elif b.codec == "zlib":
            chunk = zlib.decompress(payload)
        elif b.codec == "lz4":
            chunk = _decompress_lz4(payload, b.decompressed_size)
        else:
            raise UnsupportedCompression(f"{b.codec} compression (proprietary Oodle) at block 0x{b.offset:x}")
        if len(chunk) != b.decompressed_size:
            raise FormatError(
                f"block at 0x{b.offset:x} decompressed to {len(chunk)} bytes, header says {b.decompressed_size}"
            )
        out += chunk
    if expected_size is not None and expected_size > 0 and len(out) != expected_size:
        raise FormatError(f"decompressed {len(out)} bytes but resource declares {expected_size}")
    return bytes(out), blocks


def compress(data: bytes, codec: str = "zstd", level: int = 3) -> bytes:
    """Encode ``data`` with Frostbite block framing (used to build test fixtures)."""
    out = bytearray()
    cctx = zstandard.ZstdCompressor(level=level, write_content_size=False, write_checksum=False)
    for start in range(0, max(len(data), 1), MAX_BLOCK):
        block = data[start:start + MAX_BLOCK]
        if codec == "zstd":
            payload = cctx.compress(block)
            code = 0x0F70
            if len(payload) > len(block):
                payload, code = block, 0x0070
        elif codec == "none":
            payload, code = block, 0x0070
        elif codec == "zlib":
            payload, code = zlib.compress(block), 0x0270
        else:
            raise ValueError(codec)
        code |= (len(payload) >> 16) & 0x0F
        out += struct.pack(">IHH", len(block), code, len(payload) & 0xFFFF)
        out += payload
    return bytes(out)
