"""Reader for Frosty Mod Manager's binary ``.fbmod`` container (format versions 1-5).

Layout (little-endian)::

    u64   magic 0x01005954534F5246 ("FROSTY\\x00\\x01")
    u32   format version (1..5)
    i64   data table offset
    i32   data table entry count
    u8+N  profile name (length-prefixed), e.g. "StarWarsBattlefrontII"
    i32   game head revision the mod was made against
    cstr  title, author, category, version, description, [link (v5+)]
    i32   resource count, then per resource:
            u8 type (0 embedded, 1 ebx, 2 res, 3 chunk, 4 bundle)
            i32 data index (-1 = no payload)
            cstr name                         (v<=3: only when index != -1)
            if index != -1:
                sha1[20], i64 original size, u8 flags, i32 handler hash,
                cstr user data (v3+)
            bundle lists (v<=3: two lists, only when index != -1; v4+: one list)
            res:   u32 resType, u64 resRid, i32 metaLen, meta[metaLen]
            chunk: u32 rangeStart, rangeEnd, logicalOffset, logicalSize, i32 h32, firstMip
            bundle: cstr bundle name, i32 superbundle hash
    data table at offset: entry_count x (i64 offset, i64 size); payloads follow,
    offsets relative to the end of the table.

Legacy ``FBMOD``/``FBMODV1``/``FBMODV2`` mods (a DbObject plus ``_NN.archive``
side files) predate this format and are detected but not decoded.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .binary import FormatError, Reader
from .compression import BlockInfo, UnsupportedCompression, decompress, looks_block_compressed

log = logging.getLogger(__name__)

MAGIC = 0x01005954534F5246
MAX_VERSION = 5
RESOURCE_TYPES = {0: "embedded", 1: "ebx", 2: "res", 3: "chunk", 4: "bundle"}

FLAG_INLINE = 0x01
FLAG_TOC_CHUNK = 0x02  # "add to chunks bundle"
FLAG_MANIFEST_CHUNK = 0x04
FLAG_ADDED = 0x08


class UnsupportedModFormat(Exception):
    pass


@dataclass
class ModResource:
    index: int  # position in the resource table
    type: str
    data_index: int
    name: str | None
    sha1: str | None = None
    original_size: int = 0
    flags: int = 0
    handler_hash: int = 0
    user_data: str = ""
    added_bundles: list[int] = field(default_factory=list)
    # res
    res_type: int | None = None
    res_rid: int | None = None
    res_meta: bytes | None = None
    # chunk
    range_start: int = 0
    range_end: int = 0
    logical_offset: int = 0
    logical_size: int = 0
    h32: int = 0
    first_mip: int = -1
    # bundle
    superbundle_hash: int | None = None

    @property
    def has_payload(self) -> bool:
        return self.data_index != -1

    @property
    def is_added(self) -> bool:
        return bool(self.flags & FLAG_ADDED)

    @property
    def is_inline(self) -> bool:
        return bool(self.flags & FLAG_INLINE)

    @property
    def has_custom_handler(self) -> bool:
        return self.handler_hash != 0

    def describe(self) -> dict:
        d = {
            "table_index": self.index,
            "type": self.type,
            "name": self.name,
            "has_payload": self.has_payload,
        }
        if self.has_payload:
            d.update(
                data_index=self.data_index,
                sha1=self.sha1,
                original_size=self.original_size,
                flags=self.flags,
                is_added=self.is_added,
                is_inline=self.is_inline,
                handler_hash=f"0x{self.handler_hash & 0xFFFFFFFF:08x}" if self.handler_hash else None,
                user_data=self.user_data or None,
            )
        if self.added_bundles:
            d["added_bundle_hashes"] = [f"0x{b & 0xFFFFFFFF:08x}" for b in self.added_bundles]
        if self.type == "res":
            d.update(
                res_type=f"0x{self.res_type:08X}",
                res_rid=f"0x{self.res_rid:016X}",
                res_meta=self.res_meta.hex() if self.res_meta else "",
            )
        elif self.type == "chunk":
            d.update(
                range_start=self.range_start,
                range_end=self.range_end,
                logical_offset=self.logical_offset,
                logical_size=self.logical_size,
                h32=f"0x{self.h32 & 0xFFFFFFFF:08x}",
                first_mip=self.first_mip,
            )
        elif self.type == "bundle":
            d["superbundle_hash"] = f"0x{(self.superbundle_hash or 0) & 0xFFFFFFFF:08x}"
        return d


@dataclass
class DecodedPayload:
    data: bytes
    stored_size: int
    compressed: bool
    blocks: list[BlockInfo]
    sha1_matches: bool | None


class FbMod:
    """A parsed .fbmod file. Payloads are decoded lazily."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.raw = self.path.read_bytes()
        self.file_sha1 = hashlib.sha1(self.raw).hexdigest()
        self.warnings: list[str] = []
        self._parse()

    # -- parsing ---------------------------------------------------------
    def _parse(self) -> None:
        r = Reader(self.raw)
        head = self.raw[:8]
        if head.startswith(b"FBMOD"):
            raise UnsupportedModFormat(
                "legacy Frosty mod format (FBMOD/FBMODV1/FBMODV2 with external .archive files); "
                "re-save the mod with Frosty Editor 1.0.5+ to convert it to the binary format"
            )
        if len(self.raw) < 32:
            raise FormatError("file too small to be an .fbmod")
        magic = r.u64()
        if magic != MAGIC:
            raise FormatError(f"bad magic 0x{magic:016x}; not a Frosty binary mod")
        self.version = r.u32()
        if not 1 <= self.version <= MAX_VERSION:
            raise UnsupportedModFormat(f"unsupported .fbmod format version {self.version}")
        self.data_offset = r.i64()
        self.data_count = r.i32()
        self.profile = r.sized_string(r.u8())
        self.game_version = r.i32()

        fields = ["title", "author", "category", "mod_version", "description"]
        if self.version >= 5:
            fields.append("link")
        self.details = {f: r.cstring() for f in fields}

        count = r.i32()
        if count < 0 or count > 1_000_000:
            raise FormatError(f"implausible resource count {count}")
        self.resources: list[ModResource] = [self._read_resource(r, i) for i in range(count)]
        self.table_end = r.pos

        if self.table_end != self.data_offset:
            self.warnings.append(
                f"resource table ends at 0x{self.table_end:x} but data table starts at 0x{self.data_offset:x}"
            )
        if self.data_offset + self.data_count * 16 > len(self.raw):
            raise FormatError("data table extends past end of file")

        self.data_entries: list[tuple[int, int]] = []
        base = self.data_offset + self.data_count * 16
        dr = Reader(self.raw, self.data_offset)
        for i in range(self.data_count):
            off, size = dr.i64(), dr.i64()
            if off < 0 or size < 0 or base + off + size > len(self.raw):
                raise FormatError(f"data entry {i} (offset {off}, size {size}) is outside the file")
            self.data_entries.append((base + off, size))

        for res in self.resources:
            if res.has_payload and not 0 <= res.data_index < self.data_count:
                raise FormatError(f"resource {res.index} ({res.name}) references missing data entry {res.data_index}")

    def _read_resource(self, r: Reader, i: int) -> ModResource:
        type_id = r.u8()
        if type_id not in RESOURCE_TYPES:
            raise FormatError(f"resource {i}: unknown type {type_id} at 0x{r.pos - 1:x}")
        res = ModResource(index=i, type=RESOURCE_TYPES[type_id], data_index=r.i32(), name=None)
        if self.version > 3 or res.data_index != -1:
            res.name = r.cstring()
        if res.data_index != -1:
            res.sha1 = r.sha1()
            res.original_size = r.i64()
            res.flags = r.u8()
            res.handler_hash = r.i32()
            if self.version >= 3:
                res.user_data = r.cstring()
        if self.version <= 3:
            if res.data_index != -1:
                for _ in range(r.i32()):  # bundles the asset already existed in (ignored by Frosty)
                    r.i32()
                res.added_bundles = [r.i32() for _ in range(r.i32())]
        else:
            res.added_bundles = [r.i32() for _ in range(r.i32())]

        if res.type == "res":
            res.res_type = r.u32()
            res.res_rid = r.u64()
            meta_len = r.i32()
            if meta_len < 0 or meta_len > 0x10000:
                raise FormatError(f"resource {i}: implausible res meta length {meta_len}")
            res.res_meta = r.bytes(meta_len)
        elif res.type == "chunk":
            res.range_start = r.u32()
            res.range_end = r.u32()
            res.logical_offset = r.u32()
            res.logical_size = r.u32()
            res.h32 = r.i32()
            res.first_mip = r.i32()
        elif res.type == "bundle":
            res.name = r.cstring()
            res.superbundle_hash = r.i32()
        return res

    # -- payload access --------------------------------------------------
    def stored_bytes(self, res: ModResource) -> bytes | None:
        if not res.has_payload:
            return None
        off, size = self.data_entries[res.data_index]
        return self.raw[off:off + size]

    def payload(self, res: ModResource) -> DecodedPayload | None:
        """Return the decompressed payload of a resource (None if it has none)."""
        stored = self.stored_bytes(res)
        if stored is None:
            return None
        # Frosty's resource SHA1 is the hash of the *stored* (compressed) bytes.
        sha1_ok = None
        if res.sha1 and res.type in ("ebx", "res", "chunk"):
            sha1_ok = hashlib.sha1(stored).hexdigest() == res.sha1
        if res.type in ("ebx", "res", "chunk") and looks_block_compressed(stored, res.original_size or None):
            data, blocks = decompress(stored, res.original_size or None)
            return DecodedPayload(data, len(stored), True, blocks, sha1_ok)
        if res.type in ("ebx", "res", "chunk") and looks_block_compressed(stored):
            # header sizes disagree with the declared original size: decode anyway, but warn
            data, blocks = decompress(stored)
            self.warnings.append(
                f"{res.type} '{res.name}': declared size {res.original_size} != decompressed {len(data)}"
            )
            return DecodedPayload(data, len(stored), True, blocks, sha1_ok)
        return DecodedPayload(stored, len(stored), False, [], sha1_ok)

    def resources_of(self, kind: str) -> list[ModResource]:
        return [r for r in self.resources if r.type == kind]

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        payloads: dict[str, int] = {}
        for r in self.resources:
            counts[r.type] = counts.get(r.type, 0) + 1
            if r.has_payload:
                payloads[r.type] = payloads.get(r.type, 0) + 1
        return {
            "file": self.path.name,
            "file_size": len(self.raw),
            "file_sha1": self.file_sha1,
            "format_version": self.version,
            "profile": self.profile,
            "game_head_revision": self.game_version,
            "details": self.details,
            "resource_counts": counts,
            "resources_with_payload": payloads,
            "data_entries": self.data_count,
        }


__all__ = ["FbMod", "ModResource", "DecodedPayload", "UnsupportedModFormat", "UnsupportedCompression"]
