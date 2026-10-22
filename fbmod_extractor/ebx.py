"""EBX partition reader for Star Wars Battlefront II (magic 0x0FB4D1CE, "EBX v4").

SWBF2 partitions are self-describing: the header embeds the type names, field
descriptors and class descriptors used by the partition, so instance data can
be decoded without the game's type database.

Header (little-endian)::

    u32 magic, u32 stringsOffset, u32 stringsAndDataLen, u32 importCount,
    u16 instanceCount, u16 exportedCount, u16 uniqueClassCount, u16 classTypeCount,
    u16 fieldTypeCount, u16 typeNamesLen, u32 stringsLen, u32 arrayCount, u32 dataLen,
    guid partitionGuid, [v4: u32 boxedValueCount, u32 boxedValueOffset | v2: pad 16]
    imports[importCount]          : guid partition, guid instance
    typeNames[typeNamesLen]       : NUL-terminated names, looked up by djb2-xor hash
    fields[fieldTypeCount]        : i32 nameHash, u16 flags, u16 classRef, u32 dataOffset, u32 secondOffset
    classes[classTypeCount]       : i32 nameHash, i32 fieldIndex, u8 fieldCount, u8 alignment,
                                    u16 flags, u16 size, u16 secondSize
    instances[instanceCount]      : u16 classRef, u16 count            (pad 16)
    arrays[arrayCount]            : u32 offset, u32 count, i32 classRef (pad 16)
    boxedValues[boxedValueCount]  : u32 offset, u16 classRef, u16 type
    strings at stringsOffset, instance data at stringsOffset+stringsLen,
    array payloads at stringsOffset+stringsLen+dataLen.

Field/class kind = (flags >> 4) & 0x1F, where v4 flags are stored shifted left by 1.

Instance walk: objects of each instance record follow each other; each is
aligned to its class alignment, exported records prefix every object with its
16-byte instance GUID, classes with alignment != 4 then skip an 8-byte header.
Field data offsets are relative to ``position_after_guid`` (alignment != 4) or
``position_after_guid - 8`` (alignment 4). Pointers: 0 = null, bit 31 = import
index, otherwise a 1-based index into the partition's objects.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .binary import FormatError, Reader, guid_from_bytes

EBX_V2 = 0x0FB2D1CE
EBX_V4 = 0x0FB4D1CE

# EbxFieldType
T_INHERITED, T_DBOBJECT, T_STRUCT, T_POINTER, T_ARRAY = 0x00, 0x01, 0x02, 0x03, 0x04
T_STRING, T_CSTRING, T_ENUM, T_FILEREF, T_BOOL = 0x06, 0x07, 0x08, 0x09, 0x0A
T_I8, T_U8, T_I16, T_U16, T_I32, T_U32, T_U64, T_I64 = 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12
T_F32, T_F64, T_GUID, T_SHA1, T_RESREF, T_DELEGATE, T_TYPEREF, T_BOXED = 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A

_SCALARS = {
    T_BOOL: ("<B", 1), T_I8: ("<b", 1), T_U8: ("<B", 1), T_I16: ("<h", 2), T_U16: ("<H", 2),
    T_I32: ("<i", 4), T_U32: ("<I", 4), T_U64: ("<Q", 8), T_I64: ("<q", 8),
    T_F32: ("<f", 4), T_F64: ("<d", 8), T_RESREF: ("<Q", 8),
}
_FIXED_SIZES = {T_POINTER: 4, T_CSTRING: 4, T_ENUM: 4, T_GUID: 16, T_SHA1: 20, T_STRING: 32,
                T_FILEREF: 8, T_TYPEREF: 8, T_BOXED: 16}


def name_hash(name: str) -> int:
    h = 5381
    for ch in name.encode("latin-1", "replace"):
        h = ((h * 33) ^ ch) & 0xFFFFFFFF
    return h


@dataclass
class Import:
    partition_guid: str
    instance_guid: str


@dataclass
class FieldDesc:
    name: str
    kind: int
    class_ref: int
    data_offset: int


@dataclass
class ClassDesc:
    name: str
    field_index: int
    field_count: int
    alignment: int
    kind: int
    size: int


@dataclass
class EbxObject:
    index: int  # 1-based, as used by internal pointers
    type_name: str
    instance_guid: str | None
    offset: int
    fields: dict = field(default_factory=dict)


class EbxPartition:
    def __init__(self, data: bytes, name: str | None = None):
        self.name = name
        self.data = data
        self.errors: list[str] = []
        self.objects: list[EbxObject] = []
        self._parse_header()
        if self.self_describing:
            try:
                self._read_objects()
            except (FormatError, struct.error, IndexError) as exc:
                self.errors.append(f"instance decoding failed: {exc}")
        else:
            self.errors.append(
                "partition uses shared type descriptors (no embedded field layout); "
                "instance fields not decoded"
            )

    # -- header ------------------------------------------------------------
    def _parse_header(self) -> None:
        r = Reader(self.data)
        self.magic = r.u32()
        if self.magic not in (EBX_V2, EBX_V4):
            raise FormatError(f"not an EBX partition (magic 0x{self.magic:08x})")
        v4 = self.magic == EBX_V4
        self.strings_offset = r.u32()
        self.strings_and_data_len = r.u32()
        import_count = r.u32()
        instance_count = r.u16()
        exported_count = r.u16()
        self.unique_class_count = r.u16()
        class_count = r.u16()
        field_count = r.u16()
        type_names_len = r.u16()
        self.strings_len = r.u32()
        array_count = r.u32()
        self.data_len = r.u32()
        self.data_start = self.strings_offset + self.strings_len
        self.arrays_offset = self.data_start + self.data_len
        self.partition_guid = r.guid()
        if v4:
            self.boxed_count = r.u32()
            self.boxed_offset = r.u32() + self.data_start
        else:
            self.boxed_count = 0
            r.align(16)

        self.imports = [Import(r.guid(), r.guid()) for _ in range(import_count)]

        names: dict[int, str] = {}
        tn_end = r.pos + type_names_len
        while r.pos < tn_end:
            s = r.cstring()
            names.setdefault(name_hash(s), s)
        r.seek(tn_end)
        self.self_describing = field_count > 0 and type_names_len > 0

        def flags_kind(raw: int) -> int:
            return (((raw >> 1) if v4 else raw) >> 4) & 0x1F

        self.field_descs: list[FieldDesc] = []
        for _ in range(field_count):
            h = r.u32()
            flags, class_ref, data_off, _second = r.u16(), r.u16(), r.u32(), r.u32()
            self.field_descs.append(FieldDesc(names.get(h, f"field_{h:08x}"), flags_kind(flags), class_ref, data_off))
        self.class_descs: list[ClassDesc] = []
        for _ in range(class_count):
            h = r.u32()
            fidx, fcount, align, flags, size, _second = r.i32(), r.u8(), r.u8(), r.u16(), r.u16(), r.u16()
            self.class_descs.append(ClassDesc(names.get(h, f"class_{h:08x}"), fidx, fcount, align or 4, flags_kind(flags), size))

        self.instances: list[tuple[int, int, bool]] = []
        remaining = exported_count
        for _ in range(instance_count):
            class_ref, count = r.u16(), r.u16()
            self.instances.append((class_ref, count, remaining > 0))
            remaining = max(remaining - 1, 0)
        r.align(16)
        self.arrays = [(r.u32(), r.u32(), r.i32()) for _ in range(array_count)]
        r.align(16)
        self.boxed_values = [(r.u32(), r.u16(), r.u16()) for _ in range(self.boxed_count)]

        if self.self_describing:
            for class_ref, _, _ in self.instances:
                if class_ref >= len(self.class_descs):
                    raise FormatError(f"instance references class {class_ref} of {len(self.class_descs)}")
            self.root_type = self.class_descs[self.instances[0][0]].name if self.instances else None
        else:
            self.root_type = None

    # -- strings -----------------------------------------------------------
    def string_at(self, offset: int) -> str:
        if offset == 0xFFFFFFFF:
            return ""
        pos = self.strings_offset + offset
        if not 0 <= pos < len(self.data):
            raise FormatError(f"string offset 0x{offset:x} outside partition")
        end = self.data.find(b"\x00", pos)
        return self.data[pos:end if end >= 0 else len(self.data)].decode("utf-8", "replace")

    # -- instances ---------------------------------------------------------
    def _read_objects(self) -> None:
        pos = self.data_start
        index = 1
        for class_ref, count, exported in self.instances:
            cls = self.class_descs[class_ref]
            for _ in range(count):
                pos = _align(pos, cls.alignment)
                inst_guid = None
                if exported:
                    inst_guid = guid_from_bytes(self.data[pos:pos + 16])
                    pos += 16
                start = pos if cls.alignment != 4 else pos - 8
                obj = EbxObject(index, cls.name, inst_guid, start)
                obj.fields = self._read_class(cls, start, depth=0)
                self.objects.append(obj)
                index += 1
                pos = _align(start + cls.size, cls.alignment)
        self.walk_end = pos
        if pos > self.arrays_offset:
            raise FormatError(f"instance walk ended at 0x{pos:x}, past array region at 0x{self.arrays_offset:x}")

    def _read_class(self, cls: ClassDesc, start: int, depth: int) -> dict:
        if depth > 32:
            raise FormatError("class nesting too deep")
        out: dict = {}
        for i in range(cls.field_count):
            f = self.field_descs[cls.field_index + i]
            if f.kind == T_INHERITED:
                out.update(self._read_class(self.class_descs[f.class_ref], start, depth + 1))
            elif f.kind == T_ARRAY:
                out[f.name] = self._read_array(f, start + f.data_offset, depth)
            else:
                out[f.name] = self._read_value(f.kind, f.class_ref, start + f.data_offset, depth)
        return out

    def _read_array(self, f: FieldDesc, pos: int, depth: int) -> list:
        arr_index = struct.unpack_from("<i", self.data, pos)[0]
        if not 0 <= arr_index < len(self.arrays):
            raise FormatError(f"array index {arr_index} out of range in field {f.name}")
        arr_off, arr_count, _ = self.arrays[arr_index]
        arr_cls = self.class_descs[f.class_ref]
        elem = self.field_descs[arr_cls.field_index]
        p = self.arrays_offset + arr_off
        if elem.kind == T_STRUCT:
            scls = self.class_descs[elem.class_ref]
            stride = _align(scls.size, scls.alignment)
            p = _align(p, scls.alignment)
        elif elem.kind in _SCALARS:
            stride = _SCALARS[elem.kind][1]
        else:
            stride = _FIXED_SIZES.get(elem.kind, 4)
        if p + arr_count * stride > len(self.data):
            raise FormatError(f"array {f.name} ({arr_count} x {stride}) overruns partition")
        return [self._read_value(elem.kind, elem.class_ref, p + i * stride, depth) for i in range(arr_count)]

    def _read_value(self, kind: int, class_ref: int, pos: int, depth: int):
        d = self.data
        if kind in _SCALARS:
            fmt, _ = _SCALARS[kind]
            v = struct.unpack_from(fmt, d, pos)[0]
            return bool(v) if kind == T_BOOL else v
        if kind == T_ENUM:
            v = struct.unpack_from("<i", d, pos)[0]
            ecls = self.class_descs[class_ref] if class_ref < len(self.class_descs) else None
            if ecls:
                for i in range(ecls.field_count):
                    ef = self.field_descs[ecls.field_index + i]
                    if ef.data_offset == (v & 0xFFFFFFFF):
                        return ef.name
            return v
        if kind == T_CSTRING:
            return self.string_at(struct.unpack_from("<I", d, pos)[0])
        if kind == T_POINTER:
            v = struct.unpack_from("<I", d, pos)[0]
            if v == 0:
                return None
            if v >> 31:
                imp = self.imports[v & 0x7FFFFFFF]
                return {"$import": {"partition": imp.partition_guid, "instance": imp.instance_guid}}
            return {"$ref": v}
        if kind == T_GUID:
            return guid_from_bytes(d[pos:pos + 16])
        if kind == T_SHA1:
            return d[pos:pos + 20].hex()
        if kind == T_STRING:
            return d[pos:pos + 32].split(b"\x00", 1)[0].decode("utf-8", "replace")
        if kind in (T_FILEREF, T_TYPEREF):
            return self.string_at(struct.unpack_from("<I", d, pos)[0])
        if kind == T_BOXED:
            return {"$boxed_value": d[pos:pos + 16].hex()}
        if kind == T_STRUCT:
            scls = self.class_descs[class_ref]
            pos = _align(pos, scls.alignment)
            return self._read_class(scls, pos, depth + 1)
        return {"$unsupported_field_kind": kind}

    # -- helpers -----------------------------------------------------------
    @property
    def root(self) -> EbxObject | None:
        return self.objects[0] if self.objects else None

    def deref(self, value) -> EbxObject | None:
        if isinstance(value, dict) and "$ref" in value:
            i = value["$ref"] - 1
            if 0 <= i < len(self.objects):
                return self.objects[i]
        return None

    def asset_name(self) -> str | None:
        root = self.root
        if root and isinstance(root.fields.get("Name"), str):
            return root.fields["Name"]
        return None

    def class_chain(self, type_name: str) -> list[str]:
        """Names of a class and its bases, from the partition's own descriptors."""
        by_name = {c.name: c for c in self.class_descs}
        chain = []
        cur = by_name.get(type_name)
        while cur is not None and len(chain) < 32:
            chain.append(cur.name)
            parent = None
            for i in range(cur.field_count):
                f = self.field_descs[cur.field_index + i]
                if f.kind == T_INHERITED:
                    parent = self.class_descs[f.class_ref]
            cur = parent
        return chain

    def describe(self) -> dict:
        return {
            "magic": f"0x{self.magic:08X}",
            "partition_guid": self.partition_guid,
            "root_type": self.root_type,
            "root_type_chain": self.class_chain(self.root_type) if self.root_type else [],
            "asset_name": self.asset_name(),
            "instance_count": sum(c for _, c, _ in self.instances),
            "decoded_instance_count": len(self.objects),
            "imports": [{"partition": i.partition_guid, "instance": i.instance_guid} for i in self.imports],
            "errors": self.errors,
        }


def _align(v: int, a: int) -> int:
    return v if a <= 1 or v % a == 0 else v + (a - v % a)


def to_jsonable(value):
    """Convert decoded EBX values into JSON-safe structures."""
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53:
        return f"0x{value & 0xFFFFFFFFFFFFFFFF:016X}"
    return value
