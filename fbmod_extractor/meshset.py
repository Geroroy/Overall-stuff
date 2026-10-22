"""MeshSet resource (res type 0x49B156D4) parser for Star Wars Battlefront II.

Layout of the SWBF2 variant (offsets in the resource payload; all pointers are
64-bit offsets from the start of the payload)::

  MeshSet header
    AxisAlignedBox bbox (2 x Vec3, 16 bytes each)
    i64 lodOffsets[6], i64 unused
    i64 fullNameOffset, i64 nameOffset
    u32 nameHash, u32 meshType (0 rigid, 1 skinned, 2 composite)
    u16 lodFadeDistanceFactors[12], u32 flags
    u8 shaderDrawOrder, u8 shaderDrawOrderUserSlot, i16 shaderDrawOrderSubOrder
    u16 lodCount, u16 sectionCount
    skinned:   u16 boneCount, u16 bonePartCount, [i64 boneIndices, i64 boneBoxes]
    composite: u16 partCount, u16 boneCount,     [i64 partTransforms, i64 partBoxes]
    pad 16

  LOD
    u32 meshType, u32 maxInstances, i32 sectionCount, i64 sectionsOffset
    5 x (i32 count, i64 offset)  subset categories (opaque, transparent, decal, zonly, shadow)
    u32 layoutFlags, i32 indexBufferFormat (RenderFormat), u32 indexBufferSize,
    u32 vertexBufferSize, i32 adjacencySize, guid chunkId, u32 inlineDataOffset,
    i64 adjacencyOffset, i64 shaderDebugName, i64 name, i64 shortName, u32 nameHash, i64 unused,
    skinned: u32 boneCount, i64 boneIndexArray (u32 each)   composite: i64 partIndices
    pad 16

  Section (352 bytes)
    i64 runtime ptr x2, i64 materialNameOffset, i32 materialId, u32 lightmapUvIndex,
    u32 primitiveCount, u32 startIndex, u32 vertexOffset, u32 vertexCount,
    u8 vertexStride, u8 primitiveType, u8 bonesPerVertex(legacy), u8 boneCount(legacy),
    u32 x3 unused, u8 bonesPerVertex, u8 pad, u16 boneCount, i64 boneListOffset, u64 unused,
    2 x GeometryDeclarationDesc (16 x {u8 usage, u8 format, u8 offset, u8 stream},
                                 16 x {u8 stride, u8 classification}, u8 elementCount,
                                 u8 streamCount, u16 pad),
    f32 texCoordRatios[6], u8 unknown[44]

Vertex data for a LOD lives in its chunk (or inline after the header when the
chunk id is null): the vertex buffer first, then the index buffer at offset
``vertexBufferSize``. A section's vertices start at ``vertexOffset``; each
stream of the section is stored as a contiguous block of ``vertexCount``
vertices of that stream's stride.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .binary import NULL_GUID, FormatError, Reader
from .vertex import GeometryDeclaration, VertexElement

log = logging.getLogger(__name__)

RES_TYPE_MESHSET = 0x49B156D4
MESH_TYPES = {0: "rigid", 1: "skinned", 2: "composite"}
PRIMITIVE_TYPES = {0: "PointList", 1: "LineList", 2: "LineStrip", 3: "TriangleList", 5: "TriangleStrip",
                   7: "QuadList", 8: "XenonRectList", 9: "TrianglePatch"}
SUBSET_CATEGORIES = ["opaque", "transparent", "transparent_decal", "z_only", "shadow"]
MAX_LODS = 6
MAX_ELEMENTS = 16
MAX_STREAMS = 16  # SWBF2
SECTION_SIZE = 352


@dataclass
class MeshSection:
    index: int  # index within its LOD
    material_name: str
    material_id: int
    lightmap_uv_index: int
    primitive_count: int
    start_index: int
    vertex_offset: int
    vertex_count: int
    vertex_stride: int
    primitive_type: int
    bones_per_vertex: int
    bone_list: list[int]
    declarations: list[GeometryDeclaration]
    texcoord_ratios: list[float]

    @property
    def decl(self) -> GeometryDeclaration:
        return self.declarations[0]

    def describe(self) -> dict:
        return {
            "index": self.index,
            "material_name": self.material_name,
            "material_id": self.material_id,
            "primitive_type": PRIMITIVE_TYPES.get(self.primitive_type, self.primitive_type),
            "primitive_count": self.primitive_count,
            "start_index": self.start_index,
            "vertex_offset": self.vertex_offset,
            "vertex_count": self.vertex_count,
            "vertex_stride": self.vertex_stride,
            "bones_per_vertex": self.bones_per_vertex,
            "bone_list_count": len(self.bone_list),
            "vertex_declaration": self.decl.describe(),
        }


@dataclass
class MeshLod:
    index: int
    mesh_type: int
    max_instances: int
    sections: list[MeshSection]
    subset_categories: dict[str, list[int]]
    layout_flags: int
    index_buffer_format: int
    index_buffer_size: int
    vertex_buffer_size: int
    adjacency_buffer_size: int
    chunk_id: str
    inline_data_offset: int
    shader_debug_name: str
    name: str
    short_name: str
    name_hash: int
    bone_indices: list[int] = field(default_factory=list)
    inline_data: bytes | None = None

    @property
    def uses_chunk(self) -> bool:
        return self.chunk_id != NULL_GUID

    def section_categories(self, section_index: int) -> list[str]:
        return [c for c, idx in self.subset_categories.items() if section_index in idx]

    def describe(self) -> dict:
        return {
            "index": self.index,
            "mesh_type": MESH_TYPES.get(self.mesh_type, self.mesh_type),
            "name": self.name,
            "short_name": self.short_name,
            "chunk_id": self.chunk_id if self.uses_chunk else None,
            "inline_data_size": len(self.inline_data) if self.inline_data is not None else None,
            "layout_flags": f"0x{self.layout_flags:08x}",
            "index_buffer_format": self.index_buffer_format,
            "index_buffer_size": self.index_buffer_size,
            "vertex_buffer_size": self.vertex_buffer_size,
            "bone_count": len(self.bone_indices),
            "sections": [dict(s.describe(), categories=self.section_categories(s.index)) for s in self.sections],
        }


@dataclass
class MeshSet:
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    full_name: str
    name: str
    name_hash: int
    mesh_type: int
    flags: int
    lods: list[MeshLod]
    bone_count: int = 0
    bone_indices: list[int] = field(default_factory=list)
    part_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> dict:
        return {
            "full_name": self.full_name,
            "name": self.name,
            "mesh_type": MESH_TYPES.get(self.mesh_type, self.mesh_type),
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
            "flags": f"0x{self.flags:08x}",
            "lod_count": len(self.lods),
            "bone_count": self.bone_count,
            "bone_part_count": len(self.bone_indices),
            "lods": [lod.describe() for lod in self.lods],
            "warnings": self.warnings,
        }


def _vec3(r: Reader) -> tuple[float, float, float]:
    x, y, z, _pad = r.f32(), r.f32(), r.f32(), r.f32()
    return (x, y, z)


def _declaration(r: Reader) -> GeometryDeclaration:
    raw_elems = [(r.u8(), r.u8(), r.u8(), r.u8()) for _ in range(MAX_ELEMENTS)]
    streams = [(r.u8(), r.u8()) for _ in range(MAX_STREAMS)]
    element_count = r.u8()
    stream_count = r.u8()
    r.skip(2)
    elems = [VertexElement(u, f, o, s) for (u, f, o, s) in raw_elems[:min(element_count, MAX_ELEMENTS)]]
    return GeometryDeclaration(
        elements=elems,
        stream_strides=[s for s, _ in streams],
        stream_classifications=[c for _, c in streams],
        element_count=element_count,
        stream_count=stream_count,
    )


def _read_section(r: Reader, index: int) -> MeshSection:
    start = r.pos
    r.skip(16)  # two runtime pointers
    name_off = r.i64()
    material_id = r.i32()
    lightmap_uv = r.u32()
    prim_count, start_index, vertex_offset, vertex_count = r.u32(), r.u32(), r.u32(), r.u32()
    stride, prim_type, _bpv_legacy, _bc_legacy = r.u8(), r.u8(), r.u8(), r.u8()
    r.skip(12)
    bones_per_vertex = r.u8()
    r.skip(1)
    bone_count = r.u16()
    bone_list_off = r.i64()
    r.skip(8)
    decls = [_declaration(r), _declaration(r)]
    ratios = [r.f32() for _ in range(6)]
    r.skip(44)
    if r.pos - start != SECTION_SIZE:
        raise FormatError(f"section struct size {r.pos - start} != {SECTION_SIZE}")

    after = r.pos
    bones: list[int] = []
    if bone_count:
        r.seek(bone_list_off)
        bones = [r.u16() for _ in range(bone_count)]
    material_name = r.cstring_at(name_off) if name_off else ""
    r.seek(after)
    return MeshSection(index, material_name, material_id, lightmap_uv, prim_count, start_index, vertex_offset,
                       vertex_count, stride, prim_type, bones_per_vertex, bones, decls, ratios)


def _read_lod(r: Reader, index: int) -> MeshLod:
    mesh_type = r.u32()
    max_instances = r.u32()
    section_count = r.i32()
    sections_off = r.i64()
    categories: dict[str, list[int]] = {}
    for name in SUBSET_CATEGORIES:
        count, off = r.i32(), r.i64()
        save = r.pos
        if count:
            r.seek(off)
            categories[name] = [r.u8() for _ in range(count)]
        else:
            categories[name] = []
        r.seek(save)
    flags = r.u32()
    ib_format = r.i32()
    ib_size, vb_size = r.u32(), r.u32()
    adj_size = r.i32()
    chunk_id = r.guid()
    inline_off = r.u32()
    _adj_off = r.i64()
    s1, s2, s3 = r.i64(), r.i64(), r.i64()
    name_hash = r.u32()
    r.skip(8)
    bone_count = 0
    bones_off = 0
    if mesh_type == 1:
        bone_count = r.u32()
        bones_off = r.i64()
    elif mesh_type == 2:
        r.i64()  # part indices bitfield offset
    r.align(16)
    end = r.pos

    if section_count < 0 or section_count > 4096:
        raise FormatError(f"LOD {index}: implausible section count {section_count}")
    r.seek(sections_off)
    sections = [_read_section(r, i) for i in range(section_count)]
    bones: list[int] = []
    if bone_count and bones_off:
        r.seek(bones_off)
        bones = [r.u32() for _ in range(bone_count)]
    lod = MeshLod(
        index=index, mesh_type=mesh_type, max_instances=max_instances, sections=sections,
        subset_categories=categories, layout_flags=flags, index_buffer_format=ib_format,
        index_buffer_size=ib_size, vertex_buffer_size=vb_size, adjacency_buffer_size=adj_size,
        chunk_id=chunk_id, inline_data_offset=inline_off,
        shader_debug_name=r.cstring_at(s1) if s1 else "", name=r.cstring_at(s2) if s2 else "",
        short_name=r.cstring_at(s3) if s3 else "", name_hash=name_hash, bone_indices=bones,
    )
    r.seek(end)
    return lod


def parse_meshset(data: bytes, res_meta: bytes | None = None) -> MeshSet:
    r = Reader(data)
    bmin, bmax = _vec3(r), _vec3(r)
    lod_offsets = [r.i64() for _ in range(MAX_LODS)]
    r.skip(8)
    full_name_off, name_off = r.i64(), r.i64()
    name_hash = r.u32()
    mesh_type = r.u32()
    r.skip(MAX_LODS * 2 * 2)  # lod fade distance factors
    flags = r.u32()
    r.skip(4)  # shader draw order fields
    lod_count, section_count = r.u16(), r.u16()
    if mesh_type not in MESH_TYPES:
        raise FormatError(f"unknown mesh type {mesh_type}; not a SWBF2 MeshSet?")
    if not 1 <= lod_count <= MAX_LODS:
        raise FormatError(f"implausible LOD count {lod_count}")

    ms = MeshSet(bmin, bmax, "", "", name_hash, mesh_type, flags, [])
    if mesh_type == 1:
        ms.bone_count = r.u16()
        part_count = r.u16()
        if ms.bone_count or part_count:
            bi_off, _bbox_off = r.i64(), r.i64()
            if bi_off:
                save = r.pos
                r.seek(bi_off)
                ms.bone_indices = [r.u16() for _ in range(part_count)]
                r.seek(save)
    elif mesh_type == 2:
        ms.part_count = r.u16()
        ms.bone_count = r.u16()
        if ms.bone_count or ms.part_count:
            r.skip(16)
    r.align(16)

    for i in range(lod_count):
        if lod_offsets[i] != r.pos:
            ms.warnings.append(f"LOD {i} offset 0x{lod_offsets[i]:x} != expected 0x{r.pos:x}")
        r.seek(lod_offsets[i])
        ms.lods.append(_read_lod(r, i))
    total_sections = sum(len(lod.sections) for lod in ms.lods)
    if total_sections != section_count:
        ms.warnings.append(f"header section count {section_count} != parsed {total_sections}")

    ms.full_name = r.cstring_at(full_name_off) if full_name_off else ""
    ms.name = r.cstring_at(name_off) if name_off else ""

    # Inline vertex/index data (LODs whose chunk id is null) follow the header.
    if res_meta and len(res_meta) >= 8:
        inline_off = int.from_bytes(res_meta[0:4], "little")
        inline_size = int.from_bytes(res_meta[4:8], "little")
        if inline_off and inline_size:
            pos = inline_off
            for lod in ms.lods:
                if lod.uses_chunk:
                    continue
                n = lod.vertex_buffer_size + lod.index_buffer_size
                if pos + n > len(data):
                    ms.warnings.append(f"inline data for LOD {lod.index} overruns resource")
                    break
                lod.inline_data = data[pos:pos + n]
                pos += n
                pos += (-pos) % 16
    return ms
