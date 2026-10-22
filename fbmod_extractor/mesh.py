"""Decode MeshSet sections into vertex/index arrays and validate them."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .meshset import MeshLod, MeshSection, MeshSet
from .sdk_tables import load_render_formats
from .vertex import (FORMATS, USAGE, decode_element, unpack_tangent_space_axis_angle,
                     unpack_tangent_space_quaternion)

log = logging.getLogger(__name__)

TRIANGLE_LIST = 3
# A normal is considered consistent with the surface if it is within ~60 degrees
# of the area-weighted average of its adjacent face normals.
NORMAL_AGREEMENT_DOT = 0.5
NORMAL_AGREEMENT_REQUIRED = 0.9


class MeshDecodeError(Exception):
    pass


@dataclass
class DecodedSection:
    lod_index: int
    section_index: int
    material_name: str
    material_id: int
    categories: list[str]
    positions: np.ndarray  # (N, 3) float32
    indices: np.ndarray  # (M, 3) uint32, section-local
    normals: np.ndarray | None = None
    tangents: np.ndarray | None = None  # (N, 4), w = bitangent sign (glTF convention)
    uvs: dict[str, np.ndarray] = field(default_factory=dict)
    colors: dict[str, np.ndarray] = field(default_factory=dict)
    joints: np.ndarray | None = None  # (N, K) uint16 raw bone indices
    weights: np.ndarray | None = None  # (N, K) float32
    ignored_elements: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"lod{self.lod_index}_s{self.section_index}_{self.material_name or 'unnamed'}"


def _index_size_candidates(lod: MeshLod) -> list[int]:
    formats = load_render_formats()
    name = formats.get(lod.index_buffer_format, "")
    declared = {"R16_UINT": 2, "R32_UINT": 4}.get(name)
    cands = [declared] if declared else []
    cands += [s for s in (2, 4) if s not in cands]
    return cands


def _read_indices(buf: bytes, lod: MeshLod, sec: MeshSection, size: int) -> np.ndarray | None:
    count = sec.primitive_count * 3
    start = lod.vertex_buffer_size + sec.start_index * size
    end = start + count * size
    if end > lod.vertex_buffer_size + lod.index_buffer_size or end > len(buf):
        return None
    dtype = "<u2" if size == 2 else "<u4"
    return np.frombuffer(buf, dtype=dtype, count=count, offset=start).astype(np.uint32)


def choose_index_size(buf: bytes, lod: MeshLod) -> tuple[int, str]:
    """Pick the index width: declared RenderFormat first, verified against every section."""
    tri_sections = [s for s in lod.sections if s.primitive_count and s.primitive_type == TRIANGLE_LIST]
    for size in _index_size_candidates(lod):
        ok = True
        for sec in tri_sections:
            idx = _read_indices(buf, lod, sec, size)
            if idx is None or (len(idx) and int(idx.max()) >= sec.vertex_count):
                ok = False
                break
        if ok:
            formats = load_render_formats()
            how = "declared" if formats.get(lod.index_buffer_format) in ("R16_UINT", "R32_UINT") else "inferred"
            return size, f"{how} ({formats.get(lod.index_buffer_format, lod.index_buffer_format)})"
    raise MeshDecodeError(
        f"LOD {lod.index}: no index width (2 or 4 bytes) yields in-range indices for all sections "
        f"(index format value {lod.index_buffer_format}, index buffer {lod.index_buffer_size} bytes)"
    )


def layouts_differ(sec: MeshSection) -> bool:
    """True when declared element offsets differ from sequential packing in declaration order."""
    seq: dict[int, int] = {}
    for elem in sec.decl.elements:
        if elem.usage == 0 or elem.format == 0:
            continue
        if elem.offset != seq.get(elem.stream, 0):
            return True
        seq[elem.stream] = seq.get(elem.stream, 0) + elem.size
    return False


def _element_views(buf: bytes, sec: MeshSection, warnings: list[str], layout: str = "declared"):
    """Yield (element, raw (N, size) uint8 view) for each declared element.

    ``declared`` uses each element's offset (what the GPU input layout uses);
    ``packed`` lays elements out back to back in declaration order (how Frosty's
    importer writes and its exporter reads vertex data).
    """
    decl = sec.decl
    n = sec.vertex_count
    base = sec.vertex_offset
    stream_base = {}
    for s, stride in enumerate(decl.stream_strides):
        if stride == 0:
            continue
        stream_base[s] = base
        base += stride * n
    seq_offsets: dict[int, int] = {}
    for elem in decl.elements:
        if elem.usage == 0 or elem.format == 0:
            continue
        if elem.format not in FORMATS:
            warnings.append(f"unknown vertex format 0x{elem.format:02x} for {elem.usage_name}")
            continue
        stride = decl.stream_strides[elem.stream] if elem.stream < len(decl.stream_strides) else 0
        if stride == 0 or elem.stream not in stream_base:
            warnings.append(f"element {elem.usage_name} references empty stream {elem.stream}")
            continue
        seq = seq_offsets.get(elem.stream, 0)
        seq_offsets[elem.stream] = seq + elem.size
        offset = elem.offset if layout == "declared" else seq
        if offset + elem.size > stride:
            warnings.append(f"element {elem.usage_name} extends past stream stride {stride}")
            continue
        start = stream_base[elem.stream]
        end = start + stride * n
        if end > len(buf):
            raise MeshDecodeError(
                f"section {sec.index} ({sec.material_name}): stream {elem.stream} needs bytes "
                f"0x{start:x}-0x{end:x} but vertex data is 0x{len(buf):x} bytes"
            )
        block = np.frombuffer(buf, dtype=np.uint8, count=stride * n, offset=start).reshape(n, stride)
        yield elem, block[:, offset:offset + elem.size]
    for s, used in seq_offsets.items():
        if used != decl.stream_strides[s]:
            warnings.append(f"stream {s}: elements cover {used} of {decl.stream_strides[s]} stride bytes")


def decode_section(buf: bytes, meshset: MeshSet, lod: MeshLod, sec: MeshSection, index_size: int) -> DecodedSection:
    """Decode a section; if declared offsets disagree with packed order, keep the layout that validates."""
    if not layouts_differ(sec):
        return _decode_section(buf, meshset, lod, sec, index_size, "declared")
    candidates = []
    for layout in ("declared", "packed"):
        try:
            # a wrong candidate layout decodes arbitrary bytes; validation rejects it below
            with np.errstate(invalid="ignore", over="ignore"):
                candidates.append(_decode_section(buf, meshset, lod, sec, index_size, layout))
        except MeshDecodeError:
            continue
    if not candidates:
        raise MeshDecodeError(f"section {sec.index}: neither declared nor packed vertex layout could be decoded")

    def score(ds: DecodedSection):
        c = ds.checks
        return (bool(c.get("positions_finite")), bool(c.get("within_meshset_bbox")), c.get("normal_agreement") or 0.0)

    best = max(candidates, key=score)
    best.warnings.append(f"vertex element offsets differ from packed order; '{best.checks['vertex_layout']}' layout "
                         f"chosen because it validates better ({[score(c) for c in candidates]})")
    return best


def _decode_section(buf: bytes, meshset: MeshSet, lod: MeshLod, sec: MeshSection, index_size: int,
                    layout: str) -> DecodedSection:
    if sec.primitive_type != TRIANGLE_LIST:
        raise MeshDecodeError(
            f"section {sec.index}: primitive type {sec.primitive_type} is not a triangle list; not decoded"
        )
    warnings: list[str] = []
    n = sec.vertex_count
    positions = None
    normals = tangent_vec = binormal_vec = None
    binormal_sign = None
    uvs: dict[str, np.ndarray] = {}
    colors: dict[str, np.ndarray] = {}
    joints: dict[int, np.ndarray] = {}
    weights: dict[int, np.ndarray] = {}
    ignored: list[str] = []

    for elem, raw in _element_views(buf, sec, warnings, layout):
        u = elem.usage
        name = elem.usage_name
        if u == USAGE["Pos"]:
            v = decode_element(raw, elem.format)
            positions = v[:, :3]
        elif u == USAGE["TangentSpace"]:
            if elem.format_name == "UInt":
                tangent_vec, binormal_vec, normals = unpack_tangent_space_quaternion(raw.copy().view("<u4")[:, 0])
            elif elem.format_name in ("UByte4N", "UShort4N"):
                tangent_vec, binormal_vec, normals = unpack_tangent_space_axis_angle(decode_element(raw, elem.format))
            else:
                ignored.append(f"{name}:{elem.format_name}")
        elif u == USAGE["Normal"]:
            normals = decode_element(raw, elem.format)[:, :3]
        elif u == USAGE["Tangent"]:
            tangent_vec = decode_element(raw, elem.format)[:, :3]
        elif u == USAGE["Binormal"]:
            binormal_vec = decode_element(raw, elem.format)[:, :3]
        elif u == USAGE["BinormalSign"]:
            v = decode_element(raw, elem.format)
            if v.shape[1] == 4:
                tangent_vec = v[:, :3]
                binormal_sign = v[:, 3]
            else:
                binormal_sign = v[:, 0]
        elif name.startswith("TexCoord"):
            v = decode_element(raw, elem.format)
            if v.shape[1] >= 2:
                uvs[name] = v[:, :2].astype(np.float32)
            else:
                ignored.append(f"{name}:{elem.format_name}")
        elif name in ("Color0", "Color1"):
            v = decode_element(raw, elem.format)
            if v.shape[1] == 4:
                colors[name] = np.clip(v, 0, 1).astype(np.float32)
            else:
                ignored.append(f"{name}:{elem.format_name}")
        elif name in ("BoneIndices", "BoneIndices2"):
            joints[0 if name == "BoneIndices" else 1] = decode_element(raw, elem.format).astype(np.uint16)
        elif name in ("BoneWeights", "BoneWeights2"):
            w = decode_element(raw, elem.format)
            if elem.format_name == "UByte4":
                w = w / 255.0
            weights[0 if name == "BoneWeights" else 1] = w.astype(np.float64)
        else:
            ignored.append(f"{name}:{elem.format_name}")

    if positions is None:
        raise MeshDecodeError(f"section {sec.index} ({sec.material_name}): no position element in declaration")

    indices = _read_indices(buf, lod, sec, index_size)
    if indices is None:
        raise MeshDecodeError(f"section {sec.index}: index range outside index buffer")
    tris = indices.reshape(-1, 3)

    ds = DecodedSection(
        lod_index=lod.index, section_index=sec.index, material_name=sec.material_name,
        material_id=sec.material_id, categories=lod.section_categories(sec.index),
        positions=positions.astype(np.float32), indices=tris, uvs=uvs, colors=colors,
        ignored_elements=ignored, warnings=warnings,
    )

    if normals is not None:
        ds.normals = _unit(normals).astype(np.float32)
    if tangent_vec is not None and ds.normals is not None:
        t = _unit(tangent_vec)
        if binormal_sign is not None:
            w = np.where(binormal_sign < 0, -1.0, 1.0)
        elif binormal_vec is not None:
            w = np.where(np.einsum("ij,ij->i", np.cross(ds.normals, t), binormal_vec) < 0, -1.0, 1.0)
        else:
            w = np.ones(n)
        ds.tangents = np.concatenate([t, w[:, None]], axis=1).astype(np.float32)

    if joints:
        j = np.concatenate([joints[k] for k in sorted(joints)], axis=1)
        if set(weights) == set(joints):
            wts = np.concatenate([weights[k] for k in sorted(weights)], axis=1)
            ds.joints, ds.weights = j, wts.astype(np.float32)
        else:
            warnings.append("bone indices present without matching bone weights; skin data not exported")

    ds.checks = validate_section(ds, meshset)
    ds.checks["vertex_layout"] = layout
    return ds


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return v / norm


def face_normals(pos: np.ndarray, tris: np.ndarray) -> np.ndarray:
    p = pos.astype(np.float64)
    return np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]])  # area-weighted


def validate_section(ds: DecodedSection, meshset: MeshSet) -> dict:
    pos = ds.positions.astype(np.float64)
    tris = ds.indices
    n = len(pos)
    checks: dict = {"vertex_count": n, "triangle_count": int(len(tris))}
    checks["positions_finite"] = bool(np.isfinite(pos).all())
    if n:
        bmin, bmax = pos.min(axis=0), pos.max(axis=0)
        checks["bbox_min"] = [round(float(x), 6) for x in bmin]
        checks["bbox_max"] = [round(float(x), 6) for x in bmax]
        hmin, hmax = np.array(meshset.bbox_min), np.array(meshset.bbox_max)
        tol = 1e-3 + 1e-3 * float(np.abs(np.concatenate([hmin, hmax])).max())
        checks["within_meshset_bbox"] = bool((bmin >= hmin - tol).all() and (bmax <= hmax + tol).all())
    checks["indices_in_range"] = bool(len(tris) == 0 or int(tris.max()) < n)
    if len(tris):
        degenerate = (tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2]) | (tris[:, 0] == tris[:, 2])
        checks["degenerate_triangles"] = int(degenerate.sum())
        used = np.zeros(n, dtype=bool)
        used[tris.reshape(-1)] = True
        checks["unreferenced_vertices"] = int((~used).sum())
    for name, uv in ds.uvs.items():
        checks[f"{name}_range"] = [[round(float(x), 4) for x in uv.min(axis=0)], [round(float(x), 4) for x in uv.max(axis=0)]]
        checks[f"{name}_finite"] = bool(np.isfinite(uv).all())
    if ds.weights is not None:
        sums = ds.weights.sum(axis=1)
        checks["weights_sum_min"] = round(float(sums.min()), 4)
        checks["weights_sum_max"] = round(float(sums.max()), 4)
        checks["max_bone_index"] = int(ds.joints[ds.weights > 0].max()) if (ds.weights > 0).any() else None

    if ds.normals is not None and len(tris) and checks["indices_in_range"]:
        fn = face_normals(pos, tris)
        acc = np.zeros((n, 3))
        for k in range(3):
            np.add.at(acc, tris[:, k], fn)
        ok = np.linalg.norm(acc, axis=1) > 0
        geo = _unit(acc[ok])
        dots = np.einsum("ij,ij->i", geo, ds.normals[ok].astype(np.float64))
        agree = float((dots > NORMAL_AGREEMENT_DOT).mean()) if len(dots) else 0.0
        flipped = float((dots < -NORMAL_AGREEMENT_DOT).mean()) if len(dots) else 0.0
        checks["normal_agreement"] = round(agree, 4)
        checks["normal_agreement_if_winding_flipped"] = round(flipped, 4)
    return checks
