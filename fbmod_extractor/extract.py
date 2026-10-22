"""End-to-end extraction: .fbmod -> manifest, raw payloads, textures, glTF meshes, report."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .binary import FormatError, NULL_GUID
from .compression import UnsupportedCompression, decompress, looks_block_compressed
from .ebx import EbxPartition, to_jsonable
from .fbmod import FbMod, ModResource
from .gltf import GltfMaterial, SkeletonData, bind_pose_world, convert_positions, write_glb
from .materials import MaterialInfo, infer_texture_role, mesh_materials, variation_entries
from .mesh import DecodedSection, MeshDecodeError, choose_index_size, decode_section
from .meshset import RES_TYPE_MESHSET, MeshSet, parse_meshset
from .restypes import res_type_name
from .texture import RES_TYPE_TEXTURE, TextureHeader, decode_to_image, parse_texture_header, split_mips, write_dds

log = logging.getLogger(__name__)

NORMAL_OK = 0.9

HOWTO_CHUNK = ("Frosty Editor (with the game loaded): View > Chunk/Res Editor, filter the Chunks list by this GUID, "
               "right-click > Export, keep the file name '<guid>.chunk', put it in a folder and re-run with "
               "--extra-dir <folder>")
HOWTO_TEXTURE = ("Frosty Editor: Data Explorer, open the texture asset named here (or, if only a partition GUID is "
                 "known, open the mesh asset and follow its material's TextureParameters), right-click > Export as "
                 "DDS using the asset name, and also Export as 'Binary File' (.bin) so the extractor can link the "
                 "GUID to the name; re-run with --extra-dir <folder>")
HOWTO_SKELETON = ("Frosty Editor: Data Explorer, open the character's SkeletonAsset (e.g. under animations/.../skeleton), "
                  "right-click > Export as 'Binary File' (.bin) and pass it with --skeleton <file.bin>")


@dataclass
class ExtractOptions:
    output: Path
    lods: str = "0"  # "0", "all" or comma separated LOD indices
    axis: str = "gltf"
    normals: str = "auto"  # auto | stored | none
    skeleton: Path | None = None
    write_raw: bool = True
    names: dict[str, str] = field(default_factory=dict)  # mesh-name substring -> output base name
    extra_dirs: list[Path] = field(default_factory=list)  # Frosty Editor exports supplying game data


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unnamed"


def _contained(base: Path, rel: str) -> Path:
    """Join an asset path from the mod under ``base`` without allowing it to escape (no '..', no absolute)."""
    parts = [_safe(p) for p in rel.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    return base.joinpath(*parts) if parts else base / "unnamed"


def _short(name: str | None) -> str:
    return (name or "").replace("\\", "/").rsplit("/", 1)[-1]


_ROLE_SUFFIX = re.compile(r"_(cs|nm|nam|aosl|ao|d|diff|diffuse|n|normal|mask|msk|op|em|mesh)$")


def _tex_stem(name: str) -> str:
    return _ROLE_SUFFIX.sub("", _short(name).lower())


def _shared_prefix_tokens(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.split("_"), b.split("_")):
        if x != y:
            break
        n += 1
    return n


def to_blender(p: np.ndarray, axis: str) -> np.ndarray:
    """Raw Frostbite coordinates -> Blender world coordinates after glTF import (glTF Y-up -> Blender Z-up)."""
    g = convert_positions(p, axis).astype(np.float64)
    return np.stack([g[:, 0], -g[:, 2], g[:, 1]], axis=1)


def build_expectations(name: str, decoded: list[list[DecodedSection]], materials: list[GltfMaterial],
                       sec_mat: dict, axis: str, skeleton: SkeletonData | None) -> dict:
    """What the Blender round trip must reproduce, computed from the decoded arrays."""
    lods = []
    for sections in decoded:
        used_pts, uv_used, sec_tris = [], [], []
        verts_used = tris_nd = 0
        mats_used = set()
        expected_textures: dict[str, int] = {}
        for ds in sections:
            tris = ds.indices
            nd = ~((tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2]) | (tris[:, 0] == tris[:, 2]))
            used = np.unique(tris.reshape(-1))
            verts_used += len(used)
            tris_nd += int(nd.sum())
            sec_tris.append(int(nd.sum()))
            used_pts.append(ds.positions[used])
            if ds.uvs:
                uv_used.append(ds.uvs[sorted(ds.uvs, key=lambda n: int(n[-1]))[0]][used])
            mi = sec_mat.get((ds.lod_index, ds.section_index))
            if mi is not None:
                mats_used.add(mi)
                m = materials[mi]
                expected_textures[m.name] = int(bool(m.base_color_uri)) + int(bool(m.normal_uri))
        pts = to_blender(np.concatenate(used_pts), axis)
        entry = {
            "object": f"{name}_lod{sections[0].lod_index}",
            "vertices_used": int(verts_used),
            "triangles_nondegenerate": int(tris_nd),
            "section_triangles": sec_tris,
            "distinct_materials": len(mats_used),
            "bbox_min": pts.min(axis=0).round(6).tolist(),
            "bbox_max": pts.max(axis=0).round(6).tolist(),
            "expected_textures": {k: v for k, v in expected_textures.items() if v},
        }
        if uv_used:
            uv = np.concatenate(uv_used).astype(np.float64)
            entry["uv_bounds"] = [[float(uv[:, 0].min()), float(1 - uv[:, 1].max())],
                                  [float(uv[:, 0].max()), float(1 - uv[:, 1].min())]]
        lods.append(entry)
    out = {"mesh": name, "axis": axis, "lods": lods}
    if skeleton is not None:
        world = bind_pose_world(skeleton, axis)
        heads = [[float(w[0, 3]), float(-w[2, 3]), float(w[1, 3])] for w in world]
        out["skeleton"] = {"names": skeleton.names, "heads": heads}
    return out


class Extraction:
    def __init__(self, mod_path: Path, opts: ExtractOptions):
        self.mod_path = Path(mod_path)
        self.opts = opts
        self.out = Path(opts.output)
        self.report: dict = {
            "tool": {"name": "fbmod-extractor", "version": __version__},
            "input": str(self.mod_path),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "options": {"lods": opts.lods, "axis": opts.axis, "normals": opts.normals,
                        "skeleton": str(opts.skeleton) if opts.skeleton else None},
            "errors": [],
            "warnings": [],
            "missing_dependencies": [],
            "meshes": [],
            "textures": [],
            "unsupported": [],
        }
        self.ebx: list[dict] = []
        self.res: list[dict] = []
        self.chunks: dict[str, dict] = {}
        self.texture_outputs: dict[str, dict] = {}  # texture res name (lower) -> outputs

    # ------------------------------------------------------------------
    def run(self, blender: str | None = None) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        log.info("reading %s", self.mod_path)
        self.mod = FbMod(self.mod_path)
        log.info("format v%d, profile %s, %d resources", self.mod.version, self.mod.profile, len(self.mod.resources))
        self.report["mod"] = self.mod.summary()
        self.report["warnings"] += self.mod.warnings
        if self.mod.profile != "StarWarsBattlefrontII":
            self.report["warnings"].append(
                f"mod profile is '{self.mod.profile}', this extractor targets StarWarsBattlefrontII; "
                "mesh and texture layouts may differ")
        self._load_payloads()
        self._load_extra()
        self._analyse_ebx()
        self._analyse_res()
        self._write_manifest()
        self._extract_textures()
        self._extract_meshes()
        if blender and any(m.get("gltf") for m in self.report["meshes"]):
            from .blender_export import export_fbx
            log.info("building FBX files with %s", blender)
            export_fbx(self.report, self.out, blender)
            self.report["fbx_export"] = "blender"
        elif any(m.get("gltf") for m in self.report["meshes"]):
            self.report["fbx_export"] = "skipped (Blender not available or disabled)"
            self.report["warnings"].append("FBX files were not built: Blender was not found or --no-fbx was given")
        self._finish()
        return self.report

    # ------------------------------------------------------------------
    def _load_payloads(self) -> None:
        raw_dir = self.out / "raw"
        for r in self.mod.resources:
            entry = {"res": r, "data": None, "error": None, "payload": None}
            if r.has_payload:
                try:
                    p = self.mod.payload(r)
                    entry["data"] = p.data
                    entry["payload"] = {
                        "stored_size": p.stored_size, "decoded_size": len(p.data), "compressed": p.compressed,
                        "codecs": sorted({b.codec for b in p.blocks}), "blocks": len(p.blocks),
                        "sha1_matches_stored_bytes": p.sha1_matches,
                    }
                except (FormatError, UnsupportedCompression, ValueError) as exc:
                    entry["error"] = f"payload decode failed: {exc}"
                    self.report["errors"].append(f"{r.type} '{r.name}': {entry['error']}")
            if r.type == "ebx":
                self.ebx.append(entry)
            elif r.type == "res":
                self.res.append(entry)
            elif r.type == "chunk":
                self.chunks[(r.name or "").lower()] = entry
            if self.opts.write_raw and entry["data"] is not None:
                if r.type == "embedded":
                    ext = ".png" if entry["data"][:4] == b"\x89PNG" else (
                        ".jpg" if entry["data"][:2] == b"\xff\xd8" else ".bin")
                    path = raw_dir / "embedded" / (_safe(r.name or f"embedded_{r.index}") + ext)
                elif r.type == "ebx":
                    path = _contained(raw_dir / "ebx", f"{r.name}.ebx")
                elif r.type == "res":
                    path = _contained(raw_dir / "res", f"{r.name}.{res_type_name(r.res_type)}.res")
                elif r.type == "chunk":
                    path = _contained(raw_dir / "chunks", f"{r.name}.chunk")
                else:
                    path = raw_dir / r.type / _safe(r.name or str(r.index))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(entry["data"])
                if r.type == "res" and r.res_meta:
                    path.with_suffix(".meta").write_bytes(r.res_meta)
                entry["raw_path"] = path.relative_to(self.out).as_posix()

    def _load_extra(self) -> None:
        """Load game data exported with Frosty Editor to fill dependencies the mod does not carry.

        * ``<guid>.chunk``  Chunk/Res Explorer chunk export (decompressed bytes)
        * ``<name>.res``    Chunk/Res Explorer res export (16-byte res meta + data)
        * ``<name>.bin``    Data Explorer "Binary File" export of an EBX asset
        * ``<asset>.dds|png|tga``  texture export, named after the texture asset
        """
        self.extra_images: dict[str, Path] = {}
        used = []
        guid_re = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
        for d in self.opts.extra_dirs:
            for path in sorted(Path(d).rglob("*")):
                if not path.is_file():
                    continue
                ext = path.suffix.lower()
                rec = {"file": str(path), "used_as": None}
                try:
                    if ext in (".dds", ".png", ".tga"):
                        self.extra_images[path.stem.lower()] = path
                        rec["used_as"] = "texture image (bound by asset name when referenced)"
                    elif ext not in (".chunk", ".res", ".bin", ".ebx"):
                        continue
                    else:
                        data = path.read_bytes()
                        if ext == ".chunk":
                            if not guid_re.match(path.stem):
                                rec["used_as"] = "ignored: chunk file name must be the chunk GUID"
                            elif (path.stem.lower() in self.chunks
                                  and self.chunks[path.stem.lower()]["data"] is not None):
                                rec["used_as"] = "ignored: the mod already carries this chunk"
                            else:
                                if looks_block_compressed(data):
                                    data, _ = decompress(data)
                                self.chunks[path.stem.lower()] = {"res": None, "data": data, "error": None,
                                                                  "source": str(path)}
                                rec["used_as"] = f"chunk {path.stem.lower()}"
                        elif ext in (".bin", ".ebx"):
                            if looks_block_compressed(data):
                                data, _ = decompress(data)
                            part = EbxPartition(data, name=path.stem)
                            name = (part.asset_name() or path.stem).lower()
                            if any((e["res"].name or "").lower() == name and e["data"] is not None for e in self.ebx):
                                rec["used_as"] = "ignored: the mod already carries this EBX"
                            else:
                                res = ModResource(index=-1, type="ebx", data_index=-2, name=name)
                                self.ebx.append({"res": res, "data": data, "error": None, "payload": None,
                                                 "source": str(path)})
                                rec["used_as"] = f"ebx {name} ({part.root_type})"
                        elif ext == ".res":
                            meta, body = data[:16], data[16:]
                            res_type = None
                            try:
                                parse_meshset(body, meta)
                                res_type = RES_TYPE_MESHSET
                            except (FormatError, IndexError, ValueError):
                                try:
                                    parse_texture_header(body)
                                    res_type = RES_TYPE_TEXTURE
                                except FormatError:
                                    pass
                            if res_type is None:
                                rec["used_as"] = "ignored: not a MeshSet or Texture resource"
                            else:
                                stem = path.stem.lower()
                                target = next((e for e in self.res if _short(e["res"].name).lower() == stem), None)
                                if target is not None and target["data"] is not None:
                                    rec["used_as"] = "ignored: the mod already carries this resource"
                                else:
                                    res = ModResource(index=-1, type="res", data_index=-2,
                                                      name=target["res"].name if target else stem,
                                                      res_type=res_type, res_rid=target["res"].res_rid if target else 0,
                                                      res_meta=meta)
                                    if target is not None:
                                        self.res.remove(target)
                                    self.res.append({"res": res, "data": body, "error": None, "payload": None,
                                                     "source": str(path)})
                                    rec["used_as"] = f"{res_type_name(res_type)} resource {res.name}"
                except (FormatError, UnsupportedCompression, OSError, ValueError) as exc:
                    rec["used_as"] = f"ignored: {exc}"
                if rec["used_as"]:
                    used.append(rec)
                    log.info("extra input %s -> %s", path, rec["used_as"])
        self.report["extra_inputs"] = used

    # ------------------------------------------------------------------
    def _analyse_ebx(self) -> None:
        self.ebx_by_guid: dict[str, dict] = {}
        self.ebx_by_name: dict[str, dict] = {}
        for e in self.ebx:
            r: ModResource = e["res"]
            self.ebx_by_name[(r.name or "").lower()] = e
            if e["data"] is None:
                e["category"] = "reference_only" if not r.has_payload else "undecodable"
                continue
            try:
                part = EbxPartition(e["data"], name=r.name)
            except FormatError as exc:
                e["error"] = f"EBX parse failed: {exc}"
                e["category"] = "undecodable"
                self.report["errors"].append(f"ebx '{r.name}': {exc}")
                continue
            e["partition"] = part
            self.ebx_by_guid[part.partition_guid.lower()] = e
            chain = part.class_chain(part.root_type) if part.root_type else []
            e["chain"] = chain
            if "MeshAsset" in chain:
                e["category"] = "mesh_asset"
            elif "TextureBaseAsset" in chain:
                e["category"] = "texture_asset"
            elif "SkeletonAsset" in chain:
                e["category"] = "skeleton_asset"
            elif part.root_type in ("MeshVariationDatabase", "ObjectVariation"):
                e["category"] = "variation"
            else:
                e["category"] = "other"
            for err in part.errors:
                self.report["warnings"].append(f"ebx '{r.name}': {err}")
            if self.opts.write_raw and "raw_path" in e:
                dump = {"partition": part.describe(),
                        "objects": [{"index": o.index, "type": o.type_name, "instance_guid": o.instance_guid,
                                     "fields": to_jsonable(o.fields)} for o in part.objects]}
                Path(self.out / e["raw_path"]).with_suffix(".json").write_text(json.dumps(dump, indent=1))

    def _analyse_res(self) -> None:
        self.res_by_rid: dict[int, dict] = {}
        self.res_by_name: dict[str, dict] = {}
        for e in self.res:
            r: ModResource = e["res"]
            self.res_by_rid[r.res_rid] = e
            self.res_by_name[(r.name or "").lower()] = e
            e["kind"] = res_type_name(r.res_type)
            if e["data"] is None:
                continue
            if r.res_type == RES_TYPE_MESHSET:
                try:
                    e["meshset"] = parse_meshset(e["data"], r.res_meta)
                except (FormatError, IndexError) as exc:
                    e["error"] = f"MeshSet parse failed: {exc}"
                    self.report["errors"].append(f"res '{r.name}': {e['error']}")
            elif r.res_type == RES_TYPE_TEXTURE:
                try:
                    e["texture"] = parse_texture_header(e["data"])
                except FormatError as exc:
                    e["error"] = f"texture header parse failed: {exc}"
                    self.report["errors"].append(f"res '{r.name}': {e['error']}")
            else:
                self.report["unsupported"].append({
                    "resource": r.name, "kind": e["kind"],
                    "reason": "resource type not decoded by this extractor (listed in manifest, raw payload kept)"})

    # ------------------------------------------------------------------
    def _chunk_status(self, guid: str) -> dict:
        c = self.chunks.get(guid.lower())
        if c is None:
            return {"chunk": guid, "status": "missing", "in_mod": False}
        if c["data"] is None:
            return {"chunk": guid, "status": "no_payload" if not c["res"].has_payload else "undecodable",
                    "in_mod": True, "error": c.get("error")}
        st = {"chunk": guid, "status": "present", "in_mod": c["res"] is not None, "size": len(c["data"])}
        if c.get("source"):
            st["supplied_by"] = c["source"]
        return st

    def _write_manifest(self) -> None:
        ref_chunks: dict[str, list[str]] = {}
        resources = []
        for r in self.mod.resources:
            d = r.describe()
            entry = self._entry_for(r)
            if entry is not None:
                if entry.get("payload"):
                    d["payload"] = entry["payload"]
                if entry.get("error"):
                    d["error"] = entry["error"]
                if entry.get("raw_path"):
                    d["raw_path"] = entry["raw_path"]
            if r.type == "ebx" and entry is not None:
                d["category"] = entry.get("category")
                part: EbxPartition | None = entry.get("partition")
                if part is not None:
                    d["ebx"] = part.describe()
                    d["ebx"]["root_type_chain"] = entry.get("chain")
                    root = part.root
                    if root is not None and entry.get("category") == "mesh_asset":
                        rid = root.fields.get("MeshSetResource")
                        d["mesh_set_resource_rid"] = f"0x{rid:016X}" if isinstance(rid, int) else None
                        d["mesh_set_res_in_mod"] = isinstance(rid, int) and rid in self.res_by_rid
                    if entry.get("category") == "variation":
                        d["variation_entries"] = variation_entries(part)
                    if root is not None and entry.get("category") == "texture_asset":
                        rid = root.fields.get("Resource")
                        d["texture_resource_rid"] = f"0x{rid:016X}" if isinstance(rid, int) else None
                        d["texture_res_in_mod"] = isinstance(rid, int) and rid in self.res_by_rid
            if r.type == "res" and entry is not None:
                d["kind"] = entry.get("kind")
                ms: MeshSet | None = entry.get("meshset")
                tx: TextureHeader | None = entry.get("texture")
                if ms is not None:
                    d["meshset"] = ms.describe()
                    deps = []
                    for lod in ms.lods:
                        if lod.uses_chunk:
                            st = self._chunk_status(lod.chunk_id)
                            st["purpose"] = f"LOD {lod.index} vertex/index buffers"
                            deps.append(st)
                            ref_chunks.setdefault(lod.chunk_id.lower(), []).append(r.name)
                        else:
                            deps.append({"purpose": f"LOD {lod.index} vertex/index buffers", "status": "inline",
                                         "in_mod": True})
                    d["dependencies"] = deps
                    d["complete"] = all(x["status"] in ("present", "inline") for x in deps)
                if tx is not None:
                    d["texture"] = tx.describe()
                    if tx.chunk_id != NULL_GUID:
                        st = self._chunk_status(tx.chunk_id)
                        st["purpose"] = "pixel data"
                        d["dependencies"] = [st]
                        d["complete"] = st["status"] == "present"
                        ref_chunks.setdefault(tx.chunk_id.lower(), []).append(r.name)
            if r.type == "chunk":
                d["referenced_by"] = []  # filled below
            if not r.has_payload and r.type in ("ebx", "res", "chunk"):
                d["note"] = ("no payload: the mod only records bundle membership for this asset; "
                             "its data comes from the original game")
            resources.append(d)
        for d in resources:
            if d["type"] == "chunk":
                d["referenced_by"] = ref_chunks.get((d["name"] or "").lower(), [])
        self.manifest = {"mod": self.report["mod"], "resources": resources}
        (self.out / "manifest.json").write_text(json.dumps(self.manifest, indent=1))
        self.report["manifest"] = "manifest.json"

    def _entry_for(self, r: ModResource) -> dict | None:
        if r.type == "ebx":
            return next((e for e in self.ebx if e["res"] is r), None)
        if r.type == "res":
            return next((e for e in self.res if e["res"] is r), None)
        if r.type == "chunk":
            return self.chunks.get((r.name or "").lower())
        return None

    # ------------------------------------------------------------------
    def _extract_textures(self) -> None:
        tex_dir = self.out / "Extracted_Textures"
        for e in self.res:
            tx: TextureHeader | None = e.get("texture")
            r: ModResource = e["res"]
            if tx is None:
                continue
            info: dict = {"resource": r.name, "header": tx.describe(), "outputs": {}, "status": "pending"}
            self.report["textures"].append(info)
            ebx_e = self._texture_ebx_for(r)
            info["texture_asset_ebx"] = ebx_e["res"].name if ebx_e else None
            info["texture_asset_partition"] = ebx_e["partition"].partition_guid if ebx_e and ebx_e.get("partition") else None
            role = infer_texture_role(r.name)
            info["role"] = role
            if tx.chunk_id == NULL_GUID:
                info["status"] = "failed"
                info["error"] = "texture has no chunk id"
                continue
            chunk = self.chunks.get(tx.chunk_id.lower())
            if chunk is None or chunk["data"] is None:
                info["status"] = "missing_dependency"
                self.report["missing_dependencies"].append({
                    "needed_by": r.name, "kind": "chunk", "id": tx.chunk_id,
                    "purpose": "texture pixel data",
                    "why": "the mod replaces the texture header but not its pixel data",
                    "how_to_provide": HOWTO_CHUNK})
                continue
            try:
                mips, layout = split_mips(tx, chunk["data"])
                info["mip_layout"] = layout
                base = tex_dir / _safe(_short(r.name))
                base.parent.mkdir(parents=True, exist_ok=True)
                dds = base.with_suffix(".dds")
                write_dds(dds, tx, mips)
                info["outputs"]["dds"] = dds.relative_to(self.out).as_posix()
                img = decode_to_image(tx, mips[0])
                png = base.with_suffix(".png")
                img.save(png)
                info["outputs"]["png"] = png.relative_to(self.out).as_posix()
                info["decoded_mip"] = mips[0].level
                info["image_mode"] = img.mode
                info["outputs"].update(self._derived_channels(img, base, role, tx))
                info["status"] = "ok"
                self.texture_outputs[(r.name or "").lower()] = info
                log.info("texture %s: %s %dx%d -> %s", r.name, tx.format_name, tx.width, tx.height, info["outputs"]["png"])
            except Exception as exc:  # report, never abort the whole run on one asset
                info["status"] = "failed"
                info["error"] = f"{type(exc).__name__}: {exc}"
                self.report["errors"].append(f"texture '{r.name}': {info['error']}")
                log.debug("texture failure", exc_info=True)

    def _texture_from_image(self, tex_name: str, path: Path) -> dict | None:
        """Use a texture image the user exported from the game (``--extra-dir``)."""
        key = tex_name.lower()
        if key in self.texture_outputs:
            return self.texture_outputs[key]
        role = infer_texture_role(tex_name)
        info: dict = {"resource": tex_name, "header": {"pixel_format": "supplied image"}, "outputs": {},
                      "role": role, "supplied_by": str(path)}
        try:
            img = Image.open(path)
            img.load()
        except OSError as exc:
            info.update(status="failed", error=f"cannot read supplied image: {exc}")
            self.report["textures"].append(info)
            return None
        base = self.out / "Extracted_Textures" / _safe(_short(tex_name))
        base.parent.mkdir(parents=True, exist_ok=True)
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGBA")
        png = base.with_suffix(".png")
        img.save(png)
        info["outputs"]["png"] = png.relative_to(self.out).as_posix()
        info["image_mode"] = img.mode
        info["outputs"].update(self._derived_channels(img, base, role, None))
        info["status"] = "ok"
        self.report["textures"].append(info)
        self.texture_outputs[key] = info
        return info

    def _texture_ebx_for(self, r: ModResource) -> dict | None:
        for e in self.ebx:
            part = e.get("partition")
            if e.get("category") != "texture_asset" or part is None or part.root is None:
                continue
            if part.root.fields.get("Resource") == r.res_rid:
                return e
        e = self.ebx_by_name.get((r.name or "").lower())
        return e if e and e.get("category") == "texture_asset" else None

    def _derived_channels(self, img: Image.Image, base: Path, role: dict, tx: TextureHeader) -> dict:
        """Split packed channels per the role convention; every output is listed as derived."""
        out = {}
        arr = np.asarray(img)
        chans = role.get("channels", {})
        if arr.ndim != 3:
            return out
        if "base_color" in chans and arr.shape[2] >= 3:
            p = base.with_name(base.name + ".basecolor.png")
            Image.fromarray(arr[:, :, :3]).save(p)
            out["base_color"] = p.relative_to(self.out).as_posix()
        if chans.get("smoothness") == "A" and arr.shape[2] == 4:
            p = base.with_name(base.name + ".roughness.png")
            Image.fromarray(255 - arr[:, :, 3]).save(p)
            out["roughness_from_inverted_alpha"] = p.relative_to(self.out).as_posix()
        if chans.get("metallic") == "A" and arr.shape[2] == 4:
            p = base.with_name(base.name + ".metallic.png")
            Image.fromarray(arr[:, :, 3]).save(p)
            out["metallic_from_alpha"] = p.relative_to(self.out).as_posix()
        if "normal" in chans:
            out["normal_source"] = "pending_convention"
        return out

    def _write_normal_map(self, tex_info: dict, flip_green: bool) -> str | None:
        """OpenGL-convention normal map (what glTF/Blender expect) derived from the decoded texture."""
        png = tex_info["outputs"].get("png")
        if not png:
            return None
        arr = np.asarray(Image.open(self.out / png).convert("RGBA")).astype(np.float64) / 255.0
        x = arr[:, :, 0] * 2 - 1
        y = arr[:, :, 1] * 2 - 1
        if tex_info["header"].get("pixel_format", "").startswith("BC5") or tex_info.get("image_mode") == "RGB" and \
                np.allclose(arr[:, :, 2], 0):
            z = np.sqrt(np.clip(1 - x * x - y * y, 0, 1))
            tex_info["normal_z"] = "reconstructed from XY (two-channel source)"
        else:
            z = arr[:, :, 2] * 2 - 1
            tex_info["normal_z"] = "stored"
        if flip_green:
            y = -y
        rgb = np.dstack([x, y, z]) * 0.5 + 0.5
        p = (self.out / png).with_name(Path(png).stem + ".normal_gl.png")
        Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB").save(p)
        rel = p.relative_to(self.out).as_posix()
        tex_info["outputs"]["normal_opengl"] = rel
        tex_info["outputs"].pop("normal_source", None)
        return rel

    # ------------------------------------------------------------------
    def _selected_lods(self, ms: MeshSet) -> list[int]:
        if self.opts.lods == "all":
            return [lod.index for lod in ms.lods]
        want = {int(x) for x in self.opts.lods.split(",") if x.strip() != ""}
        return [lod.index for lod in ms.lods if lod.index in want]

    def _mesh_ebx_for(self, r: ModResource) -> dict | None:
        for e in self.ebx:
            part = e.get("partition")
            if e.get("category") == "mesh_asset" and part is not None and part.root is not None:
                if part.root.fields.get("MeshSetResource") == r.res_rid:
                    return e
        e = self.ebx_by_name.get((r.name or "").lower())
        return e if e and e.get("category") == "mesh_asset" else None

    def _output_name(self, ms_name: str) -> str:
        low = ms_name.lower()
        name = _safe(next((v for k, v in self.opts.names.items() if k.lower() in low), None) or _short(ms_name))
        used = {m["output_name"] for m in self.report["meshes"]}
        base, n = name, 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        return name

    def _extract_meshes(self) -> None:
        mesh_dir = self.out / "meshes"
        for e in self.res:
            if e.get("meshset") is None:
                continue
            try:
                self._extract_mesh(e, mesh_dir)
            except Exception as exc:  # report, never abort the whole run on one asset
                log.exception("unexpected failure while extracting mesh %s", e["res"].name)
                self.report["errors"].append(f"mesh '{e['res'].name}': internal error {type(exc).__name__}: {exc}")
                for m in self.report["meshes"]:
                    if m["resource"] == e["res"].name and m["status"] in ("pending", "decoded"):
                        m["status"] = "failed"
                        m["error"] = f"internal error {type(exc).__name__}: {exc} (see logs/extract.log)"

    def _extract_mesh(self, e: dict, mesh_dir: Path) -> None:
        ms: MeshSet = e["meshset"]
        r: ModResource = e["res"]
        out_name = self._output_name(r.name or ms.full_name)
        info: dict = {"resource": r.name, "meshset_name": ms.full_name, "output_name": out_name,
                      "mesh_type": ms.describe()["mesh_type"], "status": "pending", "lods": [],
                      "warnings": list(ms.warnings)}
        self.report["meshes"].append(info)
        mesh_e = self._mesh_ebx_for(r)
        info["mesh_asset_ebx"] = mesh_e["res"].name if mesh_e else None
        materials: list[MaterialInfo] = mesh_materials(mesh_e["partition"]) if mesh_e else []
        info["materials_source"] = "mesh asset EBX in mod" if mesh_e else "not available (mesh EBX not in mod)"

        decoded: list[list[DecodedSection]] = []
        for lod_index in self._selected_lods(ms):
            lod = ms.lods[lod_index]
            lod_info = {"lod": lod_index, "sections": [], "status": "pending"}
            info["lods"].append(lod_info)
            if lod.uses_chunk:
                st = self._chunk_status(lod.chunk_id)
                if st["status"] != "present":
                    log.warning("mesh %s LOD %d: chunk %s not available (%s)", r.name, lod_index, lod.chunk_id,
                                st["status"])
                    lod_info["status"] = "missing_dependency"
                    self.report["missing_dependencies"].append({
                        "needed_by": r.name, "kind": "chunk", "id": lod.chunk_id,
                        "purpose": f"LOD {lod_index} vertex and index buffers",
                        "why": "the mod modifies the MeshSet header but does not carry this LOD's geometry",
                        "how_to_provide": HOWTO_CHUNK})
                    continue
                buf = self.chunks[lod.chunk_id.lower()]["data"]
            else:
                buf = lod.inline_data
                if buf is None:
                    lod_info["status"] = "failed"
                    lod_info["error"] = "inline LOD data not found in resource"
                    continue
            expected = lod.vertex_buffer_size + lod.index_buffer_size
            if len(buf) < expected:
                lod_info["status"] = "failed"
                lod_info["error"] = f"geometry buffer has {len(buf)} bytes, LOD declares {expected}"
                self.report["errors"].append(f"mesh '{r.name}' LOD {lod_index}: {lod_info['error']}")
                continue
            try:
                isize, how = choose_index_size(buf, lod)
            except MeshDecodeError as exc:
                lod_info["status"] = "failed"
                lod_info["error"] = str(exc)
                self.report["errors"].append(f"mesh '{r.name}': {exc}")
                continue
            lod_info["index_size"] = isize
            lod_info["index_size_basis"] = how
            sections = []
            for sec in lod.sections:
                s_info = {"section": sec.index, "material_name": sec.material_name,
                          "material_id": sec.material_id, "categories": lod.section_categories(sec.index),
                          "vertex_declaration": sec.decl.describe()}
                lod_info["sections"].append(s_info)
                if sec.primitive_count == 0 or sec.vertex_count == 0:
                    s_info["status"] = "empty"
                    continue
                try:
                    ds = decode_section(buf, ms, lod, sec, isize)
                except (MeshDecodeError, FormatError, ValueError, IndexError) as exc:
                    s_info["status"] = "failed"
                    s_info["error"] = f"{type(exc).__name__}: {exc}"
                    self.report["errors"].append(f"mesh '{r.name}' LOD {lod_index} section {sec.index}: {exc}")
                    log.debug("section decode failure", exc_info=True)
                    continue
                s_info["status"] = "decoded"
                s_info["checks"] = ds.checks
                s_info["ignored_vertex_elements"] = ds.ignored_elements
                s_info["warnings"] = ds.warnings
                sections.append(ds)
            lod_info["status"] = "decoded" if sections else "no_geometry"
            log.info("mesh %s LOD %d: %d/%d sections decoded, %d vertices, %d triangles (index size %d, %s)",
                     r.name, lod_index, len(sections), len(lod.sections),
                     sum(len(d.positions) for d in sections), sum(len(d.indices) for d in sections), isize, how)
            for d in sections:
                log.debug("  section %d '%s': checks %s", d.section_index, d.material_name, d.checks)
            if sections:
                decoded.append(sections)

        if not decoded:
            info["status"] = "missing_dependency" if any(
                lod["status"] == "missing_dependency" for lod in info["lods"]) else "failed"
            return
        self._finalise_mesh(info, r, ms, mesh_e, materials, decoded, mesh_dir)

    def _normal_policy(self, decoded: list[list[DecodedSection]]):
        keep: dict[tuple[int, int], bool] = {}
        flip: dict[tuple[int, int], bool] = {}
        notes = []
        reliable_flips = []
        for sections in decoded:
            for ds in sections:
                key = (ds.lod_index, ds.section_index)
                c = ds.checks
                agree = c.get("normal_agreement")
                flipped = c.get("normal_agreement_if_winding_flipped")
                if ds.normals is None:
                    keep[key] = False
                    continue
                if self.opts.normals == "none":
                    keep[key] = False
                elif agree is not None and agree >= NORMAL_OK:
                    keep[key] = True
                    reliable_flips.append(False)
                elif flipped is not None and flipped >= NORMAL_OK:
                    keep[key] = True
                    flip[key] = True
                    reliable_flips.append(True)
                else:
                    keep[key] = self.opts.normals == "stored"
                    notes.append(f"LOD {ds.lod_index} section {ds.section_index}: decoded normals agree with the "
                                 f"triangle geometry for only {max(agree or 0, flipped or 0):.0%} of vertices; "
                                 + ("kept because --normals stored" if keep[key] else
                                    "omitted so Blender derives shading normals from the real geometry"))
        majority = None
        if reliable_flips:
            majority = sum(reliable_flips) > len(reliable_flips) / 2
            for sections in decoded:
                for ds in sections:
                    key = (ds.lod_index, ds.section_index)
                    if key not in flip and not (keep.get(key) and ds.checks.get("normal_agreement", 0) >= NORMAL_OK):
                        flip[key] = majority
        winding_basis = ("stored normals vs. triangle orientation" if reliable_flips
                         else "unverified (no reliable normals); source winding kept")
        return keep, flip, notes, winding_basis, majority

    def _normal_map_convention(self, decoded: list[list[DecodedSection]], keep: dict) -> dict:
        """Relate decoded tangent frames to UV gradients to learn the normal-map green convention."""
        t_dots, b_dots = [], []
        for sections in decoded:
            for ds in sections:
                if not keep.get((ds.lod_index, ds.section_index)) or ds.tangents is None or not ds.uvs:
                    continue
                uv = ds.uvs[sorted(ds.uvs)[0]].astype(np.float64)
                p = ds.positions.astype(np.float64)
                tri = ds.indices
                e1, e2 = p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]]
                d1, d2 = uv[tri[:, 1]] - uv[tri[:, 0]], uv[tri[:, 2]] - uv[tri[:, 0]]
                det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
                ok = np.abs(det) > 1e-12
                if not ok.any():
                    continue
                r = 1.0 / det[ok]
                dpdu = (e1[ok] * d2[ok, 1:2] - e2[ok] * d1[ok, 1:2]) * r[:, None]
                dpdv = (e2[ok] * d1[ok, 0:1] - e1[ok] * d2[ok, 0:1]) * r[:, None]
                v0 = tri[ok, 0]
                t = ds.tangents[v0, :3].astype(np.float64)
                n = ds.normals[v0].astype(np.float64)
                b = np.cross(n, t) * ds.tangents[v0, 3:4]
                t_dots.append(np.sign(np.einsum("ij,ij->i", t, dpdu)))
                b_dots.append(np.sign(np.einsum("ij,ij->i", b, dpdv)))
        if not t_dots:
            return {"flip_green": True, "basis": "assumed (DirectX-style, Frostbite is a D3D engine); "
                    "no reliable tangent frames to verify", "verified": False}
        t_mean = float(np.concatenate(t_dots).mean())
        b_mean = float(np.concatenate(b_dots).mean())
        # Game bitangent along +dP/dv (V down) is the opposite of Blender's +V (up): flip green.
        return {"flip_green": b_mean > 0, "verified": abs(b_mean) > 0.5 and t_mean > 0.5,
                "tangent_vs_dPdu": round(t_mean, 3), "bitangent_vs_dPdv": round(b_mean, 3),
                "basis": "decoded tangent frames compared with UV gradients"}

    def _infer_texture(self, role: str, hints: list[str]) -> tuple[str, dict, str] | None:
        """Pick the mod texture with ``role`` whose name best shares leading tokens with a hint.

        Used only when the EBX link is missing; returns None unless the best match is unique.
        """
        best, best_score, tie = None, 0, False
        for name, out in self.texture_outputs.items():
            if out["role"]["role"] != role:
                continue
            stem = _tex_stem(name)
            score = max((_shared_prefix_tokens(stem, _tex_stem(h)) for h in hints if h), default=0)
            if score > best_score:
                best, best_score, tie = (name, out), score, False
            elif score == best_score and score > 0:
                tie = True
        if best is None or tie:
            return None
        return best[0], best[1], f"inferred: '{_short(best[0])}' shares {best_score} leading name token(s) with " \
                                 f"{[h for h in hints if h]}"

    def _resolve_textures(self, info: dict, ms: MeshSet, materials: list[MaterialInfo],
                          decoded: list[list[DecodedSection]]) -> dict[str, dict]:
        """Map material key -> {role: texture binding}.

        EBX links (material parameter -> texture partition -> texture asset) are used when the
        mod carries the EBX; otherwise textures *in the mod* are matched by name and every such
        binding is labelled ``inferred``.
        """
        bindings: dict[str, dict] = {}
        unresolved = []
        mesh_hint = re.sub(r"(_mesh)?(_lod\d+)?$", "", _short(ms.full_name or info["resource"]).lower())
        if materials:
            for m in materials:
                key = f"material_{m.index}"
                roles = {}
                sec_names = sorted({ds.material_name for lod in decoded for ds in lod if ds.material_id == m.index})
                for tp in m.textures:
                    tex_ebx = self.ebx_by_guid.get(tp.partition_guid.lower())
                    tex_name = tex_ebx["res"].name if tex_ebx else None
                    out = self.texture_outputs.get((tex_name or "").lower()) if tex_name else None
                    if out is None and tex_name and _short(tex_name).lower() in self.extra_images:
                        out = self._texture_from_image(tex_name, self.extra_images[_short(tex_name).lower()])
                    role = infer_texture_role(tex_name, tp.parameter)
                    entry = {"parameter": tp.parameter, "texture_partition": tp.partition_guid,
                             "texture_asset": tex_name, "role": role, "binding_basis": "mesh EBX material parameter"}
                    if out is None and tex_name is None and role["role"] != "unknown":
                        guess = self._infer_texture(role["role"], sec_names + [mesh_hint])
                        if guess:
                            entry["texture_asset"], out, entry["binding_basis"] = guess
                            entry["binding_basis"] += " (texture EBX not in mod, so the partition GUID could not be resolved)"
                    if out is None:
                        entry["status"] = "texture data not in mod"
                        unresolved.append(entry)
                        self.report["missing_dependencies"].append({
                            "needed_by": f"{info['resource']} material {m.index}", "kind": "texture",
                            "id": tex_name or tp.partition_guid, "purpose": f"shader parameter {tp.parameter}",
                            "why": "the material references a texture the mod does not modify",
                            "how_to_provide": HOWTO_TEXTURE})
                    else:
                        entry["status"] = "bound"
                        entry["texture"] = out
                        roles.setdefault(role["role"], entry)
                bindings[key] = {"material": m.describe(), "roles": roles}
        else:
            mat_names = sorted({ds.material_name for lod in decoded for ds in lod})
            mod_roles = sorted({o["role"]["role"] for o in self.texture_outputs.values()} - {"unknown"})
            for k in mat_names:
                roles = {}
                for role_name in mod_roles:
                    guess = self._infer_texture(role_name, [k, mesh_hint])
                    if guess:
                        name, out, basis = guess
                        roles[role_name] = {"texture_asset": name, "role": out["role"], "status": "bound",
                                            "texture": out, "binding_basis": basis}
                bindings[f"name:{k}"] = {"material": {"name": k}, "roles": roles}
            if self.texture_outputs and not any(b["roles"] for b in bindings.values()):
                info["warnings"].append("textures are in the mod but none could be matched to this mesh by name; "
                                        "include the mesh EBX (MeshAsset) to link them exactly")
        info["texture_bindings_unresolved"] = unresolved
        return bindings

    def _finalise_mesh(self, info, r, ms, mesh_e, materials, decoded, mesh_dir) -> None:
        keep, flip, notes, winding_basis, majority = self._normal_policy(decoded)
        info["normals"] = {"policy": self.opts.normals, "notes": notes,
                           "kept_sections": sum(1 for v in keep.values() if v),
                           "dropped_sections": sum(1 for v in keep.values() if not v)}
        info["winding"] = {"basis": winding_basis,
                           "source_winding_reversed": bool(majority) if majority is not None else None,
                           "axis_mode": self.opts.axis}
        nm = self._normal_map_convention(decoded, keep)
        info["normal_map_convention"] = nm

        bindings = self._resolve_textures(info, ms, materials, decoded)
        gl_materials: list[GltfMaterial] = []
        sec_mat: dict[tuple[int, int], int] = {}
        mat_index: dict[str, int] = {}
        mesh_dir.mkdir(parents=True, exist_ok=True)

        def rel(p: str) -> str:
            return (Path("..") / p).as_posix()

        for sections in decoded:
            for ds in sections:
                key = f"material_{ds.material_id}" if materials else f"name:{ds.material_name}"
                if key not in mat_index:
                    gm = GltfMaterial(name=ds.material_name or key)
                    bind = bindings.get(key)
                    channels = {}
                    if bind:
                        roles = bind["roles"]
                        cs = roles.get("color_smoothness") or roles.get("base_color")
                        if cs:
                            outs = cs["texture"]["outputs"]
                            gm.base_color_uri = rel(outs.get("base_color") or outs["png"])
                            channels["base_color"] = {"file": gm.base_color_uri, "basis": cs["binding_basis"],
                                                      "role_confidence": cs["role"]["confidence"]}
                            if "roughness_from_inverted_alpha" in outs:
                                channels["roughness"] = {"file": rel(outs["roughness_from_inverted_alpha"]),
                                                         "derived": "1 - smoothness (alpha of _CS)"}
                        nmr = roles.get("normal_metallic")
                        if nmr:
                            normal_file = self._write_normal_map(nmr["texture"], nm["flip_green"])
                            if normal_file:
                                gm.normal_uri = rel(normal_file)
                                channels["normal"] = {"file": gm.normal_uri, "flip_green": nm["flip_green"],
                                                      "convention_basis": nm["basis"]}
                            outs = nmr["texture"]["outputs"]
                            if "metallic_from_alpha" in outs:
                                channels["metallic"] = {"file": rel(outs["metallic_from_alpha"]),
                                                        "derived": "alpha of normal/metallic texture"}
                        gm.extras = {"fbmod_channels": channels,
                                     "fbmod_bindings": {k: {kk: vv for kk, vv in v.items() if kk != "texture"}
                                                        for k, v in roles.items()}}
                    mat_index[key] = len(gl_materials)
                    gl_materials.append(gm)
                sec_mat[(ds.lod_index, ds.section_index)] = mat_index[key]
        info["materials"] = [{"name": m.name, "base_color": m.base_color_uri, "normal": m.normal_uri,
                              "channels": m.extras.get("fbmod_channels", {})} for m in gl_materials]

        skeleton, skel_note = self._load_skeleton(decoded)
        info["skeleton"] = skel_note
        glb = mesh_dir / f"{info['output_name']}.glb"
        summary = write_glb(glb, info["output_name"], decoded, gl_materials, sec_mat, axis=self.opts.axis,
                            skeleton=skeleton, keep_normals=keep, flip_winding=flip,
                            extras={"source_mod": self.mod_path.name, "meshset": ms.full_name,
                                    "bbox_min": list(ms.bbox_min), "bbox_max": list(ms.bbox_max)})
        info["gltf"] = glb.relative_to(self.out).as_posix()
        info["gltf_summary"] = summary
        info["totals"] = {
            "vertices": int(sum(len(ds.positions) for s in decoded for ds in s)),
            "triangles": int(sum(len(ds.indices) for s in decoded for ds in s)),
            "sections_decoded": int(sum(len(s) for s in decoded)),
        }
        all_ok = all(ds.checks.get("indices_in_range") and ds.checks.get("positions_finite")
                     for s in decoded for ds in s)
        info["status"] = "decoded" if all_ok else "decoded_with_errors"
        sidecar = {
            "mesh": info["output_name"], "axis": self.opts.axis,
            "sections": [{"lod": ds.lod_index, "section": ds.section_index, "material": ds.material_name,
                          "has_skin": ds.joints is not None} for s in decoded for ds in s],
            "skin_mode": "armature" if skeleton is not None else (
                "vertex_groups_by_bone_index" if any(ds.joints is not None for s in decoded for ds in s) else "none"),
        }
        (mesh_dir / f"{info['output_name']}.meta.json").write_text(json.dumps(sidecar, indent=1))
        expect = build_expectations(info["output_name"], decoded, gl_materials, sec_mat, self.opts.axis, skeleton)
        expect_path = mesh_dir / f"{info['output_name']}.expect.json"
        expect_path.write_text(json.dumps(expect, indent=1))
        info["blender_expectations"] = expect_path.relative_to(self.out).as_posix()

    def _load_skeleton(self, decoded) -> tuple[SkeletonData | None, dict]:
        has_skin = any(ds.joints is not None for s in decoded for ds in s)
        if not has_skin:
            return None, {"status": "not_applicable", "detail": "mesh has no bone weights"}
        part = None
        source = None
        if self.opts.skeleton:
            data = Path(self.opts.skeleton).read_bytes()
            if looks_block_compressed(data):
                data, _ = decompress(data)
            part = EbxPartition(data, name=str(self.opts.skeleton))
            source = str(self.opts.skeleton)
        else:
            skels = [e for e in self.ebx if e.get("category") == "skeleton_asset"]
            if len(skels) == 1:
                part = skels[0]["partition"]
                source = skels[0]["res"].name
        max_joint = max(int(ds.joints[ds.weights > 0].max()) for s in decoded for ds in s
                        if ds.joints is not None and (ds.weights > 0).any())
        if part is None or part.root is None:
            return None, {"status": "missing", "max_bone_index_used": max_joint,
                          "detail": "no SkeletonAsset in the mod and none supplied with --skeleton; "
                                    "bone weights are kept as vertex groups named by bone index (GLB/.blend); "
                                    "FBX cannot carry weights without an armature",
                          "how_to_provide": HOWTO_SKELETON}
        f = part.root.fields
        names = f.get("BoneNames") or []
        parents = f.get("Hierarchy") or []
        local = f.get("LocalPose") or []
        if not (names and len(names) == len(parents) == len(local)):
            return None, {"status": "invalid", "source": source,
                          "detail": "skeleton EBX lacks consistent BoneNames/Hierarchy/LocalPose"}
        if max_joint >= len(names):
            return None, {"status": "incompatible", "source": source, "bones": len(names),
                          "max_bone_index_used": max_joint,
                          "detail": "mesh references bone indices beyond the skeleton (procedural bones?); "
                                    "weights kept as vertex groups"}
        mats = []
        for lt in local:
            m = np.zeros((4, 4))
            for row, key in enumerate(("right", "up", "forward", "trans")):
                v = lt[key]
                m[row, :3] = [v["x"], v["y"], v["z"]]
            m[3, 3] = 1.0
            mats.append(m)
        return SkeletonData(list(names), list(parents), mats, source), {
            "status": "bound", "source": source, "bones": len(names), "max_bone_index_used": max_joint}

    # ------------------------------------------------------------------
    def _finish(self) -> None:
        r = self.report
        r["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if self.opts.names:
            requested = []
            for sub, out_name in self.opts.names.items():
                hits = [m for m in r["meshes"] if m["output_name"] == out_name]
                entry = {"output_name": out_name, "matches_resource_substring": sub}
                if not hits:
                    entry["status"] = "not_in_mod"
                    entry["detail"] = (f"no MeshSet resource in the mod has '{sub}' in its name, so this mesh is not "
                                       "modified by the mod and its geometry exists only in the original game. "
                                       "Export its MeshSet (.res) and LOD chunk(s) with Frosty Editor's "
                                       "Chunk/Res Editor and pass them with --extra-dir.")
                else:
                    m = hits[0]
                    entry["status"] = m["status"]
                    entry["resource"] = m["resource"]
                    entry["gltf"] = m.get("gltf")
                    entry["fbx"] = m.get("blender", {}).get("fbx")
                    if len(hits) > 1:
                        entry["note"] = f"{len(hits)} meshes matched; outputs share the name"
                requested.append(entry)
            r["requested_outputs"] = requested
        decoded = [m for m in r["meshes"] if m["status"].startswith("decoded")]
        tex_ok = [t for t in r["textures"] if t["status"] == "ok"]
        r["summary"] = {
            "meshsets_in_mod": len([e for e in self.res if e.get("meshset") is not None]),
            "meshes_decoded": len(decoded),
            "textures_in_mod": len(r["textures"]),
            "textures_decoded": len(tex_ok),
            "missing_dependencies": len(r["missing_dependencies"]),
            "errors": len(r["errors"]),
        }
        if not decoded:
            r["outcome"] = "no_geometry_decoded"
            cats: dict[str, int] = {}
            external = set()
            own = {e["partition"].partition_guid.lower() for e in self.ebx if e.get("partition")}
            for e in self.ebx:
                cats[e.get("category") or "unknown"] = cats.get(e.get("category") or "unknown", 0) + 1
                if e.get("partition"):
                    external |= {i.partition_guid.lower() for i in e["partition"].imports} - own
            if r["summary"]["meshsets_in_mod"] == 0:
                r["no_geometry_reason"] = (
                    "the mod contains no MeshSet (mesh geometry) resources. It carries "
                    f"{len(self.ebx)} EBX assets ({cats}), {len(self.res)} RES and {len(self.chunks)} chunk "
                    f"resources; its EBX assets reference {len(external)} partitions that live only in the "
                    "original game data, so any meshes it shows in game come from the base game, not the mod")
            else:
                r["no_geometry_reason"] = "MeshSet resources were found but none could be decoded; see meshes[] and errors[]"
        elif r["missing_dependencies"] or r["errors"]:
            r["outcome"] = "partial"
        else:
            r["outcome"] = "complete"
        (self.out / "Extraction_Report.json").write_text(json.dumps(r, indent=1, default=str))
