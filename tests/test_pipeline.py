import json
import numpy as np
import pytest

from fbmod_extractor.blender_export import find_blender
from fbmod_extractor.cli import main
from fbmod_extractor.extract import Extraction, ExtractOptions
from fbmod_extractor.gltf import accessor_array, read_glb
from fbmod_extractor.selftest import compare_glb_to_truth
from fbmod_extractor.synth import (DECL_SKINNED, STRIDES_SKINNED, SynthLod, SynthSection, build_lod_buffers,
                                   make_capsule, synthetic_hero_mod)


def _run(tmp_path, name="m", **kw):
    gen = {k: kw.pop(k) for k in ("version", "with_mesh_ebx", "drop_chunk", "with_skeleton") if k in kw}
    data, truth = synthetic_hero_mod(**gen)
    mod = tmp_path / f"{name}.fbmod"
    mod.write_bytes(data)
    out = tmp_path / f"{name}_out"
    report = Extraction(mod, ExtractOptions(output=out, **kw)).run()
    return report, truth, out


def test_end_to_end_complete(tmp_path):
    report, truth, out = _run(tmp_path, lods="all", names={"synth_body": "Hero_Body", "synth_head": "Hero_Head"})
    assert report["outcome"] == "complete" and not report["errors"]
    meshes = {m["output_name"]: m for m in report["meshes"]}
    body, head = meshes["Hero_Body"], meshes["Hero_Head"]
    assert body["status"] == head["status"] == "decoded"
    assert body["mesh_asset_ebx"] == "characters/heroes/synth/synth_body_mesh"
    assert body["skeleton"]["status"] == "bound"
    assert body["normal_map_convention"]["verified"] and body["normal_map_convention"]["flip_green"]
    for check in compare_glb_to_truth(out / body["gltf"], truth["body"], "gltf"):
        assert check["passed"], check
    lods = body["gltf_summary"]["lods"]
    assert [l["triangles"] for l in lods] == [truth["body"]["lod0_triangles"], truth["body"]["lod1_triangles"]]
    mat = body["materials"][0]
    assert mat["base_color"].endswith("synth_body_cs.basecolor.png")
    assert mat["normal"].endswith("synth_body_nm.normal_gl.png")
    assert set(mat["channels"]) >= {"base_color", "roughness", "normal", "metallic"}
    manifest = json.loads((out / "manifest.json").read_text())
    by_name = {r["name"]: r for r in manifest["resources"]}
    assert by_name["characters/heroes/synth/synth_body_mesh"]["type"] in ("ebx", "res")
    ms = [r for r in manifest["resources"] if r.get("kind") == "MeshSet"]
    assert all(r["complete"] for r in ms)
    untouched = by_name["characters/heroes/synth/unmodified_reference"]
    assert untouched["has_payload"] is False and "original game" in untouched["note"]
    assert (out / "Extraction_Report.json").is_file()
    for tex in ("synth_body_cs.png", "synth_body_cs.dds", "synth_body_nm.png"):
        assert (out / "Extracted_Textures" / tex).is_file()


def test_glb_is_well_formed_and_skinned(tmp_path):
    report, truth, out = _run(tmp_path)
    m = next(m for m in report["meshes"] if m["output_name"] == "synth_body_mesh")
    doc, binary = read_glb(out / m["gltf"])
    assert doc["asset"]["version"] == "2.0" and len(doc["skins"]) == 1
    prim = doc["meshes"][0]["primitives"][0]
    w = accessor_array(doc, binary, prim["attributes"]["WEIGHTS_0"])
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)
    idx = accessor_array(doc, binary, prim["indices"])
    assert idx.max() < doc["accessors"][prim["attributes"]["POSITION"]]["count"]


def test_missing_chunk_is_reported_not_fabricated(tmp_path):
    report, truth, out = _run(tmp_path, drop_chunk=True)
    body = next(m for m in report["meshes"] if "body" in m["output_name"])
    assert body["status"] == "missing_dependency" and "gltf" not in body
    dep = next(d for d in report["missing_dependencies"] if d["kind"] == "chunk")
    assert dep["id"] == truth["dropped_chunk"] and "Chunk/Res Editor" in dep["how_to_provide"]
    assert report["outcome"] == "partial"
    assert not (out / "meshes" / "synth_body_mesh.glb").exists()


def test_extra_dir_supplies_missing_chunk(tmp_path):
    """A chunk exported from the game (Frosty Chunk/Res Editor: decompressed, named <guid>.chunk) fills the gap."""
    report, truth, _ = _run(tmp_path, name="drop", drop_chunk=True)
    need = next(d["id"] for d in report["missing_dependencies"] if d["kind"] == "chunk")
    # the synthetic body geometry is deterministic, so its LOD0 buffers can be regenerated exactly
    body = make_capsule(radius=0.25, height=1.2, y0=0.0, bones=(0, 1, 2))
    belt = make_capsule(radius=0.28, height=0.2, rings=4, segs=12, y0=0.55, bones=(1, 1, 1))
    vb, ib, _ = build_lod_buffers(SynthLod([SynthSection("body_mat", 0, body, DECL_SKINNED, STRIDES_SKINNED),
                                            SynthSection("belt_mat", 1, belt, DECL_SKINNED, STRIDES_SKINNED)]))
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / f"{need}.chunk").write_bytes(vb + ib)
    fixed = Extraction(tmp_path / "drop.fbmod", ExtractOptions(output=tmp_path / "fixed", extra_dirs=[extra])).run()
    body_m = next(m for m in fixed["meshes"] if "body" in m["output_name"])
    assert body_m["status"] == "decoded"
    assert any(e["used_as"] == f"chunk {need}" for e in fixed["extra_inputs"])
    assert not [d for d in fixed["missing_dependencies"] if d["kind"] == "chunk"]


def test_without_mesh_ebx_textures_bound_by_name(tmp_path):
    report, truth, out = _run(tmp_path, with_mesh_ebx=False)
    body = next(m for m in report["meshes"] if "body" in m["output_name"])
    assert body["mesh_asset_ebx"] is None
    mat = body["materials"][0]
    assert mat["base_color"] and "inferred" in mat["channels"]["base_color"]["basis"]


def test_without_skeleton_weights_become_vertex_group_attributes(tmp_path):
    report, truth, out = _run(tmp_path, with_skeleton=False)
    body = next(m for m in report["meshes"] if "body" in m["output_name"])
    assert body["skeleton"]["status"] == "missing" and "how_to_provide" in body["skeleton"]
    doc, binary = read_glb(out / body["gltf"])
    assert "skins" not in doc
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    assert "_BONE_INDICES_0" in attrs and "JOINTS_0" not in attrs


@pytest.mark.parametrize("version", [1, 3, 4])
def test_older_container_versions_extract(tmp_path, version):
    report, _, _ = _run(tmp_path, version=version)
    assert report["outcome"] == "complete"


def test_frosty_axis_mode_keeps_raw_coordinates(tmp_path):
    report, truth, out = _run(tmp_path, axis="frosty")
    body = next(m for m in report["meshes"] if "body" in m["output_name"])
    for check in compare_glb_to_truth(out / body["gltf"], truth["body"], "frosty"):
        assert check["passed"], check


def test_cli_inspect_and_extract(tmp_path, capsys):
    data, _ = synthetic_hero_mod()
    mod = tmp_path / "c.fbmod"
    mod.write_bytes(data)
    assert main(["inspect", str(mod)]) == 0
    assert "MeshSet" in capsys.readouterr().out
    assert main(["extract", str(mod), "-o", str(tmp_path / "o"), "--no-fbx", "--name", "body=Anakin_Body"]) == 0
    assert (tmp_path / "o" / "meshes" / "Anakin_Body.glb").is_file()
    assert main(["extract", str(tmp_path / "does_not_exist.fbmod"), "-o", str(tmp_path / "o2"), "--no-fbx"]) == 2


def test_unreadable_mod_writes_failure_report(tmp_path):
    bad = tmp_path / "bad.fbmod"
    bad.write_bytes(b"\x00" * 100)
    assert main(["extract", str(bad), "-o", str(tmp_path / "o"), "--no-fbx"]) == 2
    rep = json.loads((tmp_path / "o" / "Extraction_Report.json").read_text())
    assert rep["outcome"] == "unreadable" and rep["errors"]


BLENDER = find_blender()


@pytest.mark.skipif(BLENDER is None, reason="Blender not available (set $BLENDER)")
def test_blender_fbx_build_and_validation(tmp_path):
    data, truth = synthetic_hero_mod()
    mod = tmp_path / "b.fbmod"
    mod.write_bytes(data)
    report = Extraction(mod, ExtractOptions(output=tmp_path / "out", names={"synth_body": "Anakin_Body"})).run(BLENDER)
    body = next(m for m in report["meshes"] if m["output_name"] == "Anakin_Body")
    assert body["blender"]["status"] == "validated", [c for c in body["blender"]["validation"]["checks"] if not c["passed"]]
    assert (tmp_path / "out" / "fbx" / "Anakin_Body.fbx").stat().st_size > 1000


def test_hostile_resource_names_stay_inside_output(tmp_path):
    from fbmod_extractor.synth import ModEntry, build_fbmod
    mod = tmp_path / "evil.fbmod"
    mod.write_bytes(build_fbmod([ModEntry("ebx", "../../../escape", b"\xce\xd1\xb4\x0f" + b"\x00" * 60),
                                 ModEntry("chunk", "/abs/../../x", b"C" * 10)]))
    out = tmp_path / "out"
    Extraction(mod, ExtractOptions(output=out)).run()
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert all(out in p.parents or p == mod for p in written), written
