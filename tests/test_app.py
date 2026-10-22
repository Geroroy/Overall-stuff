from pathlib import Path

from fbmod_extractor.app_core import ConvertRequest, convert, parse_names
from fbmod_extractor.synth import synthetic_hero_mod


def test_parse_names():
    assert parse_names("body=Anakin_Body, head = Anakin_Head;x=") == {"body": "Anakin_Body", "head": "Anakin_Head"}
    assert parse_names("") == {}


def test_default_output_folder_next_to_mod(tmp_path):
    req = ConvertRequest(mod=tmp_path / "My Mod.fbmod")
    assert req.output_dir() == tmp_path / "My Mod_converted"


def test_convert_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("fbmod_extractor.app_core.find_blender", lambda explicit=None: None)
    data, _ = synthetic_hero_mod()
    mod = tmp_path / "hero.fbmod"
    mod.write_bytes(data)
    res = convert(ConvertRequest(mod=mod, names={"synth_body": "Anakin_Body", "head_nope": "Anakin_Head"}))
    assert res.ok and res.output == tmp_path / "hero_converted"
    assert [p.name for p in res.glb_files] == ["Anakin_Body.glb", "synth_head_mesh.glb"]
    assert res.fbx_files == []  # no Blender
    assert "Anakin_Body" in res.summary and "Blender를 찾지 못해" in res.summary
    assert "Anakin_Head" in res.summary  # requested but not in the mod
    assert (res.output / "logs" / "extract.log").stat().st_size > 0


def test_convert_unreadable(tmp_path):
    bad = tmp_path / "bad.fbmod"
    bad.write_bytes(b"not a mod")
    res = convert(ConvertRequest(mod=bad))
    assert not res.ok and res.report is None and "읽을 수 없습니다" in res.summary
    res = convert(ConvertRequest(mod=Path(tmp_path / "missing.fbmod")))
    assert not res.ok
