import hashlib
import os
import struct
from pathlib import Path

import pytest

from fbmod_extractor.binary import FormatError
from fbmod_extractor.compression import UnsupportedCompression, compress, decompress, parse_block_headers
from fbmod_extractor.ebx import EbxPartition
from fbmod_extractor.fbmod import FbMod, UnsupportedModFormat
from fbmod_extractor.synth import ModEntry, build_fbmod


@pytest.mark.parametrize("codec", ["zstd", "none", "zlib"])
def test_block_compression_round_trip(codec):
    data = os.urandom(70000) + b"\x00" * 90000 + b"frostbite" * 5000
    blob = compress(data, codec)
    out, blocks = decompress(blob, len(data))
    assert out == data
    assert sum(b.decompressed_size for b in blocks) == len(data)
    assert all(b.decompressed_size <= 0x10000 for b in blocks)


def test_incompressible_block_is_stored():
    data = os.urandom(5000)
    blocks = parse_block_headers(compress(data))
    assert blocks[0].codec == "none"


def test_declared_size_mismatch_is_reported():
    blob = compress(b"abc" * 100)
    with pytest.raises(FormatError):
        decompress(blob, 301)


def test_dictionary_flag_is_unsupported():
    blob = bytearray(compress(b"x" * 1000))
    blob[0] |= 0x01  # top byte of decompressed size: "uses dictionary"
    with pytest.raises(UnsupportedCompression):
        decompress(bytes(blob))


def _entries():
    return [
        ModEntry("embedded", "Icon", b"\x89PNG fake", compress_payload=False),
        ModEntry("ebx", "a/b/asset", b"\xce\xd1\xb4\x0f" + b"\x00" * 60),
        ModEntry("ebx", "a/b/untouched", None),
        ModEntry("res", "a/b/res", b"R" * 300, res_type=0x6BDE20BA, res_rid=0xABCDEF, res_meta=b"\x01" * 16),
        ModEntry("chunk", "1a2b3c4d-0000-1111-2222-333344445555", b"C" * 1000),
    ]


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_container_versions(tmp_path, version):
    p = tmp_path / f"v{version}.fbmod"
    p.write_bytes(build_fbmod(_entries(), version=version))
    mod = FbMod(p)
    assert mod.version == version and mod.profile == "StarWarsBattlefrontII"
    assert mod.table_end == mod.data_offset and not mod.warnings
    names = [r.name for r in mod.resources]
    expect_untouched = "a/b/untouched" if version > 3 else None
    assert names == ["Icon", "a/b/asset", expect_untouched, "a/b/res", "1a2b3c4d-0000-1111-2222-333344445555"]
    res = mod.resources[3]
    assert res.res_type == 0x6BDE20BA and res.res_rid == 0xABCDEF and res.res_meta == b"\x01" * 16
    p_res = mod.payload(res)
    assert p_res.data == b"R" * 300 and p_res.compressed and p_res.sha1_matches
    assert mod.payload(mod.resources[0]).data == b"\x89PNG fake"
    assert mod.payload(mod.resources[2]) is None
    if version >= 5:
        assert "link" in mod.details


def test_legacy_and_bad_files(tmp_path):
    legacy = tmp_path / "legacy.fbmod"
    legacy.write_bytes(b"FBMODV2\x00" + b"\x00" * 64)
    with pytest.raises(UnsupportedModFormat):
        FbMod(legacy)
    bad = tmp_path / "bad.fbmod"
    bad.write_bytes(b"\x00" * 64)
    with pytest.raises(FormatError):
        FbMod(bad)
    good = build_fbmod(_entries())
    trunc = tmp_path / "trunc.fbmod"
    trunc.write_bytes(good[:-200])
    with pytest.raises(FormatError):
        FbMod(trunc)


def test_input_file_is_not_modified(tmp_path):
    p = tmp_path / "m.fbmod"
    p.write_bytes(build_fbmod(_entries()))
    before = hashlib.sha1(p.read_bytes()).hexdigest()
    mod = FbMod(p)
    for r in mod.resources:
        mod.payload(r)
    assert hashlib.sha1(p.read_bytes()).hexdigest() == before


# A real SWBF2 mod (Frosty format v1, 192 EBX assets) can be supplied through
# FBMOD_SAMPLES=<dir containing No.Visor.Mod.fbmod>; third-party mods are not
# redistributed with this repository.
REAL = Path(os.environ.get("FBMOD_SAMPLES", "/nonexistent")) / "No.Visor.Mod.fbmod"


@pytest.mark.skipif(not REAL.is_file(), reason="real sample mod not available")
def test_real_mod_no_visor():
    mod = FbMod(REAL)
    assert mod.version == 1 and mod.profile == "StarWarsBattlefrontII" and mod.game_version == 358418
    assert len(mod.resources) == 197 and mod.table_end == mod.data_offset
    ebx = mod.resources_of("ebx")
    assert len(ebx) == 192
    for r in ebx:
        p = mod.payload(r)
        assert p.sha1_matches and len(p.data) == r.original_size
        part = EbxPartition(p.data, name=r.name)
        assert not part.errors
        assert part.asset_name().lower() == r.name.lower()
        assert part.arrays_offset - 16 < part.walk_end <= part.arrays_offset


def test_truncated_resource_table(tmp_path):
    data = bytearray(build_fbmod(_entries()))
    # claim many more resources than exist
    head_end = data.index(b"synthetic mod for automated tests\x00") + len(b"synthetic mod for automated tests\x00") + 1
    struct.pack_into("<i", data, head_end, 50)
    p = tmp_path / "x.fbmod"
    p.write_bytes(bytes(data))
    with pytest.raises(FormatError):
        FbMod(p)
