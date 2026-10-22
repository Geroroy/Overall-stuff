import uuid

import numpy as np
import pytest
from PIL import Image

from fbmod_extractor.binary import NULL_GUID, FormatError
from fbmod_extractor.ebx import EbxPartition, name_hash
from fbmod_extractor.mesh import choose_index_size, decode_section
from fbmod_extractor.meshset import parse_meshset
from fbmod_extractor.synth import (DECL_RIGID, DECL_SKINNED, STRIDES_RIGID, STRIDES_SKINNED, EbxObj, SynthLod,
                                   SynthSection, build_ebx, build_meshset, build_texture, encode_texture,
                                   make_capsule, make_texture_image)
from fbmod_extractor.texture import decode_to_image, parse_texture_header, split_mips, write_dds


# -- EBX ------------------------------------------------------------------------------

def test_ebx_round_trip_fields_arrays_imports():
    tex_part, tex_inst = str(uuid.uuid4()), str(uuid.uuid4())
    shader = (str(uuid.uuid4()), str(uuid.uuid4()))
    mat = EbxObj("MeshMaterial", {"Shader": {"Shader": shader, "TextureParameters": [
        {"ParameterName": "_CS", "Value": (tex_part, tex_inst)}]}})
    mesh = EbxObj("SkinnedMeshAsset", {"Name": "a/b/mesh", "Materials": [mat], "MeshSetResource": 0x1122334455667788})
    data = build_ebx([mesh, mat], [shader, (tex_part, tex_inst)], "11111111-2222-3333-4444-555555555555")
    part = EbxPartition(data)
    assert not part.errors
    assert part.partition_guid == "11111111-2222-3333-4444-555555555555"
    assert part.root_type == "SkinnedMeshAsset"
    assert part.class_chain("SkinnedMeshAsset")[:3] == ["SkinnedMeshAsset", "MeshAsset", "MeshBaseAsset"]
    root = part.root.fields
    assert root["Name"] == "a/b/mesh" and root["MeshSetResource"] == 0x1122334455667788
    m = part.deref(root["Materials"][0])
    assert m.type_name == "MeshMaterial" and m.instance_guid == mat.guid
    tp = m.fields["Shader"]["TextureParameters"][0]
    assert tp == {"ParameterName": "_CS", "Value": {"$import": {"partition": tex_part, "instance": tex_inst}}}


def test_ebx_struct_arrays_and_alignment():
    def lt(x):
        return {"right": {"x": 1, "y": 0, "z": 0}, "up": {"x": 0, "y": 1, "z": 0},
                "forward": {"x": 0, "y": 0, "z": 1}, "trans": {"x": x, "y": 2 * x, "z": 3 * x}}
    sk = EbxObj("SkeletonAsset", {"Name": "skel", "BoneNames": ["a", "b", "c"], "Hierarchy": [-1, 0, 1],
                                  "LocalPose": [lt(1.0), lt(2.0), lt(3.0)]})
    part = EbxPartition(build_ebx([sk], [], str(uuid.uuid4())))
    f = part.root.fields
    assert f["BoneNames"] == ["a", "b", "c"] and f["Hierarchy"] == [-1, 0, 1]
    assert [p["trans"]["y"] for p in f["LocalPose"]] == [2.0, 4.0, 6.0]


def test_ebx_rejects_garbage():
    with pytest.raises(FormatError):
        EbxPartition(b"\x00" * 64)


def test_name_hash_matches_frostbite_djb2_xor():
    assert name_hash("") == 5381
    assert name_hash("a") == ((5381 * 33) ^ ord("a")) & 0xFFFFFFFF


# -- MeshSet -------------------------------------------------------------------------

def _meshset(inline=False, index_size=2):
    body = make_capsule()
    head = make_capsule(radius=0.1, height=0.2, y0=1.5)
    head.joints = head.weights = None
    lods = [SynthLod([SynthSection("skin", 0, body, DECL_SKINNED, STRIDES_SKINNED, bone_list=(0, 1, 2)),
                      SynthSection("eyes", 1, head, DECL_RIGID, STRIDES_RIGID, categories=("opaque", "shadow"))],
                     index_size=index_size, inline=inline)]
    data, meta, chunks = build_meshset("chars/x/x_mesh", 1, lods, ((-1, -1, -1), (1, 2, 1)), bone_count=3)
    return data, meta, chunks, lods, body, head


@pytest.mark.parametrize("inline,index_size", [(False, 2), (False, 4), (True, 2)])
def test_meshset_parse_and_decode(inline, index_size):
    data, meta, chunks, lods, body, head = _meshset(inline, index_size)
    ms = parse_meshset(data, meta)
    assert not ms.warnings
    assert ms.full_name == "chars/x/x_mesh" and ms.name == "x_mesh" and ms.mesh_type == 1
    assert ms.bone_count == 3 and ms.bone_indices == [0, 1, 2]
    lod = ms.lods[0]
    assert len(lod.sections) == 2 and lod.bone_indices == [0, 1, 2]
    assert lod.section_categories(1) == ["opaque", "shadow"]
    s0, s1 = lod.sections
    assert s0.material_name == "skin" and s0.vertex_count == len(body.positions)
    assert s0.bone_list == [0, 1, 2]
    assert [e.usage_name for e in s0.decl.elements] == ["Pos", "BoneIndices", "BoneWeights", "TangentSpace", "TexCoord0"]
    if inline:
        assert lod.chunk_id == NULL_GUID and lod.inline_data is not None
        buf = lod.inline_data
    else:
        buf = chunks[lod.chunk_id]
    size, _ = choose_index_size(buf, lod)
    assert size == index_size
    d0 = decode_section(buf, ms, lod, s0, size)
    np.testing.assert_allclose(d0.positions, body.positions, atol=2e-3)  # half precision
    np.testing.assert_array_equal(d0.indices, body.indices)
    np.testing.assert_allclose(d0.uvs["TexCoord0"], body.uvs, atol=1e-3)
    assert np.einsum("ij,ij->i", d0.normals, body.normals).min() > 0.999
    assert d0.checks["within_meshset_bbox"] and d0.checks["normal_agreement"] == 1.0
    np.testing.assert_allclose(d0.weights[:, :2].sum(axis=1), 1.0, atol=1e-6)
    d1 = decode_section(buf, ms, lod, s1, size)
    np.testing.assert_allclose(d1.positions, head.positions, atol=1e-6)  # float32
    assert d1.joints is None


def test_meshset_rejects_garbage():
    with pytest.raises(FormatError):
        parse_meshset(b"\x00" * 512)


# -- textures -----------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["BC1_UNORM", "BC3_UNORM", "BC5_UNORM", "R8G8B8A8_UNORM"])
def test_texture_decode_and_dds(tmp_path, fmt):
    img = make_texture_image("cs")
    mip0 = encode_texture(img, fmt)
    header = parse_texture_header(build_texture(fmt, 64, 64, mip0, str(uuid.uuid4())))
    assert header.format_name == fmt and header.width == 64 and not header.warnings
    mips, layout = split_mips(header, mip0)
    out = decode_to_image(header, mips[0])
    assert out.size == (64, 64)
    if fmt == "R8G8B8A8_UNORM":
        assert out.tobytes() == img.tobytes()
    dds = tmp_path / "t.dds"
    write_dds(dds, header, mips)
    reread = Image.open(dds)
    reread.load()
    assert np.array_equal(np.asarray(reread.convert(out.mode)), np.asarray(out))


def test_texture_header_validation():
    bad = build_texture("BC1_UNORM", 64, 64, b"\x00" * 100, str(uuid.uuid4()))  # wrong mip size
    h = parse_texture_header(bad)
    assert h.warnings and "mip0 size" in h.warnings[0]
    with pytest.raises(FormatError):
        parse_texture_header(b"\x00" * 10)


def test_vertex_layout_mismatch_prefers_layout_that_validates():
    """Frosty's importer packs elements in declaration order regardless of declared offsets."""
    from fbmod_extractor.synth import build_lod_buffers
    from fbmod_extractor.vertex import USAGE
    mesh = make_capsule(radius=0.2, height=0.5, y0=0.2)
    mesh.joints = mesh.weights = None
    declared = [(USAGE["Pos"], 0x03, 0, 0), (USAGE["TexCoord0"], 0x06, 16, 0), (USAGE["TangentSpace"], 0x20, 12, 0)]
    packed = [(USAGE["Pos"], 0x03, 0, 0), (USAGE["TexCoord0"], 0x06, 12, 0), (USAGE["TangentSpace"], 0x20, 16, 0)]
    lod = SynthLod([SynthSection("m", 0, mesh, declared, [20])])
    data, meta, chunks = build_meshset("x/y_mesh", 0, [lod], ((-1, -1, -1), (1, 1, 1)))
    vb, ib, _ = build_lod_buffers(SynthLod([SynthSection("m", 0, mesh, packed, [20])], chunk_id=lod.chunk_id))
    ms = parse_meshset(data, meta)
    d = decode_section(vb + ib, ms, ms.lods[0], ms.lods[0].sections[0], 2)
    assert d.checks["vertex_layout"] == "packed"
    np.testing.assert_allclose(d.uvs["TexCoord0"], mesh.uvs, atol=1e-3)
    assert d.checks["normal_agreement"] == 1.0


def test_mesh_variation_database_texture_overrides():
    from fbmod_extractor.materials import variation_entries
    mesh = (str(uuid.uuid4()), str(uuid.uuid4()))
    mat = (mesh[0], str(uuid.uuid4()))
    tex = (str(uuid.uuid4()), str(uuid.uuid4()))
    db = EbxObj("MeshVariationDatabase", {"Name": "mvdb", "Entries": [
        {"Mesh": mesh, "VariationAssetNameHash": 1234, "Materials": [
            {"Material": mat, "TextureParameters": [{"ParameterName": "_CS", "Value": tex}]}]}]})
    part = EbxPartition(build_ebx([db], [mesh, mat, tex], str(uuid.uuid4())))
    assert not part.errors and part.root_type == "MeshVariationDatabase"
    entries = variation_entries(part)
    assert entries == [{"mesh": {"partition": mesh[0], "instance": mesh[1]}, "variation_name_hash": 1234,
                        "materials": [{"material": {"partition": mat[0], "instance": mat[1]},
                                       "texture_parameters": [{"parameter": "_CS", "texture_partition": tex[0],
                                                               "texture_instance": tex[1]}],
                                       "vector_parameters": []}]}]
