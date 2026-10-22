"""Frostbite resource type hashes (res type field of RES entries)."""

RES_TYPES = {
    0x6BDE20BA: "Texture", 0x8D9E6F01: "PcaComponentWeightsResource", 0x957C32B1: "AtlasTexture",
    0x49B156D4: "MeshSet", 0x36F3F2C0: "IShaderDatabase", 0x85AC783D: "EAClothAssetData",
    0xBA02FEE0: "MeshAdjancencyResource", 0x1091C8C5: "MorphTargetsResource", 0x85EA8656: "EAClothEntityData",
    0x59C79990: "FaceFxResource", 0x91043F65: "HavokPhysicsData", 0x30B4A553: "OccluderMesh",
    0x70C5CB3E: "EnlightenDatabase", 0xC6CD3286: "EnlightenStaticDatabase", 0xAD1AC4FD: "LargeParticleCloud",
    0xF04F0C81: "Dx11ShaderProgramDatabase", 0x41D57E10: "RenderTexture", 0x9D00966A: "UITtfFontFile",
    0x51A3C853: "AssetBank", 0x0DEAFE10: "IesResource", 0xEB228507: "MorphResource",
    0x52EE0D39: "PlayerPresetResource", 0x15E1F32E: "TerrainDecals", 0xA23E75DB: "TerrainLayerCombinations",
    0x1CA38E06: "VisualTerrain", 0x22FE8AC8: "TerrainStreamingTree", 0xEFC70728: "ZoneStreamerGrid",
    0xAFECB022: "CompiledLuaResource", 0x86521D6C: "LinearMediaAsset", 0x59CEEB57: "EnlightenShaderDatabase",
    0xD070EED1: "AnimTrackData", 0x5E862E05: "LocalizedStringResource", 0xCB8BCD07: "GtsoLut",
    0xC611F34A: "MeshEmitterResource", 0x387CA0AD: "EAClothData", 0xC664A660: "PamReplayResource",
    0xE36F0D59: "HavokClothPhysicsData", 0x9C4FAA17: "HeightfieldDecal", 0x5C4954A6: "DxTexture",
    0x76742DC8: "DelayLoadBundleResource", 0x24A019CC: "MorphMaterialResource", 0x319D8CD0: "RagdollResource",
    0x2D47A5FF: "SwfMovie", 0x10F0E5A1: "DxShaderProgramDatabase", 0x428EC9D4: "BundleRefTableResource",
    0xEF23407C: "FifaPhysicsResourceData",
}


def res_type_name(value: int | None) -> str:
    if value is None:
        return "?"
    return RES_TYPES.get(value, f"Unknown_0x{value:08X}")
