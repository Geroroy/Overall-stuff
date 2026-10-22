# Format notes: Frosty `.fbmod` → SWBF2 meshes and textures

Research notes behind `fbmod_extractor`. Layouts were established by reading the
open-source [Frosty Tool Suite](https://github.com/CadeEvs/FrostyToolsuite)
(commit `e400dcb`, Sep 2023) and confirmed against a real Star Wars Battlefront II
mod where possible (see "Verification status" at the end). No Frosty code is
included in this repository (Frosty is CC BY-NC-ND 4.0); the extractor is an
independent Python implementation of the formats.

## 1. Container (`.fbmod`)

| Field | Type | Notes |
|---|---|---|
| magic | u64 | `0x01005954534F5246` (`FROSTY\x00\x01`) |
| version | u32 | 1–5 (5 = Frosty 1.0.6+/1.0.7) |
| dataOffset | i64 | start of the payload table |
| dataCount | i32 | number of payload table entries |
| profile | u8 length + chars | `StarWarsBattlefrontII` for SWBF2 |
| gameVersion | i32 | game head revision the mod was built against |
| details | NUL strings | title, author, category, version, description, link (v5+) |
| resourceCount | i32 | then the resource table |

Resource entry: `u8 type` (0 embedded, 1 ebx, 2 res, 3 chunk, 4 bundle), `i32
dataIndex` (-1 = no payload), `cstr name` (v≤3: only when a payload exists),
then for payload entries `sha1[20]`, `i64 originalSize`, `u8 flags` (0x01
inline, 0x02 add-to-chunks-bundle, 0x04 manifest chunk, 0x08 added asset),
`i32 handlerHash`, `cstr userData` (v3+). Bundle lists: v≤3 store two lists
(existing bundles, ignored; added bundles), v4+ one list (added bundles).
Type-specific tails: RES `u32 resType, u64 resRid, i32 metaLen, meta`; chunk
`u32 rangeStart, rangeEnd, logicalOffset, logicalSize, i32 h32, firstMip`;
bundle `cstr name, i32 superbundleHash`.

Payload table: `dataCount × (i64 offset, i64 size)`; offsets are relative to the
end of the table. The stored SHA-1 is the hash of the *stored* (compressed)
bytes. Entries with `dataIndex == -1` carry **no data**: the asset is only
listed (bundle membership); its content is the unmodified game asset.

Legacy formats `FBMOD`, `FBMODV1`, `FBMODV2` (a DbObject plus `_NN.archive`
files) predate the binary format and are detected but **not supported**.

## 2. Compression

Payloads (EBX, RES, chunks) use Frostbite block framing:
`u32 BE decompressedSize | u8 type | u8 flags | u16 BE compressedSize | data`,
blocks of at most 0x10000 decompressed bytes. Type 0x0F = zstd, 0x00 = stored,
0x02 = zlib, 0x09 = LZ4, 0x11/0x15 = Oodle. The low nibble of the flags byte
holds bits 16–19 of the compressed size. Frosty writes SWBF2 payloads as zstd
(`0x0F70`) **without** a dictionary (the zstd dictionary flag, top byte of the
size, is only used by FIFA titles). Oodle is proprietary and unsupported (not
used for SWBF2 mods). Embedded resources (icon, screenshots) are stored raw.

## 3. EBX (asset metadata), magic `0x0FB4D1CE`

SWBF2 partitions are self-describing: the header embeds type names (hashed with
djb2-xor: `h = h*33 ^ c`, seed 5381), field descriptors (`nameHash, flags,
classRef, dataOffset`) and class descriptors (`nameHash, fieldIndex,
fieldCount, alignment, flags, size`). Field kind = `((flags >> 1) >> 4) & 0x1F`.
Instances follow the string table; each is aligned to its class alignment,
exported instances are prefixed by their 16-byte GUID, classes with alignment
≠ 4 then have an 8-byte header. Field offsets are relative to the position after
the GUID (alignment ≠ 4) or that position − 8 (alignment 4). Arrays index an
array table (payloads after the instance data), strings are offsets into the
string table, pointers are `0` (null), `0x80000000 | importIndex` (reference into
another partition: partition GUID + instance GUID) or a 1-based instance index.

Relevant asset types:

* `SkinnedMeshAsset` / `RigidMeshAsset` / `CompositeMeshAsset` (base `MeshAsset`):
  `MeshSetResource` (u64 resource id of the MeshSet RES), `Materials[]` →
  `MeshMaterial` → `Shader.TextureParameters[] {ParameterName, Value → TextureAsset import}`.
  Section `MaterialId` indexes this list.
* `TextureAsset`: `Resource` (u64 id of the Texture RES).
* `SkeletonAsset`: `BoneNames[]`, `Hierarchy[]` (parent index), `LocalPose[]`
  (`LinearTransform` right/up/forward/trans, parent-relative).
* `MeshVariationDatabase` / `ObjectVariation`: per-variation material/texture overrides.

The extractor decodes every EBX in the mod and dumps it to JSON
(`raw/ebx/*.json`), then follows these links.

## 4. MeshSet resource (res type `0x49B156D4`), SWBF2 variant

```
MeshSet header
  AABB (2 × Vec3 padded to 16 B), i64 lodOffsets[6], i64 unused,
  i64 fullName, i64 name, u32 nameHash, u32 meshType (0 rigid, 1 skinned, 2 composite),
  u16 lodFadeDistanceFactors[12], u32 flags, u8 drawOrder, u8 userSlot, i16 subOrder,
  u16 lodCount, u16 sectionCount,
  skinned:   u16 boneCount, u16 bonePartCount, [i64 boneIndices(u16), i64 boneBoxes]
  composite: u16 partCount, u16 boneCount, [i64 partTransforms, i64 partBoxes]
  pad 16
LOD (one per lodOffsets[i])
  u32 meshType, u32 maxInstances, i32 sectionCount, i64 sections,
  5 × (i32 count, i64 offset) subset categories (opaque, transparent, decal, z-only, shadow),
  u32 layoutFlags, i32 indexBufferFormat (RenderFormat: R16_UINT=33, R32_UINT=46),
  u32 indexBufferSize, u32 vertexBufferSize, i32 adjacencySize, guid chunkId,
  u32 inlineDataOffset, i64 adjacency, i64 shaderDebugName, i64 name, i64 shortName,
  u32 nameHash, i64 unused, skinned: u32 boneCount + i64 boneIndexArray(u32)
  pad 16
Section (352 bytes)
  i64 ×2 runtime, i64 materialName, i32 materialId, u32 lightmapUvIndex,
  u32 primitiveCount, startIndex, vertexOffset, vertexCount,
  u8 vertexStride, u8 primitiveType, u8 ×2 legacy, u32 ×3 unused,
  u8 bonesPerVertex, u8 pad, u16 boneCount, i64 boneList(u16), u64 unused,
  2 × GeometryDeclaration { 16 × (u8 usage, u8 format, u8 offset, u8 stream),
                            16 × (u8 stride, u8 classification), u8 elementCount,
                            u8 streamCount, u16 pad },
  f32 texCoordRatios[6], u8 unknown[44]
```

Geometry for a LOD lives in its chunk (`chunkId`), or inline after the header
when `chunkId` is null (`resMeta[0..4]` = inline offset, `[4..8]` = size). The
vertex buffer comes first, the index buffer starts at `vertexBufferSize`. A
section's vertices start at `vertexOffset`; each non-empty stream of the
section's first declaration is a contiguous block of `vertexCount × stride`.
Indices are section-local; triangles start at `startIndex`.

Vertex elements used by SWBF2 characters (usage / format):
`Pos` Half4 or Float3 · `TangentSpace` UInt (packed quaternion) or
UByte4N/UShort4N (axis-angle) · `TexCoord0..7` Half2/Float2 · `BoneIndices(2)`
UByte4/UShort4 · `BoneWeights(2)` UByte4N · `Color0/1` UByte4N. For SWBF2 vertex
bone indices are **skeleton bone indices** (0x8000 marks procedural bones), not
indices into the section bone list.

### Tangent frames

`TangentSpace` (UInt) packs a quaternion with the "smallest three" scheme:
bits 31–22 component A (10 bits), 21–13 C (9 bits), 12–3 B (10 bits), 2–1 the
slot of the dropped largest component, bit 0 handedness. Stored values map
`[0, 2^n−1] → [−1/√2, 1/√2]`; the three stored components fill the remaining
slots in the order A, C, B. With `R` the rotation of that quaternion the game
shader uses **normal = R·x̂, tangent = −R·ŷ, binormal = ±(R·x̂ × R·ŷ)**.
`fbmod_extractor.vertex` implements this and is tested against known answers
generated from the game shader.

Finding: the importer-side encoder in this Frosty version does not invert that
decode (its matrix/quaternion conventions differ), so tangent frames in
Frosty-imported meshes may not match the geometry. The extractor therefore
checks decoded normals against the triangle geometry per section and drops them
(Blender recomputes shading from the real faces) when fewer than 90 % agree. The
choice is recorded in the report; nothing is invented.

## 5. Texture resource (res type `0x6BDE20BA`), SWBF2 variant (132 bytes)

`u32 mipOffset[2], u32 type, i32 pixelFormat (RenderFormat), u32 unknown, u16 flags
(0x2 sRGB), u16 width, height, depth, sliceCount, u8 mipCount, u8 firstMip, guid
chunkId, u32 mipSizes[15], u32 chunkSize, u32 assetNameHash, char group[16]`.
The chunk holds the mips in order, each mip containing all slices. The
`RenderFormat` numbering is game specific (BC1_UNORM = 54 … BC7_SRGB = 67); the
extractor ships the SWBF2 table in `fbmod_extractor/data/swbf2_render_formats.json`,
read from the metadata of Frosty's generated SDK assembly (regenerate with
`fbmod-extract sdk-tables`).

Texture roles follow SWBF2 naming conventions documented by the modding
community and are labelled `convention` in the report: `_CS` = base colour RGB +
smoothness A (exported roughness = 1 − A), `_NM` = normal RGB + metallic A,
`_AOSL` = occlusion pack (layout not verified, not wired). The green-channel
convention of normal maps is measured per mesh by comparing decoded tangent
frames with UV gradients; only when no reliable frames exist is DirectX style
(flip green for Blender/glTF) assumed and flagged.

## 6. Coordinates and units

Frostbite data is left-handed (X right, Y up, Z forward; Frosty's own viewport
uses `LookAtLH`/`PerspectiveFovLH`), in metres. The default export converts to
glTF's right-handed space with `(x, y, z) → (−x, y, z)` (mirror-free; winding
and tangent handedness are reversed accordingly). `--axis frosty` reproduces
Frosty Editor's FBX export, which writes raw coordinates into a right-handed
file (the model appears mirrored along X, but round-trips into Frosty).

## 7. Unsupported or unknown structures

| Structure | Status |
|---|---|
| Legacy `FBMOD`/`FBMODV1`/`FBMODV2` mods | detected, not decoded |
| Oodle-compressed blocks | not supported (proprietary; not used by SWBF2 mods) |
| Custom-handler resources (handlerHash ≠ 0, e.g. ShaderBlockDepot merges) | payload kept raw, not interpreted |
| RES types other than MeshSet/Texture (EACloth*, MeshAdjacency, OccluderMesh, Morph*, …) | listed in manifest + report `unsupported`, raw payload kept |
| Section `unknown[44]`, LOD/meshset unused 64-bit fields, adjacency buffers | skipped (not needed for geometry) |
| Second geometry declaration of each section | parsed, not used (Frosty also uses only the first) |
| Composite (part-based) meshes | geometry decoded; part transforms not applied |
| Procedural bones (index & 0x8000) | weights kept, need the mesh EBX's SkinnedProceduralAnimation to name them |
| Primitive types other than triangle lists | reported, not decoded |
| Texture arrays / cube maps | first slice exported only |
| BC6H / float formats | decoded and clipped to 8-bit PNG; DDS keeps the original data |
| EBX partitions without embedded type descriptors | header only (not used by SWBF2) |

## 8. Verification status

* Container, compression and EBX: verified on a real SWBF2 mod
  (`No.Visor.Mod.fbmod`, format v1, 192 EBX): every payload decompresses to its
  declared size, every stored SHA-1 matches, every EBX decodes and its `Name`
  matches the resource name, and each instance walk ends inside the 16-byte
  padding before the array region.
* MeshSet, vertex streams, tangent frames, textures, glTF/FBX: verified end to
  end on synthetic SWBF2 data written by an independent writer
  (`fbmod_extractor/synth.py`) and round-tripped through Blender 4.2. The
  tangent decoder matches the game shader's output exactly on known answers.
  No publicly downloadable SWBF2 mod containing a MeshSet was found, so the
  MeshSet layout has not yet been exercised on a real mod; the per-section
  integrity checks in the report (positions inside the MeshSet's own bounding
  box, index range, normal/geometry agreement) are designed to expose any
  layout mismatch immediately.
