# Status: `Undies_Anakin.fbmod` → `Anakin_Body.fbx`, `Anakin_Head.fbx`

## Summary

The extractor is implemented, tested and runnable end to end (container →
EBX/RES/chunk decoding → mesh and texture decoding → glTF → Blender FBX →
re-import validation → report). The deliverables for **this specific mod were not
generated**, because the input file is not available in this environment. No
placeholder FBX files, textures or reports were created for it.

## Blocker 1: the input file is not available

* The repository was empty when work started and no `.fbmod` file existed
  anywhere in the container.
* The mod is published on Nexus Mods as "Anakin's Undies"
  (`nexusmods.com/starwarsbattlefront22017/mods/6013`). Nexus requires a
  logged-in account for downloads (and an adult-content opt-in for this mod):
  the files page returned HTTP 403 and the Nexus API HTTP 401 ("Please provide
  an authentication method"). No mirror was found.

**To resolve:** download `Undies_Anakin.fbmod` from the mod page with your Nexus
account and run (Blender 4.2+ installed):

```
pip install -e .
fbmod-extract inspect Undies_Anakin.fbmod
fbmod-extract extract Undies_Anakin.fbmod -o out --blender /path/to/blender \
    --name body=Anakin_Body --name head=Anakin_Head
```

Read `out/Extraction_Report.json`: `outcome`, `requested_outputs`,
`missing_dependencies`, and per mesh `lods[].sections[].checks` and
`blender.validation`. The `--name` substrings must match the MeshSet resource
names in the mod; `inspect` prints them. Adjust them if Anakin's meshes are
named differently (for example `--name anakin_01_body=Anakin_Body`).

## Blocker 2: data that may be missing from the mod itself

Which of these apply cannot be known until the file is inspected; each is
detected and reported automatically, with the exact identifier and the Frosty
Editor steps to export it (`--extra-dir`, see
[MISSING_DEPENDENCIES.md](MISSING_DEPENDENCIES.md)).

1. **Head mesh.** An underwear mod most likely replaces body meshes and
   textures only. If no MeshSet in the mod matches `head`,
   `requested_outputs` reports `not_in_mod` and `Anakin_Head.fbx` is not
   produced; the head would have to be exported from the game (MeshSet `.res`
   plus LOD `.chunk` files).
2. **Skeleton.** Mods almost never contain the `SkeletonAsset`. Without it, bone
   weights are kept exactly (glTF custom attributes, and vertex groups
   `bone_###` in the `.blend`), but FBX cannot carry weights without an
   armature. Export the skeleton EBX from Frosty and pass `--skeleton`.
3. **Unmodified textures.** Materials may reference textures the mod does not
   change (e.g. the head's skin); those are listed under
   `missing_dependencies` with asset names or partition GUIDs.
4. **Mesh EBX.** If the mod changes only the MeshSet and chunks (not the
   `MeshAsset` EBX), section-to-texture bindings are inferred from names and
   labelled `inferred`; exporting the `MeshAsset` as `.bin` gives exact links.

## What has been verified, and how

| Area | Evidence |
|---|---|
| `.fbmod` container v1–v5, payload table, SHA-1 | Real SWBF2 mod `No.Visor.Mod.fbmod` (v1, 197 resources; SHA-1 `078d83e4…`): table ends exactly at the data offset, all 192 EBX payloads decompress to their declared sizes, all stored SHA-1s match; synthetic v1–v5 containers |
| zstd block decompression (SWBF2, no dictionary) | same real mod (all blocks `0x0F70`) + round-trip tests |
| EBX decoding | all 192 real partitions decode with no errors; every `Name` field equals the resource name; every instance walk ends in the 16-byte padding before the arrays |
| MeshSet (SWBF2 layout), vertex streams, indices, inline LODs | synthetic MeshSets written by an independent writer; positions within half precision, UVs, normals, indices and weights recovered exactly |
| Tangent frames | decoder equals the game shader on known-answer vectors (all slot indices, both handedness bits) |
| Textures | BC1/BC3/BC5/RGBA decode identical to Pillow's DDS reader; DDS output re-reads bit-identically |
| glTF → Blender → FBX → Blender | synthetic body (2 sections, 2 LODs, skinned, 3-bone skeleton) and head: every validation check passes (counts, face indices, bbox/orientation, UV bounds, per-material triangles, embedded textures, bone bind positions within 2e-7) |
| Real-mod behaviour without geometry | `No.Visor.Mod.fbmod` → `outcome: no_geometry_decoded` with an explicit reason (EBX-only mod referencing 66 game partitions); no outputs fabricated; input file unchanged |

## Remaining technical risks once the file is available

* **MeshSet layout on real data.** No publicly downloadable SWBF2 mod with
  mesh data was found, so the MeshSet parser has not yet run on a real mod. Its
  layout follows Frosty's SWBF2 code path exactly; any deviation shows up
  immediately in the per-section checks (`within_meshset_bbox`,
  `indices_in_range`, `normal_agreement`) and as an error rather than as silent
  garbage.
* **Normals from Frosty imports.** This Frosty version's tangent encoder does
  not invert the game's decoder. If the mod's mesh was imported with such a
  version, the decoded normals will disagree with the geometry; the extractor
  then omits them (shading is derived from the real faces) and says so under
  `meshes[].normals.notes`. `--normals stored` keeps them anyway.
* **Texture channel conventions** (`_CS` colour + smoothness, `_NM` normal +
  metallic) come from community documentation and are labelled `convention`.
  The raw decoded PNG and the untouched DDS of every texture are always
  written, so these can be reinterpreted.
