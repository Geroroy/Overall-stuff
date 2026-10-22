# Providing original game data

A Frosty mod only contains the assets it changes. When a mod changes, for
example, a MeshSet header but not the vertex buffers, or a material that points
at unmodified textures, the missing pieces exist only in your Star Wars
Battlefront II installation. The extractor never fills such gaps with invented
data: it lists each one in `Extraction_Report.json` under
`missing_dependencies` (with the exact chunk GUID, asset name or EBX partition
GUID, what it is needed for, and how to provide it), and leaves the affected
mesh/LOD/texture out.

You can supply the data yourself with Frosty Editor (with SWBF2 loaded) and
re-run with `--extra-dir <folder>`. Put all exports in one folder; the file
names matter:

| Missing item (`kind`) | How to export it in Frosty Editor | File name to use |
|---|---|---|
| `chunk` (mesh LOD vertex/index buffers, texture pixels) | **View → Chunk/Res Editor**, Chunks tab, filter by the GUID from the report, right-click → **Export** | `<guid>.chunk` (the default) |
| MeshSet or Texture resource not in the mod | **View → Chunk/Res Editor**, Resources tab, find the resource by name, right-click → **Export** | `<resource short name>.res` (the default) |
| `texture` (material references a texture the mod does not change) | **Data Explorer**, open the texture asset, right-click → **Export** → DDS | `<texture asset short name>.dds` |
| texture identified only by partition GUID | open the mesh asset in Frosty, follow the material's `TextureParameters` to the texture asset, export it as **Binary File (.bin)** (links GUID → name) and as DDS (pixels) | `<asset>.bin`, `<asset>.dds` |
| mesh EBX (exact material ↔ section ↔ texture links) | **Data Explorer**, open the `MeshAsset`, right-click → **Export** → **Binary File** | `<asset>.bin` |
| skeleton (bone names, hierarchy, bind pose) | **Data Explorer**, open the character's `SkeletonAsset`, export as **Binary File** | any name; pass it with `--skeleton <file.bin>` |

Example:

```
fbmod-extract extract Undies_Anakin.fbmod -o out \
    --name body=Anakin_Body --name head=Anakin_Head \
    --extra-dir frosty_exports --skeleton frosty_exports/anakin_skeleton.bin
```

Every file used is listed under `extra_inputs` in the report, and any binding
that relies on it is labelled with its source. Files that duplicate something
the mod already contains are ignored (the mod's version wins).

## A head or body mesh that the mod does not change

If you ask for an output (`--name head=Anakin_Head`) and no MeshSet in the mod
matches, `requested_outputs` says `not_in_mod`: the mod does not modify that
mesh, so its geometry is the unmodified game asset. To convert the original
asset with the same pipeline:

1. Export its MeshSet resource (`<name>.res`) into the extra folder and run the
   extractor once; the report's `missing_dependencies` then lists the exact LOD
   chunk GUIDs that MeshSet needs.
2. Export those chunks (`<guid>.chunk`) into the same folder and run again.
