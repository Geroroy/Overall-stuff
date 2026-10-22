# fbmod-extractor

Converts **Star Wars Battlefront II (2017)** Frosty mods (`.fbmod`) into
Blender-compatible assets: glTF 2.0 meshes (geometry, normals, UVs, vertex
colours, skin weights, materials), FBX files built with Blender's own exporter,
DDS/PNG textures, a machine-readable resource manifest, and an extraction report
that states exactly what was decoded, what was not, and why.

It only exports data that is actually decoded from the mod (or from game data
you supply). Missing geometry, UVs, textures or bones are reported, never
synthesised.

## Desktop app (drop a .fbmod, get FBX)

`FBMOD-Converter` is a small window app: drop a `.fbmod` on it (or on its icon)
and it converts immediately, finding Blender automatically. Windows builds come
from the **Build FBMOD Converter (Windows)** workflow (Actions tab → artifact
`FBMOD-Converter-windows`) or `packaging\build_windows.bat`. From source:
`pip install -e .[gui]` then `fbmod-converter`. User guide (Korean):
[docs/APP_KO.md](docs/APP_KO.md).

## Install

```
pip install -e .            # numpy, zstandard, Pillow
pip install -e .[test]      # + pytest
```

Blender (tested with 4.2 LTS) is needed for the FBX step; point to it with
`--blender /path/to/blender` or `$BLENDER`. Without Blender you still get glTF,
textures, manifest and report.

## Use

```
fbmod-extract inspect Undies_Anakin.fbmod                # list resources
fbmod-extract extract Undies_Anakin.fbmod -o out \
    --name body=Anakin_Body --name head=Anakin_Head    # name outputs by resource substring
```

Useful options: `--lods all` (default `0`), `--extra-dir DIR` (game data exported
with Frosty Editor, see [docs/MISSING_DEPENDENCIES.md](docs/MISSING_DEPENDENCIES.md)),
`--skeleton file.bin` (SkeletonAsset EBX for a real armature), `--axis frosty`
(raw coordinates like Frosty Editor's own FBX export), `--normals auto|stored|none`,
`--no-fbx`, `-v`.

Self test (builds a synthetic SWBF2 mod and runs the whole pipeline, including
Blender if available): `fbmod-extract selftest -o selftest_out`.

## Output

```
out/
  Extraction_Report.json   outcome, per-mesh/LOD/section decode results and integrity checks,
                           texture/material bindings (with their basis), missing dependencies,
                           unsupported resources, Blender validation results
  manifest.json            every resource in the mod: type, payload/compression/SHA-1 status,
                           decoded EBX type and links, MeshSet/Texture headers, chunk dependencies
                           and whether each is present in the mod
  meshes/<name>.glb        glTF 2.0, one node per exported LOD, one primitive per section
  meshes/<name>.expect.json  values the Blender round trip must reproduce
  fbx/<name>.fbx           Blender FBX export (textures embedded)
  blender/<name>.blend     Blender scene (keeps bone-index vertex groups when no skeleton is known)
  Extracted_Textures/      <tex>.dds (original block data), <tex>.png (decoded mip 0), derived
                           channel images (basecolor, roughness = 1-smoothness, metallic, normal_gl)
  raw/                     decompressed EBX (+ decoded JSON), RES (+ meta), chunks, embedded images
  logs/                    extract.log, Blender build/validation logs and results
```

`outcome` is `complete`, `partial` (something was decoded but dependencies are
missing or errors occurred), `no_geometry_decoded` (with `no_geometry_reason`)
or `unreadable`.

## What is checked

Per decoded section: positions finite and inside the MeshSet's own bounding
box, indices in range, degenerate/unreferenced counts, UV ranges, skin-weight
sums, and agreement of decoded normals with the triangle geometry (which also
decides winding). Per FBX (re-imported in a fresh Blender scene): object
present, non-zero vertices/triangles, valid face indices, triangle-only faces,
vertex/triangle counts equal to the decoded data, bounding box equal to the
decoded data after the axis conversion (scale and orientation), UV bounds,
triangles per material slot (no missing sections), textures present with pixel
data, and armature bone names/bind positions when a skeleton is bound.

## How it works

`fbmod.py` container → `compression.py` Frostbite blocks (zstd/zlib/LZ4) →
`ebx.py` self-describing EBX → `meshset.py` + `mesh.py` + `vertex.py` geometry →
`texture.py` textures → `materials.py` material/texture links → `gltf.py` →
`blender/*.py` FBX build and validation. Format details, conventions and the
list of unsupported structures: [docs/FORMAT_NOTES.md](docs/FORMAT_NOTES.md).
Status of the `Undies_Anakin.fbmod` conversion: [docs/STATUS.md](docs/STATUS.md).

## Tests

```
pytest                                   # unit + end-to-end tests on synthetic SWBF2 data
BLENDER=/path/to/blender pytest          # + FBX build/validation test
FBMOD_SAMPLES=/dir/with/No.Visor.Mod.fbmod pytest   # + checks on a real SWBF2 mod
```

## License notes

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No Frosty Tool Suite
code is included; mods and game assets are never committed (`*.fbmod` is
git-ignored).
