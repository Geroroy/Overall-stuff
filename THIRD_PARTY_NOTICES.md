# Third-party notices

## Frosty Tool Suite

The `.fbmod`, EBX, MeshSet and Texture layouts implemented here were learned by
reading the source of the [Frosty Tool Suite](https://github.com/CadeEvs/FrostyToolsuite)
(© its authors, licensed CC BY-NC-ND 4.0). This repository contains no Frosty
source code, binaries or assets; `fbmod_extractor` is an independent Python
implementation written for interoperability.

`fbmod_extractor/data/swbf2_render_formats.json` lists the name/value pairs of
Star Wars Battlefront II's `RenderFormat` enumeration (facts about the game's
type system). They were read from the metadata of Frosty's generated
`StarWarsIISDK.dll`; the file records that DLL's SHA-1, and
`fbmod-extract sdk-tables` regenerates it from a local copy.

The known-answer vectors in `tests/test_vertex.py` are numeric outputs of the
game's tangent-frame decoding, used only to test this implementation.

## Python dependencies

* numpy (BSD-3-Clause)
* zstandard / python-zstandard (BSD-3-Clause)
* Pillow (MIT-CMU)
* optional: dnfile (MIT), lz4 (BSD-3-Clause), pytest (MIT)

## Blender

FBX files are produced by running Blender (GPL-2.0-or-later) as an external
program; no Blender code is distributed here. The scripts in
`fbmod_extractor/blender/` use Blender's Python API.

## Game content

Star Wars Battlefront II is © Electronic Arts / Lucasfilm. Mods and game assets
are not included in this repository; extracted output is for personal use
within the terms of the game's EULA and the mod author's permissions.
