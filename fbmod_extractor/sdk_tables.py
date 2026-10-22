"""Extract the game's ``RenderFormat`` enum from Frosty's pre-generated SWBF2 SDK.

Texture and index-buffer formats are stored in SWBF2 resources as integer
values of the game's ``RenderFormat`` enum. The numbering is game-specific and
is not stored in the resources themselves. Frosty Tool Suite ships the enum in
``FrostySdk/Profiles/StarWarsIISDK.dll`` (a .NET assembly generated from the
game's type info), so this module reads the enum's constant values straight
from that assembly's metadata tables.

FrostyToolsuite is licensed CC BY-NC-ND 4.0: this project does not ship the DLL
or any Frosty code. The committed ``data/swbf2_render_formats.json`` holds only
the enum's name/value pairs (facts about the game) and records the DLL's SHA-1;
it can be regenerated with ``fbmod-extract sdk-tables --dll <path>``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

FROSTY_COMMIT = "e400dcb283ad743071263ea7b34ff86780ffe910"
FROSTY_SDK_URL = (
    f"https://github.com/CadeEvs/FrostyToolsuite/raw/{FROSTY_COMMIT}/FrostySdk/Profiles/StarWarsIISDK.dll"
)
RENDER_FORMATS_JSON = Path(__file__).parent / "data" / "swbf2_render_formats.json"


def read_enum(dll_path: str | Path, enum_name: str = "RenderFormat") -> dict[str, int]:
    try:
        import dnfile  # type: ignore
    except ImportError as exc:
        raise SystemExit("reading the SDK needs the optional 'dnfile' package (pip install dnfile)") from exc

    pe = dnfile.dnPE(str(dll_path))
    md = pe.net.mdtables
    consts: dict[int, bytes] = {}
    for c in md.Constant.rows:
        if c.Parent.table and c.Parent.table.name == "Field":
            v = c.Value
            consts[c.Parent.row_index] = bytes(v.value if hasattr(v, "value") else v)
    for t in md.TypeDef.rows:
        if str(t.TypeName) != enum_name:
            continue
        values: dict[str, int] = {}
        for f in t.FieldList:
            raw = consts.get(f.row_index)
            if raw is not None and len(raw) == 4:
                values[str(f.row.Name)] = struct.unpack("<i", raw)[0]
        return values
    raise KeyError(f"enum {enum_name} not found in {dll_path}")


def build_render_format_table(dll_path: str | Path) -> dict:
    dll_path = Path(dll_path)
    values = read_enum(dll_path, "RenderFormat")
    return {
        "game": "StarWarsBattlefrontII",
        "enum": "RenderFormat",
        "source": {
            "file": dll_path.name,
            "sha1": hashlib.sha1(dll_path.read_bytes()).hexdigest(),
            "note": "Name/value pairs read from FrostyToolsuite's generated SDK assembly "
                    f"(github.com/CadeEvs/FrostyToolsuite @ {FROSTY_COMMIT}).",
        },
        "values": dict(sorted(values.items(), key=lambda kv: kv[1])),
    }


def write_render_format_table(dll_path: str | Path, out: Path = RENDER_FORMATS_JSON) -> Path:
    table = build_render_format_table(dll_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1) + "\n")
    log.info("wrote %d RenderFormat values to %s", len(table["values"]), out)
    return out


def download_sdk_dll(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", FROSTY_SDK_URL)
    with urllib.request.urlopen(FROSTY_SDK_URL, timeout=120) as resp:  # noqa: S310 - fixed https URL
        dest.write_bytes(resp.read())
    return dest


def load_render_formats() -> dict[int, str]:
    """value -> name without the ``RenderFormat_`` prefix (e.g. 66 -> 'BC7_UNORM')."""
    table = json.loads(RENDER_FORMATS_JSON.read_text())
    return {v: k.removeprefix("RenderFormat_") for k, v in table["values"].items()}
