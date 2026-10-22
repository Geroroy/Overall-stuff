"""Command line interface: ``fbmod-extract`` / ``python -m fbmod_extractor``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .binary import FormatError
from .fbmod import FbMod, UnsupportedModFormat

log = logging.getLogger("fbmod_extractor")


def _setup_logging(verbose: bool, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", handlers=handlers, force=True)


def cmd_inspect(args) -> int:
    from .restypes import res_type_name
    mod = FbMod(args.mod)
    if args.json:
        print(json.dumps({"summary": mod.summary(), "resources": [r.describe() for r in mod.resources]}, indent=1))
        return 0
    s = mod.summary()
    print(f"{s['file']}: Frosty mod format v{s['format_version']}, profile {s['profile']}, game rev {s['game_head_revision']}")
    for k, v in s["details"].items():
        print(f"  {k:12s} {v}")
    print(f"  resources    {s['resource_counts']} (with payload: {s['resources_with_payload']})")
    for r in mod.resources:
        extra = ""
        if r.type == "res":
            extra = f" [{res_type_name(r.res_type)} rid=0x{r.res_rid:016X}]"
        size = f"{r.original_size} B" if r.has_payload else "no payload"
        print(f"  {r.index:4d} {r.type:8s} {size:>12s}  {r.name}{extra}")
    for w in mod.warnings:
        print(f"  warning: {w}")
    return 0


def cmd_extract(args) -> int:
    from .blender_export import find_blender
    from .extract import Extraction, ExtractOptions

    out = Path(args.output)
    _setup_logging(args.verbose, out / "logs" / "extract.log")
    names = {}
    for item in args.name or []:
        if "=" not in item:
            log.error("--name expects SUBSTRING=OUTPUT_NAME, got %r", item)
            return 2
        k, v = item.split("=", 1)
        names[k] = v
    blender = None
    if not args.no_fbx:
        blender = find_blender(args.blender)
        if blender is None:
            log.warning("Blender not found (use --blender PATH or set $BLENDER); FBX files will not be built")
    opts = ExtractOptions(output=out, lods=args.lods, axis=args.axis, normals=args.normals,
                          skeleton=Path(args.skeleton) if args.skeleton else None,
                          write_raw=not args.no_raw, names=names, extra_dirs=[Path(d) for d in args.extra_dir])
    try:
        report = Extraction(Path(args.mod), opts).run(blender=blender)
    except (FormatError, UnsupportedModFormat, OSError) as exc:
        log.error("cannot read %s: %s", args.mod, exc)
        out.mkdir(parents=True, exist_ok=True)
        (out / "Extraction_Report.json").write_text(json.dumps(
            {"input": str(args.mod), "outcome": "unreadable", "errors": [str(exc)]}, indent=1))
        return 2
    s = report["summary"]
    log.info("outcome: %s | meshes decoded %d/%d | textures decoded %d/%d | missing dependencies %d | errors %d",
             report["outcome"], s["meshes_decoded"], s["meshsets_in_mod"], s["textures_decoded"],
             s["textures_in_mod"], s["missing_dependencies"], s["errors"])
    for m in report["meshes"]:
        b = m.get("blender", {})
        log.info("  mesh %-40s %-20s fbx=%s validation=%s", m["output_name"], m["status"], b.get("fbx"),
                 b.get("status"))
    log.info("report: %s", out / "Extraction_Report.json")
    return 0 if report["outcome"] in ("complete", "partial") else 1


def cmd_sdk_tables(args) -> int:
    from .sdk_tables import download_sdk_dll, write_render_format_table
    _setup_logging(args.verbose)
    dll = Path(args.dll) if args.dll else None
    if dll is None:
        if not args.download:
            log.error("give --dll PATH (FrostySdk/Profiles/StarWarsIISDK.dll) or --download")
            return 2
        dll = download_sdk_dll(Path(args.cache) / "StarWarsIISDK.dll")
    write_render_format_table(dll)
    return 0


def cmd_selftest(args) -> int:
    from .selftest import run_selftest
    out = Path(args.output)
    _setup_logging(args.verbose, out / "selftest.log")
    from .blender_export import find_blender
    blender = None if args.no_fbx else find_blender(args.blender)
    ok = run_selftest(out, blender)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fbmod-extract", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    i = sub.add_parser("inspect", help="list the resources in an .fbmod")
    i.add_argument("mod")
    i.add_argument("--json", action="store_true")
    i.set_defaults(func=cmd_inspect)

    e = sub.add_parser("extract", help="extract meshes/textures to glTF, FBX, PNG/DDS and write a report")
    e.add_argument("mod")
    e.add_argument("-o", "--output", required=True)
    e.add_argument("--lods", default="0", help="'0' (default), 'all', or comma separated LOD indices")
    e.add_argument("--axis", choices=["gltf", "frosty"], default="gltf",
                   help="gltf: convert Frostbite's left-handed space without mirroring (default); "
                        "frosty: raw coordinates like Frosty Editor's FBX export")
    e.add_argument("--normals", choices=["auto", "stored", "none"], default="auto",
                   help="auto: keep decoded normals only if they agree with the geometry")
    e.add_argument("--skeleton", help="SkeletonAsset EBX (raw or Frostbite-compressed) to bind bone weights")
    e.add_argument("--extra-dir", action="append", default=[], metavar="DIR",
                   help="folder of Frosty Editor exports (<guid>.chunk, <name>.res, <asset>.bin EBX, "
                        "<texture>.dds/.png) supplying game data the mod does not contain")
    e.add_argument("--name", action="append", metavar="SUBSTRING=NAME",
                   help="name outputs of meshes whose resource name contains SUBSTRING (e.g. body=Anakin_Body)")
    e.add_argument("--blender", help="path to the Blender executable (default: $BLENDER or PATH)")
    e.add_argument("--no-fbx", action="store_true", help="skip the Blender FBX step")
    e.add_argument("--no-raw", action="store_true", help="do not write decompressed raw payloads")
    e.add_argument("-v", "--verbose", action="store_true")
    e.set_defaults(func=cmd_extract)

    s = sub.add_parser("sdk-tables", help="regenerate the RenderFormat table from Frosty's SWBF2 SDK assembly")
    s.add_argument("--dll")
    s.add_argument("--download", action="store_true", help="fetch the DLL from the upstream FrostyToolsuite repo")
    s.add_argument("--cache", default=".cache")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_sdk_tables)

    t = sub.add_parser("selftest", help="build a synthetic SWBF2 mod and run the whole pipeline on it")
    t.add_argument("-o", "--output", required=True)
    t.add_argument("--blender")
    t.add_argument("--no-fbx", action="store_true")
    t.add_argument("-v", "--verbose", action="store_true")
    t.set_defaults(func=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        _setup_logging(False)
    try:
        return args.func(args)
    except BrokenPipeError:  # output piped into e.g. `head`
        return 0


if __name__ == "__main__":
    sys.exit(main())
