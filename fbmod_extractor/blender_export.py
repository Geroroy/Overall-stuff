"""Drive Blender (headless) to turn extracted GLBs into FBX files and validate them."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

SCRIPTS = Path(__file__).parent / "blender"


def _version_key(path: Path) -> tuple:
    digits = [int(x) for x in "".join(c if c.isdigit() else " " for c in str(path)).split()]
    return tuple(digits)


def _installed_blenders() -> list[Path]:
    """Blender executables in the standard install locations, newest version first."""
    found: list[Path] = []
    if sys.platform == "win32":
        roots = [os.environ.get(v) for v in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
        for root in filter(None, roots):
            found += Path(root).glob("Blender Foundation/Blender*/blender.exe")
        steam = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Steam/steamapps/common/Blender/blender.exe"
        if steam.is_file():
            found.append(steam)
    elif sys.platform == "darwin":
        found += Path("/Applications").glob("Blender*.app/Contents/MacOS/Blender")
        found += Path.home().glob("Applications/Blender*.app/Contents/MacOS/Blender")
    else:
        for pattern in ("/opt/blender*/blender", "/opt/*/blender-*/blender", "/snap/bin/blender"):
            found += Path("/").glob(pattern.lstrip("/"))
    return sorted((p for p in found if p.is_file()), key=_version_key, reverse=True)


def find_blender(explicit: str | None = None) -> str | None:
    """Explicit path, then $BLENDER, then PATH, then the standard install locations."""
    cands = [explicit, os.environ.get("BLENDER"), shutil.which("blender")] + [str(p) for p in _installed_blenders()]
    for cand in cands:
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _run(blender: str, script: str, args: list[str], log_path: Path, timeout: int) -> int:
    cmd = [blender, "-b", "--factory-startup", "--python-exit-code", "3", "--python", str(SCRIPTS / script), "--"] + args
    with open(log_path, "w") as f:
        # no console window popping up when started from the GUI on Windows
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=False,
                              creationflags=flags)
    return proc.returncode


def export_fbx(report: dict, out_dir: Path, blender: str, timeout: int = 900) -> None:
    """For every decoded mesh in ``report``: GLB -> .blend + .fbx, then re-import and validate."""
    fbx_dir = out_dir / "fbx"
    fbx_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for mesh in report["meshes"]:
        if not mesh.get("gltf"):
            continue
        name = mesh["output_name"]
        glb = out_dir / mesh["gltf"]
        fbx = fbx_dir / f"{name}.fbx"
        blend = out_dir / "blender" / f"{name}.blend"
        blend.parent.mkdir(parents=True, exist_ok=True)
        build_json = logs / f"{name}.build.json"
        result: dict = {"blender": blender}
        rc = _run(blender, "build_fbx.py", ["--glb", str(glb), "--fbx", str(fbx), "--blend", str(blend),
                                            "--summary", str(build_json)], logs / f"{name}.build.log", timeout)
        if rc != 0 or not fbx.is_file():
            result["status"] = "build_failed"
            result["log"] = (logs / f"{name}.build.log").relative_to(out_dir).as_posix()
            report["errors"].append(f"Blender FBX build failed for {name} (exit {rc}); see {result['log']}")
            mesh["blender"] = result
            continue
        result["fbx"] = fbx.relative_to(out_dir).as_posix()
        result["blend"] = blend.relative_to(out_dir).as_posix()
        result["build_summary"] = json.loads(build_json.read_text())
        val_json = logs / f"{name}.validation.json"
        rc = _run(blender, "validate_fbx.py", ["--fbx", str(fbx), "--expect", str(out_dir / mesh["blender_expectations"]),
                                               "--out", str(val_json)], logs / f"{name}.validate.log", timeout)
        if rc != 0 or not val_json.is_file():
            result["status"] = "validation_failed_to_run"
            report["errors"].append(f"Blender validation failed to run for {name} (exit {rc})")
        else:
            val = json.loads(val_json.read_text())
            result["validation"] = val
            result["status"] = "validated" if val["passed"] else "validation_failed"
            if not val["passed"]:
                failed = [c["check"] for c in val["checks"] if not c["passed"]]
                report["errors"].append(f"FBX validation failed for {name}: {failed}")
        mesh["blender"] = result
