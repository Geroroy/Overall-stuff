"""End-to-end self test on a synthetic SWBF2 mod with known ground truth."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from .extract import Extraction, ExtractOptions
from .gltf import accessor_array, convert_positions, read_glb
from .synth import synthetic_hero_mod

log = logging.getLogger(__name__)


def compare_glb_to_truth(glb: Path, truth_mesh: dict, axis: str) -> list[dict]:
    """Check LOD0 primitives against the generator's arrays (half-float precision for positions)."""
    doc, binary = read_glb(glb)
    results = []
    lod0 = doc["meshes"][0]["primitives"]
    for k, prim in enumerate(lod0):
        pos = accessor_array(doc, binary, prim["attributes"]["POSITION"])
        want = convert_positions(truth_mesh["positions"][k], axis)
        err = float(np.abs(pos - want).max())
        tol = 2e-3 if "uvs" in truth_mesh else 1e-6
        results.append({"check": f"primitive {k} positions", "passed": err < tol, "max_error": err})
        if "uvs" in truth_mesh:
            uv = accessor_array(doc, binary, prim["attributes"]["TEXCOORD_0"])
            uerr = float(np.abs(uv - truth_mesh["uvs"][k]).max())
            results.append({"check": f"primitive {k} UVs", "passed": uerr < 1e-3, "max_error": uerr})
        if "normals" in truth_mesh and "NORMAL" in prim["attributes"]:
            n = accessor_array(doc, binary, prim["attributes"]["NORMAL"])
            want_n = convert_positions(truth_mesh["normals"][k], axis)
            dots = np.einsum("ij,ij->i", n, want_n)
            results.append({"check": f"primitive {k} normals", "passed": float(dots.min()) > 0.99,
                            "min_dot": float(dots.min())})
        if "indices" in truth_mesh:
            idx = accessor_array(doc, binary, prim["indices"]).reshape(-1, 3)
            src = truth_mesh["indices"][k].astype(np.uint32)
            same = np.array_equal(np.sort(idx, axis=1), np.sort(src, axis=1))
            results.append({"check": f"primitive {k} triangles (as vertex sets)", "passed": bool(same)})
    return results


def run_selftest(out: Path, blender: str | None) -> bool:
    out.mkdir(parents=True, exist_ok=True)
    data, truth = synthetic_hero_mod()
    mod_path = out / "synthetic_hero.fbmod"
    mod_path.write_bytes(data)
    rep = Extraction(mod_path, ExtractOptions(output=out / "extracted", lods="all",
                                              names={"synth_body": "Synth_Body", "synth_head": "Synth_Head"})).run(blender)
    checks = [{"check": "extraction outcome complete", "passed": rep["outcome"] == "complete",
               "detail": rep["outcome"]}]
    meshes = {m["output_name"]: m for m in rep["meshes"]}
    for name, key in (("Synth_Body", "body"), ("Synth_Head", "head")):
        m = meshes.get(name)
        checks.append({"check": f"{name} decoded", "passed": bool(m and m["status"] == "decoded")})
        if not m:
            continue
        checks += compare_glb_to_truth(out / "extracted" / m["gltf"], truth[key], "gltf")
        lod0 = m["gltf_summary"]["lods"][0]
        checks.append({"check": f"{name} LOD0 vertex count", "passed": lod0["vertices"] == truth[key]["lod0_vertices"],
                       "detail": [lod0["vertices"], truth[key]["lod0_vertices"]]})
        checks.append({"check": f"{name} LOD0 triangle count",
                       "passed": lod0["triangles"] == truth[key]["lod0_triangles"]})
        if blender:
            b = m.get("blender", {})
            checks.append({"check": f"{name} FBX built and validated in Blender",
                           "passed": b.get("status") == "validated",
                           "detail": [c for c in b.get("validation", {}).get("checks", []) if not c["passed"]]})
    for t in rep["textures"]:
        short = t["resource"].rsplit("/", 1)[-1]
        src = truth["textures"].get(short)
        if src is None or t["status"] != "ok":
            checks.append({"check": f"texture {short} decoded", "passed": False})
            continue
        got = np.asarray(Image.open(out / "extracted" / t["outputs"]["png"]).convert("RGBA")).astype(int)
        ref = np.asarray(src.convert("RGBA")).astype(int)
        err = int(np.abs(got - ref).max())
        lossy = t["header"]["pixel_format"].startswith("BC")
        checks.append({"check": f"texture {short} pixels", "passed": err <= (40 if lossy else 0), "max_error": err})
    ok = all(c["passed"] for c in checks)
    (out / "selftest_report.json").write_text(json.dumps({"passed": ok, "checks": checks}, indent=1, default=str))
    for c in checks:
        log.info("%s %s", "PASS" if c["passed"] else "FAIL", c["check"])
    log.info("selftest %s", "PASSED" if ok else "FAILED")
    return ok
