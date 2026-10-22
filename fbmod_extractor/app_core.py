"""One-call conversion used by the desktop app: .fbmod in, output folder + plain-language summary out."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .binary import FormatError
from .blender_export import find_blender
from .extract import Extraction, ExtractOptions
from .fbmod import UnsupportedModFormat

log = logging.getLogger(__name__)


@dataclass
class ConvertRequest:
    mod: Path
    output: Path | None = None  # default: "<mod name>_converted" next to the mod
    blender: str | None = None  # default: auto-detect
    names: dict[str, str] = field(default_factory=dict)
    all_lods: bool = False
    extra_dirs: list[Path] = field(default_factory=list)
    skeleton: Path | None = None

    def output_dir(self) -> Path:
        return self.output or self.mod.with_name(self.mod.stem + "_converted")


@dataclass
class ConvertResult:
    ok: bool
    output: Path
    report: dict | None
    summary: str
    fbx_files: list[Path]
    glb_files: list[Path]
    blender_used: str | None


def parse_names(text: str) -> dict[str, str]:
    """'body=Anakin_Body, head=Anakin_Head' -> {'body': 'Anakin_Body', 'head': 'Anakin_Head'}."""
    names = {}
    for part in text.replace(";", ",").replace("\n", ",").split(","):
        if "=" in part:
            k, v = (x.strip() for x in part.split("=", 1))
            if k and v:
                names[k] = v
    return names


def convert(req: ConvertRequest) -> ConvertResult:
    out = req.output_dir()
    out.mkdir(parents=True, exist_ok=True)
    blender = find_blender(req.blender)
    log.info("Blender: %s", blender or "없음 (FBX 생략, glTF만 생성)")
    opts = ExtractOptions(output=out, lods="all" if req.all_lods else "0", names=dict(req.names),
                          extra_dirs=list(req.extra_dirs), skeleton=req.skeleton)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    file_log = logging.FileHandler(out / "logs" / "extract.log", mode="w", encoding="utf-8")
    file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    file_log.setLevel(logging.INFO)
    pkg_log = logging.getLogger("fbmod_extractor")
    old_level = pkg_log.level
    if pkg_log.getEffectiveLevel() > logging.INFO:
        pkg_log.setLevel(logging.INFO)
    pkg_log.addHandler(file_log)
    try:
        report = Extraction(req.mod, opts).run(blender=blender)
    except (FormatError, UnsupportedModFormat, OSError) as exc:
        log.error("읽을 수 없는 파일: %s", exc)
        return ConvertResult(False, out, None, f"이 파일을 읽을 수 없습니다.\n\n{exc}", [], [], blender)
    finally:
        pkg_log.setLevel(old_level)
        pkg_log.removeHandler(file_log)
        file_log.close()
    fbx = sorted((out / "fbx").glob("*.fbx")) if (out / "fbx").is_dir() else []
    glb = sorted((out / "meshes").glob("*.glb")) if (out / "meshes").is_dir() else []
    return ConvertResult(report["outcome"] in ("complete", "partial"), out, report,
                         summarize(report, blender), fbx, glb, blender)


def summarize(report: dict, blender: str | None) -> str:
    """Plain-language (Korean) summary of an extraction report."""
    s = report["summary"]
    lines = []
    outcome = report["outcome"]
    title = {
        "complete": "변환 완료",
        "partial": "일부만 변환됨",
        "no_geometry_decoded": "변환할 3D 메시가 없습니다",
    }.get(outcome, outcome)
    lines.append(title)
    lines.append("")
    if outcome == "no_geometry_decoded":
        if s["meshsets_in_mod"] == 0:
            lines.append("이 모드에는 메시(3D 모델) 데이터가 들어 있지 않습니다. 게임 원본 모델을 다른 설정으로 "
                         "바꾸는 종류의 모드라서, 모델 자체는 게임 파일에만 있습니다.")
        else:
            lines.append("메시 데이터는 있지만 해독하지 못했습니다. 아래 오류를 확인하세요.")
    for m in report["meshes"]:
        b = m.get("blender", {})
        t = m.get("totals", {})
        if m["status"].startswith("decoded"):
            line = f"• {m['output_name']}: 정점 {t.get('vertices', 0):,}개, 삼각형 {t.get('triangles', 0):,}개"
            if b.get("status") == "validated":
                line += " → FBX 완료 (Blender 검증 통과)"
            elif b.get("status"):
                line += f" → FBX 문제: {b['status']}"
            elif not blender:
                line += " → GLB만 생성 (Blender 없음)"
            lines.append(line)
            sk = m.get("skeleton", {})
            if sk.get("status") == "missing":
                lines.append("    - 스켈레톤이 모드에 없어서 FBX에는 본/웨이트가 빠집니다 (.blend/.glb에는 보존)")
            if m.get("normals", {}).get("dropped_sections"):
                lines.append("    - 저장된 노멀이 형상과 맞지 않아 일부 섹션은 형상 기준 셰이딩을 사용합니다")
        elif m["status"] == "missing_dependency":
            lines.append(f"• {m['output_name']}: 모드에 정점 데이터가 없어 게임 원본 데이터가 필요합니다")
        else:
            lines.append(f"• {m['output_name']}: 실패 ({m.get('error', m['status'])})")
    for r in report.get("requested_outputs", []):
        if r["status"] == "not_in_mod":
            lines.append(f"• {r['output_name']}: 모드에 '{r['matches_resource_substring']}' 메시가 없습니다 "
                         "(모드가 수정하지 않은 부분)")
    if s["textures_in_mod"]:
        lines.append(f"• 텍스처 {s['textures_decoded']}/{s['textures_in_mod']}개 추출 (Extracted_Textures 폴더)")
    if report["missing_dependencies"]:
        lines.append("")
        lines.append(f"게임 원본 데이터가 필요한 항목 {len(report['missing_dependencies'])}개가 있습니다. "
                     "Extraction_Report.json의 missing_dependencies에 Frosty Editor로 내보내는 방법이 적혀 있습니다.")
    if report["meshes"] and not blender:
        lines.append("")
        lines.append("Blender를 찾지 못해 FBX는 만들지 않았습니다. Blender를 설치하거나 설정에서 경로를 지정하세요. "
                     "생성된 .glb 파일은 Blender에서 바로 열 수 있습니다.")
    if report["errors"]:
        lines.append("")
        lines.append(f"오류 {len(report['errors'])}건: " + "; ".join(report["errors"][:3]))
    return "\n".join(lines)
