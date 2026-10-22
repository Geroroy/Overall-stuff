"""FBMOD Converter: a small desktop app. Drop a .fbmod on the window (or on the program icon) to convert it.

Run with ``fbmod-converter [file.fbmod]`` or ``python -m fbmod_extractor.gui``.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from . import __version__
from .app_core import ConvertRequest, ConvertResult, convert, parse_names
from .blender_export import find_blender

try:  # optional: real drag & drop onto the window
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
except Exception:  # pragma: no cover - depends on the environment
    TkinterDnD = None
    DND_FILES = None

CONFIG = Path.home() / ".fbmod_converter.json"
log = logging.getLogger("fbmod_extractor")


def open_path(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa: S606 - opening a local folder/file the user asked for
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__(logging.INFO)
        self.q = q
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        self.q.put(("log", self.format(record)))


class App:
    def __init__(self, root: tk.Tk, initial: str | None = None):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.result: ConvertResult | None = None
        self.cfg = self._load_config()

        root.title(f"FBMOD Converter {__version__}")
        root.minsize(620, 560)
        if sys.platform == "win32":
            root.option_add("*Font", ("Malgun Gothic", 10))

        self.mod_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.blender_var = tk.StringVar(value=self.cfg.get("blender") or find_blender() or "")
        self.names_var = tk.StringVar(value=self.cfg.get("names", ""))
        self.lods_var = tk.BooleanVar(value=self.cfg.get("all_lods", False))
        self.extra_var = tk.StringVar(value=self.cfg.get("extra_dir", ""))
        self.skeleton_var = tk.StringVar()
        self.status_var = tk.StringVar(value="파일을 기다리는 중")
        self._build()

        logging.getLogger("fbmod_extractor").addHandler(_QueueHandler(self.q))
        logging.getLogger("fbmod_extractor").setLevel(logging.INFO)
        root.after(100, self._poll)
        if initial:
            root.after(300, lambda: self.start(Path(initial)))

    # -- layout --------------------------------------------------------------
    def _build(self) -> None:
        pad = {"padx": 10, "pady": 4}
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        self.drop = tk.Label(main, text=".fbmod 파일을 여기로 끌어다 놓으세요\n(또는 클릭해서 선택)",
                             relief="groove", bd=2, height=5, bg="#eef3fb", fg="#234", cursor="hand2",
                             font=("Malgun Gothic" if sys.platform == "win32" else "TkDefaultFont", 13))
        self.drop.pack(fill="x", **pad)
        self.drop.bind("<Button-1>", lambda e: self.choose_mod())
        if TkinterDnD is not None and hasattr(self.drop, "drop_target_register"):
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", self._on_drop)
        else:
            self.drop.configure(text=".fbmod 파일을 선택하려면 클릭하세요\n(또는 프로그램 아이콘 위에 파일을 끌어다 놓으세요)")

        grid = ttk.Frame(main)
        grid.pack(fill="x", **pad)
        grid.columnconfigure(1, weight=1)
        self._row(grid, 0, "모드 파일", self.mod_var, self.choose_mod)
        self._row(grid, 1, "저장 위치", self.out_var, self.choose_out, hint="비워두면 모드 파일 옆에 자동 생성")
        self._row(grid, 3, "Blender", self.blender_var, self.choose_blender,
                  hint="FBX 생성에 필요 (자동 감지)")

        self.adv_open = tk.BooleanVar(value=False)
        ttk.Checkbutton(main, text="고급 설정", variable=self.adv_open, command=self._toggle_adv).pack(anchor="w", **pad)
        self.adv = ttk.Frame(main)
        self.adv.columnconfigure(1, weight=1)
        ttk.Label(self.adv, text="출력 이름").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.adv, textvariable=self.names_var).grid(row=0, column=1, columnspan=2, sticky="ew", padx=4)
        ttk.Label(self.adv, text="예: body=Anakin_Body, head=Anakin_Head", foreground="#777").grid(
            row=1, column=1, sticky="w", padx=4)
        ttk.Checkbutton(self.adv, text="모든 LOD 내보내기 (기본: 최고 품질 LOD만)", variable=self.lods_var).grid(
            row=2, column=1, sticky="w", padx=4, pady=2)
        self._row(self.adv, 3, "게임 데이터 폴더", self.extra_var, self.choose_extra,
                  hint="선택: Frosty Editor로 내보낸 .chunk/.res/.bin/.dds")
        self._row(self.adv, 5, "스켈레톤 파일", self.skeleton_var, self.choose_skeleton,
                  hint="선택: SkeletonAsset을 Binary(.bin)로 내보낸 파일")

        self.bottom = ttk.Frame(main)
        self.bottom.pack(fill="both", expand=True)
        actions = ttk.Frame(self.bottom)
        actions.pack(fill="x", **pad)
        self.convert_btn = ttk.Button(actions, text="변환 시작", command=self._convert_clicked)
        self.convert_btn.pack(side="left")
        self.progress = ttk.Progressbar(actions, mode="determinate", length=180, maximum=100)
        self.progress.pack(side="left", padx=10)
        ttk.Label(actions, textvariable=self.status_var).pack(side="left")

        res = ttk.Frame(self.bottom)
        res.pack(side="bottom", fill="x", **pad)  # result buttons stay visible however tall the log is
        self.logbox = ScrolledText(self.bottom, height=10, state="disabled", wrap="word")
        self.logbox.pack(fill="both", expand=True, **pad)
        self.open_btn = ttk.Button(res, text="결과 폴더 열기", state="disabled",
                                   command=lambda: self.result and open_path(self.result.output))
        self.open_btn.pack(side="left")
        self.report_btn = ttk.Button(res, text="리포트 열기", state="disabled",
                                     command=lambda: self.result and open_path(self.result.output / "Extraction_Report.json"))
        self.report_btn.pack(side="left", padx=6)

    def _row(self, parent, row, label, var, command, hint=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=4)
        ttk.Button(parent, text="찾아보기", command=command).grid(row=row, column=2)
        if hint:
            ttk.Label(parent, text=hint, foreground="#777").grid(row=row + 1, column=1, sticky="w", padx=4)

    def _toggle_adv(self):
        if self.adv_open.get():
            self.adv.pack(fill="x", padx=10, pady=4, before=self.bottom)
        else:
            self.adv.pack_forget()

    # -- file choosers -------------------------------------------------------
    def choose_mod(self):
        p = filedialog.askopenfilename(title=".fbmod 파일 선택", filetypes=[("Frosty mod", "*.fbmod"), ("모든 파일", "*.*")])
        if p:
            self.start(Path(p))

    def choose_out(self):
        p = filedialog.askdirectory(title="저장 위치 선택")
        if p:
            self.out_var.set(p)

    def choose_blender(self):
        types = [("Blender", "blender.exe"), ("모든 파일", "*.*")] if sys.platform == "win32" else [("모든 파일", "*")]
        p = filedialog.askopenfilename(title="Blender 실행 파일 선택", filetypes=types)
        if p:
            self.blender_var.set(p)

    def choose_extra(self):
        p = filedialog.askdirectory(title="Frosty Editor로 내보낸 게임 데이터 폴더")
        if p:
            self.extra_var.set(p)

    def choose_skeleton(self):
        p = filedialog.askopenfilename(title="스켈레톤 .bin 선택", filetypes=[("EBX binary", "*.bin *.ebx"), ("모든 파일", "*.*")])
        if p:
            self.skeleton_var.set(p)

    def _on_drop(self, event):
        paths = [Path(p) for p in self.root.tk.splitlist(event.data)]
        mods = [p for p in paths if p.suffix.lower() == ".fbmod"] or paths
        if mods:
            self.start(mods[0])

    # -- conversion ----------------------------------------------------------
    def _convert_clicked(self):
        if self.mod_var.get():
            self.start(Path(self.mod_var.get()))
        else:
            self.choose_mod()

    def start(self, mod: Path) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("변환 중", "이미 변환 중입니다. 끝날 때까지 기다려 주세요.")
            return
        if not mod.is_file():
            messagebox.showerror("파일 없음", f"파일을 찾을 수 없습니다:\n{mod}")
            return
        self.mod_var.set(str(mod))
        req = ConvertRequest(
            mod=mod,
            output=Path(self.out_var.get()) if self.out_var.get().strip() else None,
            blender=self.blender_var.get().strip() or None,
            names=parse_names(self.names_var.get()),
            all_lods=self.lods_var.get(),
            extra_dirs=[Path(self.extra_var.get())] if self.extra_var.get().strip() else [],
            skeleton=Path(self.skeleton_var.get()) if self.skeleton_var.get().strip() else None,
        )
        self._save_config()
        self._clear_log()
        self._log(f"변환 시작: {mod.name}\n저장 위치: {req.output_dir()}")
        self.convert_btn.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        self.report_btn.configure(state="disabled")
        self.status_var.set("변환 중…")
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(12)
        self.worker = threading.Thread(target=self._work, args=(req,), daemon=True)
        self.worker.start()

    def _work(self, req: ConvertRequest) -> None:
        try:
            result = convert(req)
        except Exception as exc:  # show the failure instead of dying silently
            log.exception("변환 중 예기치 못한 오류")
            result = ConvertResult(False, req.output_dir(), None, f"예기치 못한 오류: {exc}", [], [], None)
        self.q.put(("done", result))

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "done":
                    self._finished(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _finished(self, result: ConvertResult) -> None:
        self.result = result
        self.progress.stop()
        self.progress.configure(mode="determinate", value=100 if result.ok else 0)
        self.convert_btn.configure(state="normal")
        self.open_btn.configure(state="normal" if result.output.is_dir() else "disabled")
        self.report_btn.configure(state="normal" if (result.output / "Extraction_Report.json").is_file() else "disabled")
        self.status_var.set("완료" if result.ok else "확인 필요")
        self._log("\n" + result.summary)
        if result.fbx_files:
            self._log("\nFBX: " + ", ".join(p.name for p in result.fbx_files))
        (messagebox.showinfo if result.ok else messagebox.showwarning)("FBMOD Converter", result.summary)

    # -- helpers -------------------------------------------------------------
    def _log(self, text: str) -> None:
        self.logbox.configure(state="normal")
        self.logbox.insert("end", text + "\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def _clear_log(self) -> None:
        self.logbox.configure(state="normal")
        self.logbox.delete("1.0", "end")
        self.logbox.configure(state="disabled")

    def _load_config(self) -> dict:
        try:
            return json.loads(CONFIG.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_config(self) -> None:
        cfg = {"blender": self.blender_var.get(), "names": self.names_var.get(),
               "all_lods": self.lods_var.get(), "extra_dir": self.extra_var.get()}
        try:
            CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    initial = next((a for a in argv if not a.startswith("-")), None)
    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    App(root, initial)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
