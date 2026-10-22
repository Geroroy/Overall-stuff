# PyInstaller spec for the FBMOD Converter desktop app (one-file, windowed).
# Build: pyinstaller packaging/fbmod_converter.spec   (from the repository root)
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = [
    ("../fbmod_extractor/blender/build_fbx.py", "fbmod_extractor/blender"),
    ("../fbmod_extractor/blender/validate_fbx.py", "fbmod_extractor/blender"),
    ("../fbmod_extractor/data/swbf2_render_formats.json", "fbmod_extractor/data"),
]
binaries = []
try:  # drag & drop support is optional
    datas += collect_data_files("tkinterdnd2")
    binaries += collect_dynamic_libs("tkinterdnd2")
except Exception:
    pass

a = Analysis(
    ["../fbmod_extractor/gui_entry.py"],
    pathex=[".."],
    datas=datas,
    binaries=binaries,
    hiddenimports=["tkinterdnd2", "PIL.DdsImagePlugin", "PIL.PngImagePlugin", "PIL.TgaImagePlugin"],
    excludes=["dnfile", "pytest"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="FBMOD-Converter",
    console=False,
    upx=False,
)
