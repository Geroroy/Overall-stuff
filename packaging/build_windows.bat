@echo off
rem Builds dist\FBMOD-Converter.exe. Run from the repository root with Python 3.10+ installed.
python -m pip install -e .[gui] pyinstaller || exit /b 1
pyinstaller --noconfirm --clean packaging\fbmod_converter.spec || exit /b 1
echo.
echo Done: dist\FBMOD-Converter.exe
