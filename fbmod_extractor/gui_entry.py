"""Entry point for the frozen (PyInstaller) desktop app."""

import sys

from fbmod_extractor.gui import main

if __name__ == "__main__":
    sys.exit(main())
