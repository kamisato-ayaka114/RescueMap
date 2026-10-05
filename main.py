from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from rescuemap.ui.main_window import MainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RescueMap 3D model viewer")
    parser.add_argument("model", nargs="?", help="Optional PLY model to open")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("RescueMap")

    window = MainWindow()
    window.resize(1280, 800)
    window.show()

    if args.model:
        window.load_model(Path(args.model))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
