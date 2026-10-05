from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RescueMap 3D model viewer")
    parser.add_argument("model", nargs="?", help="Optional PLY model to open")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("RescueMap")

    # Give immediate feedback while the VTK/PyVista 3-D runtime is loading.
    # Importing MainWindow lazily lets this splash appear first instead of
    # leaving the user with several seconds of no visible response.
    pixmap = QPixmap(460, 180)
    pixmap.fill(QColor("#20252b"))
    painter = QPainter(pixmap)
    painter.setPen(QColor("#f3f5f7"))
    painter.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
    painter.drawText(28, 72, "RescueMap")
    painter.setPen(QColor("#9aa7b2"))
    painter.setFont(QFont("Microsoft YaHei UI", 11))
    painter.drawText(30, 112, "正在加载三维地图组件……")
    painter.end()
    splash = QSplashScreen(pixmap, Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()

    from rescuemap.ui.main_window import MainWindow

    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    splash.finish(window)

    if args.model:
        window.load_model(Path(args.model))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
