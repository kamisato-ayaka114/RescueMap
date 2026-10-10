from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, find_objects, label
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(slots=True)
class _Region:
    label_id: int
    bounds: tuple[int, int, int, int]


class _MaskCanvas(QWidget):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self.rgb: np.ndarray | None = None
        self.static_mask: np.ndarray | None = None
        self.regions: list[_Region] = []
        self.component_labels: np.ndarray | None = None
        self.start: QPoint | None = None
        self.current: QPoint | None = None
        self._display_rect = QRect()

    def set_data(self, rgb: np.ndarray, static_mask: np.ndarray) -> None:
        self.rgb = rgb
        self.static_mask = static_mask.copy()
        self._rebuild_regions()
        self.update()

    def _rebuild_regions(self) -> None:
        self.regions.clear()
        self.component_labels = None
        if self.static_mask is None:
            return
        dynamic = ~self.static_mask
        merge_iterations = min(18, max(4, min(dynamic.shape) // 120))
        merged_labels, _count = label(binary_dilation(dynamic, iterations=merge_iterations))
        self.component_labels = merged_labels
        for region_id, slices in enumerate(find_objects(merged_labels), start=1):
            if slices is None:
                continue
            y_slice, x_slice = slices
            merged_region = merged_labels[slices] == region_id
            if int(np.count_nonzero(dynamic[slices] & merged_region)) < 4:
                continue
            self.regions.append(_Region(region_id, (
                int(x_slice.start), int(y_slice.start),
                int(x_slice.stop - 1), int(y_slice.stop - 1),
            )))

    def _image_point(self, point: QPoint) -> tuple[int, int] | None:
        if self.rgb is None or not self._display_rect.contains(point):
            return None
        height, width = self.rgb.shape[:2]
        x = int((point.x() - self._display_rect.x()) * width / max(self._display_rect.width(), 1))
        y = int((point.y() - self._display_rect.y()) * height / max(self._display_rect.height(), 1))
        return min(width - 1, max(0, x)), min(height - 1, max(0, y))

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#20252b"))
        if self.rgb is None:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "没有图片")
            return
        height, width = self.rgb.shape[:2]
        image = QImage(self.rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy()).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        x = (self.width() - pixmap.width()) // 2
        y = (self.height() - pixmap.height()) // 2
        self._display_rect = QRect(x, y, pixmap.width(), pixmap.height())
        painter.drawPixmap(x, y, pixmap)
        sx, sy = pixmap.width() / width, pixmap.height() / height
        painter.setPen(QPen(QColor("#ff3030"), 2))
        for region in self.regions:
            x0, y0, x1, y1 = region.bounds
            painter.drawRect(QRect(int(x + x0*sx), int(y + y0*sy), max(2, int((x1-x0+1)*sx)), max(2, int((y1-y0+1)*sy))))
        if self.start is not None and self.current is not None:
            painter.setPen(QPen(QColor("#22d3ee"), 3, Qt.PenStyle.DashLine))
            painter.drawRect(QRect(self.start, self.current).normalized())

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._image_point(event.position().toPoint()):
            self.start = event.position().toPoint()
            self.current = self.start
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            point = self._image_point(event.position().toPoint())
            if point is not None and self.static_mask is not None:
                x, y = point
                for index in range(len(self.regions) - 1, -1, -1):
                    x0, y0, x1, y1 = self.regions[index].bounds
                    if x0 <= x <= x1 and y0 <= y <= y1:
                        if self.component_labels is not None:
                            self.static_mask[
                                self.component_labels == self.regions[index].label_id
                            ] = True
                        self._rebuild_regions()
                        self.changed.emit()
                        self.update()
                        break

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self.start is not None:
            self.current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self.start is None or self.static_mask is None:
            return
        first, second = self._image_point(self.start), self._image_point(event.position().toPoint())
        self.start = self.current = None
        if first is not None and second is not None:
            x0, x1 = sorted((first[0], second[0])); y0, y1 = sorted((first[1], second[1]))
            if x1 - x0 >= 3 and y1 - y0 >= 3:
                self.static_mask[y0:y1+1, x0:x1+1] = False
                self._rebuild_regions(); self.changed.emit()
        self.update()


class DynamicReviewDialog(QDialog):
    """Review automatic dynamic regions; left-drag adds and right-click removes."""

    def __init__(self, image_directory: Path, mask_directory: Path, *, phase: str, title: str, parent=None) -> None:
        super().__init__(parent)
        self.image_directory = Path(image_directory)
        self.mask_directory = Path(mask_directory)
        self.phase = phase
        self.paths = sorted(path for path in self.image_directory.rglob("*")
                            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        self.current_index = -1
        self.dirty = False
        self.setWindowTitle(title)
        self.resize(1020, 660)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(230)
        automatic = self._automatic_detection_counts()
        first_detected = -1
        for index, path in enumerate(self.paths):
            relative = str(path.relative_to(self.image_directory)).replace("\\", "/")
            count = automatic.get(relative, 0)
            if self.phase == "review_initial":
                prefix = f"● 算法{count}处" if count > 0 else "○ 未检出"
            else:
                mask_path = self.mask_directory / (relative + ".png")
                selected = False
                if mask_path.is_file():
                    with Image.open(mask_path) as opened:
                        selected = bool(np.any(np.asarray(opened.convert("L")) <= 127))
                prefix = "● 已选平滑区" if selected else "○ 无需处理"
            item = QListWidgetItem(f"{prefix} | {relative}")
            if count > 0 or (self.phase != "review_initial" and selected):
                item.setForeground(QColor("#d62828"))
                if first_detected < 0:
                    first_detected = index
            self.list_widget.addItem(item)
        self.list_widget.currentRowChanged.connect(self._load_index)
        self.canvas = _MaskCanvas()
        self.canvas.changed.connect(self._mark_dirty)
        detected_count = sum(value > 0 for value in automatic.values())
        if self.phase == "review_initial":
            help_text = (
                f"算法已在 {detected_count}/{len(self.paths)} 张图片中生成初始动态候选框。"
                "红框是算法初标或人工补充；左键拖框添加，右键单击框内删除。\n"
                "左侧“未检出”只表示算法没有足够证据，仍需人工检查。切换图片时自动保存。\n"
                "保存继续后，掩膜、带框图片和框坐标会留在本次工程中，下次可直接续跑。"
            )
        else:
            help_text = (
                "跨帧静态背景拼接已经完成。这里只框选仍需局部补全或接缝平滑的区域；"
                "未框选部分将原样保留。\n"
                "左键拖框选择，右键单击红框删除；算法会根据四周边界纹理补全，"
                "并只在选区内侧边缘做高斯羽化。切换图片时自动保存。\n"
                "选区和带框图片会保存在工程中，之后可以从本步骤继续，无需重新检测。"
            )
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setStyleSheet("background:#fff3b0;color:#202020;padding:8px;border:1px solid #d6a800;")
        previous_button = QPushButton("上一张")
        next_button = QPushButton("下一张")
        previous_button.clicked.connect(lambda: self.list_widget.setCurrentRow(max(0, self.current_index - 1)))
        next_button.clicked.connect(lambda: self.list_widget.setCurrentRow(min(len(self.paths)-1, self.current_index + 1)))
        accept_button = QPushButton(
            "保存动态框并继续" if self.phase == "review_initial"
            else "应用所选补全与平滑区域"
        )
        cancel_button = QPushButton("取消本次重建")
        accept_button.clicked.connect(self._accept)
        cancel_button.clicked.connect(self.reject)
        buttons = QHBoxLayout(); buttons.addWidget(previous_button); buttons.addWidget(next_button); buttons.addStretch(); buttons.addWidget(cancel_button); buttons.addWidget(accept_button)
        right = QWidget(); right_layout = QVBoxLayout(right); right_layout.addWidget(help_label); right_layout.addWidget(self.canvas, 1); right_layout.addLayout(buttons)
        splitter = QSplitter(); splitter.addWidget(self.list_widget); splitter.addWidget(right); splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self); layout.addWidget(splitter)
        if self.paths:
            self.list_widget.setCurrentRow(first_detected if first_detected >= 0 else 0)

    def _automatic_detection_counts(self) -> dict[str, int]:
        if self.phase != "review_initial":
            return {}
        report_path = self.mask_directory.parent / "dynamic_report.json"
        if not report_path.is_file():
            return {}
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            return {
                str(item["image"]).replace("\\", "/"): int(item.get("dynamic_feature_clusters", 0))
                for item in report.get("images", [])
                if int(item.get("masked_pixels", 0)) > 0
            }
        except (OSError, ValueError, TypeError, KeyError):
            return {}

    def _mask_path(self, image_path: Path) -> Path:
        return self.mask_directory / (str(image_path.relative_to(self.image_directory)) + ".png")

    def _save_current(self) -> None:
        if self.current_index < 0 or not self.dirty or self.canvas.static_mask is None:
            return
        path = self._mask_path(self.paths[self.current_index]); path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.where(self.canvas.static_mask, 255, 0).astype(np.uint8), "L").save(path)
        self.dirty = False

    def _load_index(self, index: int) -> None:
        self._save_current()
        if index < 0 or index >= len(self.paths):
            return
        self.current_index = index
        with Image.open(self.paths[index]) as opened:
            rgb = np.asarray(opened.convert("RGB")).copy()
        mask_path = self._mask_path(self.paths[index])
        if mask_path.is_file():
            with Image.open(mask_path) as opened:
                static_mask = np.asarray(opened.convert("L")) > 127
        else:
            static_mask = np.ones(rgb.shape[:2], dtype=bool)
        if static_mask.shape != rgb.shape[:2]:
            QMessageBox.warning(self, "掩膜尺寸错误", f"{mask_path.name} 与照片尺寸不一致。")
            static_mask = np.ones(rgb.shape[:2], dtype=bool)
        self.canvas.set_data(rgb, static_mask); self.dirty = False

    def _mark_dirty(self) -> None:
        self.dirty = True

    def _accept(self) -> None:
        self._save_current()
        self._export_boxed_dataset()
        self.accept()

    def _export_boxed_dataset(self) -> None:
        """Save reproducible annotations and red-box previews separate from inputs."""
        workspace = self.mask_directory.parent
        preview_directory = workspace / (
            "manual_review_initial" if self.phase == "review_initial"
            else "manual_review_final"
        )
        preview_directory.mkdir(parents=True, exist_ok=True)
        records = []
        for image_path in self.paths:
            relative = image_path.relative_to(self.image_directory)
            with Image.open(image_path) as opened:
                preview = opened.convert("RGB")
            mask_path = self._mask_path(image_path)
            if mask_path.is_file():
                with Image.open(mask_path) as opened:
                    dynamic = np.asarray(opened.convert("L")) <= 127
            else:
                dynamic = np.zeros((preview.height, preview.width), dtype=bool)
            merge_iterations = min(18, max(4, min(dynamic.shape) // 120))
            labels, _count = label(binary_dilation(dynamic, iterations=merge_iterations))
            boxes = []
            draw = ImageDraw.Draw(preview)
            for region_id, slices in enumerate(find_objects(labels), start=1):
                if slices is None:
                    continue
                y_slice, x_slice = slices
                merged_region = labels[slices] == region_id
                if int(np.count_nonzero(dynamic[slices] & merged_region)) < 4:
                    continue
                box = [int(x_slice.start), int(y_slice.start),
                       int(x_slice.stop - 1), int(y_slice.stop - 1)]
                boxes.append(box)
                draw.rectangle(box, outline=(255, 35, 35), width=max(2, min(preview.size)//500))
            target = preview_directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            options = {"quality": 92} if target.suffix.lower() in {".jpg", ".jpeg"} else {}
            preview.save(target, **options)
            records.append({"image": str(relative).replace("\\", "/"), "boxes": boxes})
        (workspace / f"{self.phase}_annotations.json").write_text(
            json.dumps({"phase": self.phase, "images": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class _ComparisonCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(620, 380)
        self.before: np.ndarray | None = None
        self.after: np.ndarray | None = None

    def set_images(self, before: np.ndarray, after: np.ndarray) -> None:
        self.before = before
        self.after = after
        self.update()

    @staticmethod
    def _pixmap(rgb: np.ndarray) -> QPixmap:
        rgb = np.ascontiguousarray(rgb)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image.copy())

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#20252b"))
        if self.before is None or self.after is None:
            return
        margin, title_height, gap = 12, 34, 12
        panel_width = max(1, (self.width() - margin * 2 - gap) // 2)
        panel_height = max(1, self.height() - margin * 2 - title_height)
        painter.setPen(QColor("#f3f5f7"))
        painter.drawText(QRect(margin, margin, panel_width, title_height), Qt.AlignmentFlag.AlignCenter, "处理前")
        painter.drawText(QRect(margin + panel_width + gap, margin, panel_width, title_height), Qt.AlignmentFlag.AlignCenter, "处理后")
        for index, rgb in enumerate((self.before, self.after)):
            pixmap = self._pixmap(rgb).scaled(
                panel_width, panel_height, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            panel_x = margin + index * (panel_width + gap)
            x = panel_x + (panel_width - pixmap.width()) // 2
            y = margin + title_height + (panel_height - pixmap.height()) // 2
            painter.drawPixmap(x, y, pixmap)


class DatasetComparisonDialog(QDialog):
    """Display every processing result before the reconstruction may continue."""

    def __init__(
        self,
        before_directory: Path,
        after_directory: Path,
        *,
        title: str,
        confirm_text: str,
        alternative_text: str | None = None,
        message_text: str = "逐张检查处理前后的差异，确认后才会进入下一阶段。",
        show_cancel: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.before_directory = Path(before_directory)
        self.after_directory = Path(after_directory)
        self.choice = "cancel"
        self.paths = sorted(
            path for path in self.after_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self.setWindowTitle(title)
        self.resize(1080, 680)
        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(230)
        for path in self.paths:
            self.list_widget.addItem(str(path.relative_to(self.after_directory)))
        self.canvas = _ComparisonCanvas()
        self.list_widget.currentRowChanged.connect(self._load_index)
        message = QLabel(message_text)
        message.setStyleSheet("background:#e8f4ff;color:#202020;padding:8px;border:1px solid #4b9bea;")
        confirm = QPushButton(confirm_text)
        confirm.clicked.connect(lambda: self._finish("confirm"))
        buttons = QHBoxLayout(); buttons.addStretch()
        if show_cancel:
            cancel = QPushButton("取消本次任务")
            cancel.clicked.connect(self.reject)
            buttons.addWidget(cancel)
        if alternative_text:
            alternative = QPushButton(alternative_text)
            alternative.clicked.connect(lambda: self._finish("alternative"))
            buttons.addWidget(alternative)
        buttons.addWidget(confirm)
        right = QWidget(); right_layout = QVBoxLayout(right)
        right_layout.addWidget(message); right_layout.addWidget(self.canvas, 1); right_layout.addLayout(buttons)
        splitter = QSplitter(); splitter.addWidget(self.list_widget); splitter.addWidget(right); splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self); layout.addWidget(splitter)
        if self.paths:
            self.list_widget.setCurrentRow(0)

    def _load_index(self, index: int) -> None:
        if index < 0 or index >= len(self.paths):
            return
        after_path = self.paths[index]
        relative = after_path.relative_to(self.after_directory)
        before_path = self.before_directory / relative
        if not before_path.is_file():
            return
        with Image.open(before_path) as opened:
            before = np.asarray(opened.convert("RGB")).copy()
        with Image.open(after_path) as opened:
            after = np.asarray(opened.convert("RGB")).copy()
        self.canvas.set_images(before, after)

    def _finish(self, choice: str) -> None:
        self.choice = choice
        self.accept()
