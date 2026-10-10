from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import QElapsedTimer, QEvent, QPoint, QRect, QSettings, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QProgressDialog,
    QRubberBand,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from scipy.spatial import cKDTree
from vtkmodules.vtkFiltersGeneral import vtkExtractSelectedFrustum
from vtkmodules.vtkRenderingCore import vtkPointPicker

from rescuemap.core.calibration import (
    flip_vertical_transform,
    ground_alignment_transform,
    uniform_scale_transform,
)
from rescuemap.core.candidate_validation import (
    CandidateValidationResult,
    evaluate_region_candidates,
)
from rescuemap.core.colmap_runner import (
    ColmapRunner,
    PreprocessingResult,
    ReconstructionResult,
    bundled_colmap_path,
)
from rescuemap.core.density import DensityResult, analyze_point_density
from rescuemap.core.path_planning import RouteResult, plan_safe_path
from rescuemap.core.scene import SceneModel
from rescuemap.core.terrain import (
    ACCEPTED,
    FILL_AUTO,
    FILL_CONCAVE,
    FILL_CONVEX,
    FILL_FLAT,
    FILL_PLANE,
    FILL_QUADRATIC,
    FILL_STRATEGY_NAMES,
    PREDICTED,
    RESHOOT,
    UNCERTAIN,
    TerrainGrid,
    build_terrain_grid,
)
from rescuemap.core.terrain_mesh import build_terrain_mesh
from rescuemap.ui.dynamic_review_dialog import (
    DatasetComparisonDialog,
    DynamicReviewDialog,
)


class CollapsibleGroupBox(QGroupBox):
    """A compact checkable section that can be collapsed inside a scroll pane."""

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setCheckable(True)
        self.setChecked(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.toggled.connect(self._set_expanded)

    def _set_expanded(self, expanded: bool) -> None:
        for child in self.findChildren(
            QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly
        ):
            child.setVisible(expanded)
        self.setMaximumHeight(16_777_215 if expanded else 30)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RescueMap Viewer 0.8")
        self.scene: SceneModel | None = None
        self.density_result: DensityResult | None = None
        self.color_mode = "original"
        self.green_multiplier = 1.5
        self.red_multiplier = 3.0
        self.picking_mode: str | None = None
        self.picking_target = 0
        self.picked_points: list[np.ndarray] = []
        self.guide_target_region_id: int | None = None
        self.manual_fill_target_region_id: int | None = None
        self.manual_fill_strategy = FILL_PLANE
        self.manual_fill_amplitude = 0.0
        self.scene_point_tree: cKDTree | None = None
        self.mouse_press_position: tuple[float, float] | None = None
        self.rectangle_start_position: QPoint | None = None
        self.terrain_roi: tuple[float, float, float, float] | None = None
        self.terrain_grid: TerrainGrid | None = None
        self.terrain_settings: dict[str, float | int] | None = None
        self.candidate_results: dict[int, CandidateValidationResult] = {}
        self.terrain_mesh: pv.PolyData | None = None
        self.terrain_mesh_settings: dict[str, float] | None = None
        self.terrain_mesh_path: Path | None = None
        self.route_start_point: np.ndarray | None = None
        self.route_end_point: np.ndarray | None = None
        self.route_result: RouteResult | None = None
        self.context_mesh: SceneModel | None = None
        self.settings = QSettings("RescueMap", "RescueMap")
        self.active_pipeline: str | None = None
        self.last_preprocessing_result: PreprocessingResult | None = None
        self.processing_wait_dialog: QProgressDialog | None = None
        self.processing_wait_title = ""
        self.processing_elapsed = QElapsedTimer()
        self.processing_timer = QTimer(self)
        self.processing_timer.setInterval(1000)
        self.processing_timer.timeout.connect(self._update_processing_wait_dialog)
        self.colmap_runner = ColmapRunner(self)
        self.colmap_runner.log_received.connect(self._append_reconstruction_log)
        self.colmap_runner.stage_changed.connect(self._update_reconstruction_stage)
        self.colmap_runner.running_changed.connect(self._set_reconstruction_running)
        self.colmap_runner.completed.connect(self._reconstruction_completed)
        self.colmap_runner.failed.connect(self._reconstruction_failed)
        self.colmap_runner.review_requested.connect(self._review_dynamic_dataset)
        self.colmap_runner.enhancement_review_requested.connect(
            self._review_enhanced_dataset
        )
        self.colmap_runner.final_review_requested.connect(
            self._review_final_static_dataset
        )
        self.colmap_runner.preprocessing_completed.connect(
            self._preprocessing_completed
        )

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#20252b")
        self.plotter.add_axes()
        self.plotter.show_grid(color="#66727d")
        self.plotter.interactor.installEventFilter(self)
        self.selection_rubber_band = QRubberBand(
            QRubberBand.Shape.Rectangle, self.plotter.interactor
        )
        self.selection_rubber_band.setStyleSheet(
            "QRubberBand { border: 3px solid #22d3ee; "
            "background-color: rgba(34, 211, 238, 45); }"
        )
        self.color_legend_widget = self._build_color_legend()

        self.info_form = QFormLayout()
        self.info_form.addRow(QLabel("尚未加载模型"))
        info_group = CollapsibleGroupBox("模型信息（点击标题折叠）")
        info_group.setLayout(self.info_form)

        side_panel = QWidget()
        side_panel.setMinimumWidth(350)
        side_layout = QVBoxLayout(side_panel)
        title = QLabel("RescueMap")
        title.setStyleSheet("font-size: 20px; font-weight: 650;")
        side_layout.addWidget(title)
        self.picking_help_widget = QFrame()
        picking_help_layout = QVBoxLayout(self.picking_help_widget)
        picking_help_layout.setContentsMargins(8, 6, 8, 6)
        self.picking_help_label = QLabel()
        self.picking_help_label.setWordWrap(True)
        self.cancel_picking_button = QPushButton("取消选点（Esc）")
        self.cancel_picking_button.clicked.connect(self.cancel_picking)
        picking_help_layout.addWidget(self.picking_help_label)
        picking_help_layout.addWidget(self.cancel_picking_button)
        self.picking_help_widget.setStyleSheet(
            "QFrame { background:#fff3b0; color:#202020; border:2px solid #d6a800; "
            "border-radius:8px; } QLabel { border:none; } QPushButton { "
            "background:white; border:1px solid #b78d00; padding:5px; }"
        )
        self.picking_help_widget.hide()
        reconstruction_group = self._build_reconstruction_group()
        preprocessing_group = self._build_preprocessing_group()
        side_layout.addWidget(preprocessing_group)
        side_layout.addWidget(reconstruction_group)
        side_layout.addWidget(info_group)
        side_layout.addWidget(self._build_full_map_group())
        side_layout.addWidget(self._build_calibration_group())
        side_layout.addWidget(self._build_prediction_group())
        side_layout.addWidget(self._build_mesh_group())
        side_layout.addWidget(self._build_path_planning_group())
        side_layout.addStretch(1)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setWidget(side_panel)
        side_scroll.setMinimumWidth(370)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.plotter.interactor)
        splitter.addWidget(side_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)
        self.picking_help_widget.setParent(self.plotter.interactor)
        self._create_actions()
        self.statusBar().showMessage("请打开 fused.ply")

    def _build_reconstruction_group(self) -> QGroupBox:
        group = CollapsibleGroupBox("照片一键重建（点击标题折叠）")
        layout = QVBoxLayout(group)

        self.image_directory_edit = QLineEdit()
        self.image_directory_edit.setPlaceholderText("包含无人机照片的文件夹")
        image_button = QPushButton("选择照片文件夹")
        image_button.clicked.connect(self.choose_reconstruction_images)

        self.project_directory_edit = QLineEdit()
        self.project_directory_edit.setPlaceholderText("建议选择F盘中的工程文件夹")
        project_button = QPushButton("选择工程保存目录")
        project_button.clicked.connect(self.choose_reconstruction_project)

        built_in_colmap = bundled_colmap_path()
        saved_colmap = str(self.settings.value("colmap_path", ""))
        self.colmap_path_edit = QLineEdit(
            str(built_in_colmap) if built_in_colmap is not None else saved_colmap
        )
        self.colmap_path_edit.setPlaceholderText("选择 COLMAP 的 bin\\colmap.exe")
        colmap_button = QPushButton("选择 COLMAP 程序")
        colmap_button.clicked.connect(self.choose_colmap_program)
        if built_in_colmap is not None:
            self.colmap_path_edit.setReadOnly(True)
            colmap_button.setText("已内置 COLMAP 引擎")
            colmap_button.setEnabled(False)

        self.reconstruction_quality_combo = QComboBox()
        self.reconstruction_quality_combo.addItem("快速预览", "fast")
        self.reconstruction_quality_combo.addItem("标准重建（推荐）", "standard")
        self.reconstruction_quality_combo.addItem("高质量重建", "high")
        self.reconstruction_quality_combo.setCurrentIndex(1)

        buttons = QHBoxLayout()
        self.start_reconstruction_button = QPushButton("开始一键重建")
        self.start_reconstruction_button.clicked.connect(self.start_reconstruction)
        self.cancel_reconstruction_button = QPushButton("取消")
        self.cancel_reconstruction_button.clicked.connect(self.colmap_runner.cancel)
        self.cancel_reconstruction_button.setEnabled(False)
        buttons.addWidget(self.start_reconstruction_button)
        buttons.addWidget(self.cancel_reconstruction_button)

        self.reconstruction_progress = QProgressBar()
        self.reconstruction_progress.setRange(0, 7)
        self.reconstruction_progress.setValue(0)
        self.reconstruction_progress.setFormat("尚未开始")
        self.reconstruction_log = QPlainTextEdit()
        self.reconstruction_log.setReadOnly(True)
        self.reconstruction_log.setMaximumHeight(150)
        self.reconstruction_log.setPlaceholderText("重建日志会显示在这里")

        layout.addWidget(QLabel("照片目录："))
        layout.addWidget(self.image_directory_edit)
        layout.addWidget(image_button)
        layout.addWidget(QLabel("工程目录（中间文件也保存在这里）："))
        layout.addWidget(self.project_directory_edit)
        layout.addWidget(project_button)
        layout.addWidget(QLabel("COLMAP程序："))
        layout.addWidget(self.colmap_path_edit)
        layout.addWidget(colmap_button)
        layout.addWidget(QLabel("重建质量："))
        layout.addWidget(self.reconstruction_quality_combo)
        layout.addLayout(buttons)
        layout.addWidget(self.reconstruction_progress)
        layout.addWidget(self.reconstruction_log)
        return group

    def _build_preprocessing_group(self) -> QGroupBox:
        group = CollapsibleGroupBox("照片预处理（先处理并确认，再进行重建）")
        layout = QVBoxLayout(group)
        explanation = QLabel(
            "独立生成最终静态图片数据集：弱光增强 → 自动动态检测 → "
            "人工增删框 → 多视角背景恢复 → 最终确认。"
        )
        explanation.setWordWrap(True)
        ordering_note = QLabel(
            "重要：软件按文件名排序来寻找前后相邻照片。请让文件名随拍摄时间递增，"
            "例如 0001.jpg、0002.jpg、0003.jpg。"
        )
        ordering_note.setWordWrap(True)
        ordering_note.setStyleSheet(
            "color: #9a5b00; background: #fff4d6; border: 1px solid #e3b341; "
            "border-radius: 4px; padding: 6px;"
        )
        self.preprocess_image_directory_edit = QLineEdit()
        self.preprocess_image_directory_edit.setPlaceholderText("包含原始照片的文件夹")
        image_button = QPushButton("选择原始照片文件夹")
        image_button.clicked.connect(self.choose_preprocess_images)
        self.preprocess_project_directory_edit = QLineEdit()
        self.preprocess_project_directory_edit.setPlaceholderText("预处理成果与中间文件保存目录")
        project_button = QPushButton("选择预处理保存目录")
        project_button.clicked.connect(self.choose_preprocess_project)
        self.low_light_enhancement_checkbox = QCheckBox(
            "统一增强夜间/弱光照片（处理后先预览）"
        )
        self.low_light_enhancement_checkbox.setChecked(True)
        self.dynamic_strength_combo = QComboBox()
        self.dynamic_strength_combo.addItem("保守：优先保留静态内容", "conservative")
        self.dynamic_strength_combo.addItem("标准：动态抑制与保留内容平衡（推荐）", "standard")
        self.dynamic_strength_combo.addItem("激进：动态物体非常多时使用", "aggressive")
        self.dynamic_strength_combo.setCurrentIndex(1)
        buttons = QHBoxLayout()
        self.start_preprocessing_button = QPushButton("开始照片预处理")
        self.start_preprocessing_button.clicked.connect(self.start_preprocessing)
        self.cancel_preprocessing_button = QPushButton("取消")
        self.cancel_preprocessing_button.clicked.connect(self.colmap_runner.cancel)
        self.cancel_preprocessing_button.setEnabled(False)
        buttons.addWidget(self.start_preprocessing_button)
        buttons.addWidget(self.cancel_preprocessing_button)
        self.view_static_dataset_button = QPushButton("再次查看最近的静态数据集")
        self.view_static_dataset_button.setEnabled(False)
        self.view_static_dataset_button.clicked.connect(self._view_last_static_dataset)
        resume_label = QLabel("复用已有预处理成果（可直接从中间步骤继续）：")
        self.resume_workspace_edit = QLineEdit()
        self.resume_workspace_edit.setPlaceholderText("选择 reconstruction_日期_时间 工程目录")
        resume_workspace_button = QPushButton("选择已有预处理工程")
        resume_workspace_button.clicked.connect(self.choose_resume_workspace)
        self.resume_stage_combo = QComboBox()
        self.resume_stage_combo.addItem("重新执行自动动态检测", "generate_masks")
        self.resume_stage_combo.addItem("从第一次人工复核动态框开始", "review_initial")
        self.resume_stage_combo.addItem("从首次背景恢复开始", "complete_backgrounds")
        self.resume_stage_combo.addItem("从第二次人工复核静态图片开始", "review_completed")
        self.resume_stage_combo.addItem("从最终静态数据生成开始", "complete_final")
        self.resume_stage_combo.addItem("只查看并确认最终静态数据集", "review_final_static")
        self.resume_preprocessing_button = QPushButton("从所选步骤继续")
        self.resume_preprocessing_button.clicked.connect(self.resume_preprocessing)
        self.preprocessing_progress = QProgressBar()
        self.preprocessing_progress.setRange(0, 1)
        self.preprocessing_progress.setValue(0)
        self.preprocessing_progress.setFormat("尚未开始")
        self.preprocessing_log = QPlainTextEdit()
        self.preprocessing_log.setReadOnly(True)
        self.preprocessing_log.setMaximumHeight(140)
        layout.addWidget(explanation)
        layout.addWidget(ordering_note)
        layout.addWidget(QLabel("原始照片目录："))
        layout.addWidget(self.preprocess_image_directory_edit)
        layout.addWidget(image_button)
        layout.addWidget(QLabel("预处理保存目录："))
        layout.addWidget(self.preprocess_project_directory_edit)
        layout.addWidget(project_button)
        layout.addWidget(self.low_light_enhancement_checkbox)
        layout.addWidget(QLabel("自动动态检测强度："))
        layout.addWidget(self.dynamic_strength_combo)
        layout.addLayout(buttons)
        layout.addWidget(self.view_static_dataset_button)
        layout.addWidget(resume_label)
        layout.addWidget(self.resume_workspace_edit)
        layout.addWidget(resume_workspace_button)
        layout.addWidget(self.resume_stage_combo)
        layout.addWidget(self.resume_preprocessing_button)
        layout.addWidget(self.preprocessing_progress)
        layout.addWidget(self.preprocessing_log)
        return group

    def choose_reconstruction_images(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择照片文件夹")
        if directory:
            self.image_directory_edit.setText(directory)

    def choose_preprocess_images(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择原始照片文件夹")
        if directory:
            self.preprocess_image_directory_edit.setText(directory)

    def choose_preprocess_project(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择预处理保存目录")
        if directory:
            self.preprocess_project_directory_edit.setText(directory)

    def choose_resume_workspace(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择已有照片预处理工程")
        if directory:
            self.resume_workspace_edit.setText(directory)

    def choose_reconstruction_project(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择工程保存目录")
        if directory:
            self.project_directory_edit.setText(directory)

    def choose_colmap_program(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择 COLMAP 程序",
            str(self.colmap_path_edit.text()),
            "COLMAP程序 (colmap.exe COLMAP.exe COLMAP.bat);;可执行文件 (*.exe *.bat)",
        )
        if filename:
            self.colmap_path_edit.setText(filename)
            self.settings.setValue("colmap_path", filename)

    def start_reconstruction(self) -> None:
        self.reconstruction_log.clear()
        self.active_pipeline = "reconstruction"
        try:
            workspace = self.colmap_runner.start(
                Path(self.colmap_path_edit.text().strip()),
                Path(self.image_directory_edit.text().strip()),
                Path(self.project_directory_edit.text().strip()),
                str(self.reconstruction_quality_combo.currentData()),
                False,
                "conservative",
                False,
                False,
            )
            self.settings.setValue("colmap_path", self.colmap_path_edit.text().strip())
            self.statusBar().showMessage(f"正在重建：{workspace}")
        except Exception as exc:
            self.active_pipeline = None
            QMessageBox.warning(self, "无法开始重建", str(exc))

    def start_preprocessing(self) -> None:
        self.preprocessing_log.clear()
        self.active_pipeline = "preprocessing"
        try:
            workspace = self.colmap_runner.start(
                Path(self.colmap_path_edit.text().strip()),
                Path(self.preprocess_image_directory_edit.text().strip()),
                Path(self.preprocess_project_directory_edit.text().strip()),
                "fast",
                True,
                str(self.dynamic_strength_combo.currentData()),
                self.low_light_enhancement_checkbox.isChecked(),
                True,
            )
            self.statusBar().showMessage(f"正在预处理照片：{workspace}")
        except Exception as exc:
            self.active_pipeline = None
            QMessageBox.warning(self, "无法开始照片预处理", str(exc))

    def resume_preprocessing(self) -> None:
        self.preprocessing_log.clear()
        self.active_pipeline = "preprocessing"
        try:
            workspace = self.colmap_runner.resume_preprocessing(
                Path(self.colmap_path_edit.text().strip()),
                Path(self.preprocess_image_directory_edit.text().strip()),
                Path(self.resume_workspace_edit.text().strip()),
                str(self.resume_stage_combo.currentData()),
            )
            self.statusBar().showMessage(f"继续照片预处理：{workspace}")
        except Exception as exc:
            self.active_pipeline = None
            QMessageBox.warning(self, "无法继续照片预处理", str(exc))

    def _append_reconstruction_log(self, text: str) -> None:
        target = (
            self.preprocessing_log
            if self.active_pipeline == "preprocessing"
            else self.reconstruction_log
        )
        cursor = target.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        target.setTextCursor(cursor)
        target.ensureCursorVisible()

    def _update_reconstruction_stage(self, current: int, total: int, title: str) -> None:
        self._hide_processing_wait_dialog()
        progress = (
            self.preprocessing_progress
            if self.active_pipeline == "preprocessing"
            else self.reconstruction_progress
        )
        progress.setRange(0, total)
        progress.setValue(current - 1)
        progress.setFormat(f"第 {current}/{total} 步：{title}")
        self.statusBar().showMessage(f"{'照片预处理' if self.active_pipeline == 'preprocessing' else 'COLMAP'}：{title}")
        if title in {
            "统一增强夜间图像",
            "快速提取低分辨率特征",
            "快速匹配相邻图像",
            "估计预览相机位姿",
            "分析跨帧几何并生成动态掩膜",
            "多视角恢复静态背景",
            "生成最终静态图片数据集",
        }:
            self._show_processing_wait_dialog(title)

    def _show_processing_wait_dialog(self, title: str) -> None:
        self.processing_wait_title = title
        self.processing_elapsed.start()
        dialog = QProgressDialog(self)
        dialog.setWindowTitle("照片处理中")
        dialog.setLabelText(f"{title}\n已耗时 0 秒，请稍候……")
        dialog.setRange(0, 0)
        dialog.setCancelButton(None)
        dialog.setMinimumDuration(0)
        dialog.setMinimumWidth(360)
        dialog.setAutoClose(False)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.processing_wait_dialog = dialog
        self.processing_timer.start()
        dialog.show()

    def _update_processing_wait_dialog(self) -> None:
        if self.processing_wait_dialog is None:
            self.processing_timer.stop()
            return
        seconds = max(0, self.processing_elapsed.elapsed() // 1000)
        minutes, remainder = divmod(seconds, 60)
        elapsed_text = f"{minutes} 分 {remainder} 秒" if minutes else f"{remainder} 秒"
        self.processing_wait_dialog.setLabelText(
            f"{self.processing_wait_title}\n已耗时 {elapsed_text}，请稍候……"
        )

    def _hide_processing_wait_dialog(self) -> None:
        self.processing_timer.stop()
        if self.processing_wait_dialog is not None:
            self.processing_wait_dialog.close()
            self.processing_wait_dialog.deleteLater()
            self.processing_wait_dialog = None

    def _set_reconstruction_running(self, running: bool) -> None:
        self.start_reconstruction_button.setEnabled(not running)
        self.start_preprocessing_button.setEnabled(not running)
        self.resume_preprocessing_button.setEnabled(not running)
        self.resume_stage_combo.setEnabled(not running)
        self.cancel_reconstruction_button.setEnabled(
            running and self.active_pipeline == "reconstruction"
        )
        self.cancel_preprocessing_button.setEnabled(
            running and self.active_pipeline == "preprocessing"
        )
        self.low_light_enhancement_checkbox.setEnabled(not running)
        self.dynamic_strength_combo.setEnabled(not running)

    def _reconstruction_failed(self, message: str) -> None:
        self._hide_processing_wait_dialog()
        progress = (
            self.preprocessing_progress
            if self.active_pipeline == "preprocessing"
            else self.reconstruction_progress
        )
        progress.setFormat("照片预处理未完成" if self.active_pipeline == "preprocessing" else "重建未完成")
        self.statusBar().showMessage(message)
        QMessageBox.warning(self, "重建未完成", message)
        self.active_pipeline = None

    def _preprocessing_completed(self, result: PreprocessingResult) -> None:
        self._hide_processing_wait_dialog()
        self.last_preprocessing_result = result
        self.view_static_dataset_button.setEnabled(True)
        self.preprocessing_progress.setValue(self.preprocessing_progress.maximum())
        self.preprocessing_progress.setFormat("照片预处理完成，已送入一键重建")
        self.image_directory_edit.setText(str(result.static_images))
        self.project_directory_edit.setText(str(result.workspace.parent))
        self.statusBar().showMessage(f"照片预处理完成：{result.static_images}")
        self.active_pipeline = None
        QMessageBox.information(
            self,
            "照片预处理完成",
            "最终静态图片数据集已经生成，并自动填入下方的“照片一键重建”。\n\n"
            "请确认照片目录后，再点击“开始一键重建”。\n\n"
            f"最终图片：{result.static_images}\n"
            f"动态掩膜：{result.masks}\n"
            f"第一次带框图片：{result.workspace / 'manual_review_initial'}\n"
            f"第二次带框图片：{result.workspace / 'manual_review_final'}",
        )

    def _view_last_static_dataset(self) -> None:
        result = self.last_preprocessing_result
        if result is None or not result.static_images.is_dir():
            QMessageBox.warning(self, "无法查看", "最近的静态图片数据集已经不存在。")
            self.view_static_dataset_button.setEnabled(False)
            return
        original = Path(self.preprocess_image_directory_edit.text().strip())
        dialog = DatasetComparisonDialog(
            original,
            result.static_images,
            title="查看最终静态图片数据集",
            confirm_text="关闭查看",
            message_text="这里保留最近一次处理后的静态数据集，可逐张与原照片比较。",
            show_cancel=False,
            parent=self,
        )
        dialog.exec()

    def _review_dynamic_dataset(
        self, phase: str, image_directory: str, mask_directory: str
    ) -> None:
        title = (
            "第一次复核：修正自动检测的动态区域"
            if phase == "review_initial"
            else "第二次复核：检查恢复后的静态图片"
        )
        dialog = DynamicReviewDialog(
            Path(image_directory), Path(mask_directory), phase=phase,
            title=title, parent=self
        )
        accepted = dialog.exec() == dialog.DialogCode.Accepted
        self.colmap_runner.resume_after_review(accepted)

    def _review_enhanced_dataset(self, original_directory: str, enhanced_directory: str) -> None:
        dialog = DatasetComparisonDialog(
            Path(original_directory), Path(enhanced_directory),
            title="检查夜间图像增强结果",
            confirm_text="使用增强照片继续",
            alternative_text="放弃增强，使用原始照片",
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            choice = "cancel"
        else:
            choice = "original" if dialog.choice == "alternative" else "enhanced"
        self.colmap_runner.resume_after_enhancement_review(choice)

    def _review_final_static_dataset(self, original_directory: str, static_directory: str) -> None:
        dialog = DatasetComparisonDialog(
            Path(original_directory), Path(static_directory),
            title="确认最终静态图片数据集",
            confirm_text="确认并保存静态数据集",
            alternative_text="返回继续修正动态框",
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            choice = "cancel"
        else:
            choice = "revise" if dialog.choice == "alternative" else "confirm"
        self.colmap_runner.resume_after_final_review(choice)

    def _reconstruction_completed(self, result: ReconstructionResult) -> None:
        self._hide_processing_wait_dialog()
        self.reconstruction_progress.setValue(self.reconstruction_progress.maximum())
        self.reconstruction_progress.setFormat("重建完成，正在加载三维地图")
        self.load_model(result.point_cloud)
        self.load_full_map_mesh(result.poisson_mesh)
        self.show_full_map_checkbox.setChecked(True)
        self.reconstruction_progress.setFormat("重建与地图加载完成")
        self.statusBar().showMessage(f"重建完成：{result.workspace}")
        self.active_pipeline = None
        QMessageBox.information(
            self,
            "重建完成",
            "稠密点云和 Poisson 网格已经生成并自动加载。\n\n"
            f"工作区：{result.workspace}",
        )

    def _show_picking_help(self, message: str) -> None:
        self.picking_help_label.setText(message)
        self.picking_help_widget.show()
        self._position_picking_help()
        self.picking_help_widget.raise_()

    def _position_picking_help(self) -> None:
        if not self.picking_help_widget.isVisible():
            return
        available_width = self.plotter.interactor.width()
        width = min(620, max(320, available_width - 40))
        self.picking_help_widget.setFixedWidth(width)
        self.picking_help_widget.adjustSize()
        x = max(10, (available_width - width) // 2)
        self.picking_help_widget.move(x, 12)

    def _build_color_legend(self) -> QFrame:
        legend = QFrame(self.plotter.interactor)
        legend.setStyleSheet(
            "QFrame { background-color: rgba(24, 29, 35, 220); "
            "border: 1px solid #66727d; border-radius: 6px; }"
        )
        layout = QVBoxLayout(legend)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(1)
        entries = (
            ("● 未判断区域", "#f2be2d"),
            ("● 已接受区域", "#4696eb"),
            ("● 待定区域", "#a0a0a0"),
            ("● 补拍区域", "#ef3b3b"),
            ("● 批量勾选区域", "#22d3ee"),
            ("● 当前区域", "#d946ef"),
        )
        for text, color in entries:
            label = QLabel(text)
            label.setStyleSheet(
                f"color:{color}; font-size:14px; font-weight:600; border:none;"
            )
            layout.addWidget(label)
        legend.adjustSize()
        legend.hide()
        return legend

    def _position_color_legend(self) -> None:
        if not self.color_legend_widget.isVisible():
            return
        self.color_legend_widget.adjustSize()
        x = max(
            8,
            self.plotter.interactor.width()
            - self.color_legend_widget.width()
            - 12,
        )
        y = max(
            8,
            self.plotter.interactor.height()
            - self.color_legend_widget.height()
            - 12,
        )
        self.color_legend_widget.move(x, y)
        self.color_legend_widget.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._position_picking_help()
        self._position_color_legend()

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self.plotter.interactor and self.picking_mode in (
            "region-box", "fill-box"
        ):
            if (
                event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self.rectangle_start_position = event.position().toPoint()
                self.selection_rubber_band.setGeometry(
                    QRect(self.rectangle_start_position, self.rectangle_start_position)
                )
                self.selection_rubber_band.show()
                self.selection_rubber_band.raise_()
            elif (
                event.type() == QEvent.Type.MouseMove
                and self.rectangle_start_position is not None
                and event.buttons() & Qt.MouseButton.LeftButton
            ):
                current = event.position().toPoint()
                self.selection_rubber_band.setGeometry(
                    QRect(self.rectangle_start_position, current).normalized()
                )
            elif (
                event.type() == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self.selection_rubber_band.hide()
                self.rectangle_start_position = None
        if watched is self.plotter.interactor and self.picking_mode is None:
            if (
                event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                position = event.position()
                self.mouse_press_position = (position.x(), position.y())
            elif (
                event.type() == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton
                and self.mouse_press_position is not None
            ):
                position = event.position()
                start_x, start_y = self.mouse_press_position
                self.mouse_press_position = None
                if (
                    abs(position.x() - start_x) <= 4.0
                    and abs(position.y() - start_y) <= 4.0
                    and self.terrain_grid is not None
                ):
                    self._select_prediction_from_screen(position.x(), position.y())
        return super().eventFilter(watched, event)

    def _select_prediction_from_screen(self, x: float, y: float) -> None:
        actor = self.plotter.actors.get("terrain-prediction")
        if actor is None or self.terrain_grid is None:
            return
        picker = vtkPointPicker()
        picker.PickFromListOn()
        picker.AddPickList(actor)
        picker.SetTolerance(0.04)
        vtk_y = self.plotter.interactor.height() - y
        if not picker.Pick(x, vtk_y, 0.0, self.plotter.renderer):
            return
        point_id = int(picker.GetPointId())
        if point_id < 0:
            return
        cloud = self.terrain_grid.prediction_polydata()
        if point_id >= cloud.n_points:
            return
        region_id = int(np.asarray(cloud.point_data["RegionId"])[point_id])
        self._select_region_id(region_id)

    def _build_full_map_group(self) -> QGroupBox:
        group = CollapsibleGroupBox("完整三维地图（点击标题折叠）")
        layout = QVBoxLayout(group)
        load_button = QPushButton("加载完整 Poisson 网格")
        load_button.clicked.connect(self.choose_full_map_mesh)
        self.show_source_checkbox = QCheckBox("显示原始点云")
        self.show_source_checkbox.setChecked(True)
        self.show_source_checkbox.toggled.connect(self._show_scene)
        self.show_full_map_checkbox = QCheckBox("显示完整三维网格")
        self.show_full_map_checkbox.setChecked(False)
        self.show_full_map_checkbox.setEnabled(False)
        self.show_full_map_checkbox.toggled.connect(self._show_scene)
        export_button = QPushButton("导出已校准的完整地图")
        export_button.clicked.connect(self.export_full_map_mesh)
        self.full_map_summary_label = QLabel("尚未加载完整网格")
        self.full_map_summary_label.setWordWrap(True)
        layout.addWidget(load_button)
        layout.addWidget(self.show_source_checkbox)
        layout.addWidget(self.show_full_map_checkbox)
        layout.addWidget(export_button)
        layout.addWidget(self.full_map_summary_label)
        return group

    def _build_calibration_group(self) -> QGroupBox:
        group = CollapsibleGroupBox("坐标与尺度校准（点击标题折叠）")
        layout = QVBoxLayout(group)
        ground_button = QPushButton("选择3个地面点，校正Z轴")
        ground_button.clicked.connect(self.start_ground_calibration)
        scale_button = QPushButton("选择2个点，设置真实长度")
        scale_button.clicked.connect(self.start_scale_calibration)
        flip_button = QPushButton("如果上下颠倒：翻转Z方向")
        flip_button.clicked.connect(self.flip_vertical)
        layout.addWidget(ground_button)
        layout.addWidget(scale_button)
        layout.addWidget(flip_button)
        return group

    def _build_prediction_group(self) -> QGroupBox:
        group = CollapsibleGroupBox("地形预测与人工决策（点击标题折叠）")
        layout = QVBoxLayout(group)
        roi_button = QPushButton("选择2点确定地形矩形范围")
        roi_button.clicked.connect(self.start_roi_selection)
        predict_button = QPushButton("生成空白区域预测")
        predict_button.clicked.connect(self.generate_terrain_prediction)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem("二维边界约束完全补全（推荐）", "harmonic")
        self.algorithm_combo.addItem("局部线性曲面（基线）", "linear")
        self.algorithm_combo.addItem("行列 Kalman（旧版，仅用于对比）", "kalman")
        self.region_combo = QComboBox()
        self.region_combo.currentIndexChanged.connect(self._update_region_summary)
        map_select_button = QPushButton("在三维地图中单击选择预测区域")
        map_select_button.clicked.connect(self.start_map_region_selection)
        box_select_button = QPushButton("左键拖框，勾选多个预测区域")
        box_select_button.clicked.connect(self.start_rectangle_region_selection)
        finish_box_select_button = QPushButton("完成框选，恢复旋转（Esc）")
        finish_box_select_button.clicked.connect(self.cancel_picking)
        guide_points_button = QPushButton("选择周围参考点，拟合目标地表")
        guide_points_button.clicked.connect(self.start_fill_guide_selection)
        manual_fill_box_button = QPushButton("俯视框选人工补充范围")
        manual_fill_box_button.clicked.connect(self.start_manual_fill_rectangle)
        self.region_list = QListWidget()
        self.region_list.setMinimumHeight(100)
        self.region_list.setMaximumHeight(180)
        self.region_list.currentItemChanged.connect(self._select_region_from_list)
        self.region_list.itemChanged.connect(self._draw_checked_regions_highlight)
        self.region_summary_label = QLabel("尚未生成预测区域")
        self.region_summary_label.setWordWrap(True)
        self.candidate_validation_label = QLabel("尚未评价多个候选地形")
        self.candidate_validation_label.setWordWrap(True)
        self.fill_strategy_combo = QComboBox()
        for strategy in (
            FILL_AUTO,
            FILL_FLAT,
            FILL_PLANE,
            FILL_QUADRATIC,
            FILL_CONVEX,
            FILL_CONCAVE,
        ):
            self.fill_strategy_combo.addItem(FILL_STRATEGY_NAMES[strategy], strategy)
        apply_strategy_button = QPushButton("将填充策略应用到当前区域")
        apply_strategy_button.clicked.connect(self.apply_fill_strategy_to_current)
        evaluate_candidates_button = QPushButton("评价三个候选地形（几何＋图像）")
        evaluate_candidates_button.clicked.connect(self.evaluate_current_candidates)
        apply_recommended_button = QPushButton("应用软件推荐候选")
        apply_recommended_button.clicked.connect(self.apply_recommended_candidate)
        apply_batch_strategy_button = QPushButton("将填充策略应用到勾选区域")
        apply_batch_strategy_button.clicked.connect(self.apply_fill_strategy_to_checked)
        decision_row = QHBoxLayout()
        accept_button = QPushButton("接受")
        accept_button.clicked.connect(lambda: self.set_region_decision(ACCEPTED))
        uncertain_button = QPushButton("待定")
        uncertain_button.clicked.connect(lambda: self.set_region_decision(UNCERTAIN))
        reshoot_button = QPushButton("补拍")
        reshoot_button.clicked.connect(lambda: self.set_region_decision(RESHOOT))
        decision_row.addWidget(accept_button)
        decision_row.addWidget(uncertain_button)
        decision_row.addWidget(reshoot_button)
        selection_row = QHBoxLayout()
        select_unreviewed_button = QPushButton("勾选全部未判断")
        select_unreviewed_button.clicked.connect(self.select_unreviewed_regions)
        clear_selection_button = QPushButton("清空勾选")
        clear_selection_button.clicked.connect(self.clear_region_checks)
        selection_row.addWidget(select_unreviewed_button)
        selection_row.addWidget(clear_selection_button)
        batch_row = QHBoxLayout()
        batch_accept_button = QPushButton("批量接受")
        batch_accept_button.clicked.connect(
            lambda: self.apply_batch_decision(ACCEPTED)
        )
        batch_uncertain_button = QPushButton("批量待定")
        batch_uncertain_button.clicked.connect(
            lambda: self.apply_batch_decision(UNCERTAIN)
        )
        batch_reshoot_button = QPushButton("批量补拍")
        batch_reshoot_button.clicked.connect(
            lambda: self.apply_batch_decision(RESHOOT)
        )
        batch_row.addWidget(batch_accept_button)
        batch_row.addWidget(batch_uncertain_button)
        batch_row.addWidget(batch_reshoot_button)
        layout.addWidget(roi_button)
        layout.addWidget(QLabel("预测算法："))
        layout.addWidget(self.algorithm_combo)
        layout.addWidget(predict_button)
        layout.addWidget(QLabel("预测区域："))
        layout.addWidget(self.region_combo)
        layout.addWidget(map_select_button)
        layout.addWidget(box_select_button)
        layout.addWidget(finish_box_select_button)
        layout.addWidget(self.region_summary_label)
        layout.addWidget(evaluate_candidates_button)
        layout.addWidget(apply_recommended_button)
        layout.addWidget(self.candidate_validation_label)
        layout.addWidget(guide_points_button)
        layout.addWidget(manual_fill_box_button)
        layout.addWidget(QLabel("选择要应用的人工填充策略："))
        layout.addWidget(self.fill_strategy_combo)
        layout.addWidget(apply_strategy_button)
        layout.addLayout(decision_row)
        layout.addWidget(QLabel("批量处理（勾选多个区域）："))
        layout.addWidget(self.region_list)
        layout.addLayout(selection_row)
        layout.addWidget(apply_batch_strategy_button)
        layout.addLayout(batch_row)
        return group

    def _build_mesh_group(self) -> QGroupBox:
        group = CollapsibleGroupBox("规划地形网格（点击标题折叠）")
        layout = QVBoxLayout(group)
        generate_button = QPushButton("生成地形网格")
        generate_button.clicked.connect(self.generate_terrain_mesh)
        import_button = QPushButton("导入已经生成的地形网格")
        import_button.clicked.connect(self.import_terrain_mesh)
        self.show_terrain_mesh_checkbox = QCheckBox("显示规划地形网格")
        self.show_terrain_mesh_checkbox.setChecked(True)
        self.show_terrain_mesh_checkbox.setEnabled(False)
        self.show_terrain_mesh_checkbox.toggled.connect(self._show_scene)
        terrain_only_button = QPushButton("仅查看规划地形网格")
        terrain_only_button.clicked.connect(self.show_only_terrain_mesh)
        export_button = QPushButton("导出地形网格（PLY + VTP）")
        export_button.clicked.connect(self.export_terrain_mesh)
        clear_button = QPushButton("清除网格预览")
        clear_button.clicked.connect(self.clear_terrain_mesh)
        self.mesh_summary_label = QLabel("尚未生成地形网格")
        self.mesh_summary_label.setWordWrap(True)
        layout.addWidget(generate_button)
        layout.addWidget(import_button)
        layout.addWidget(self.show_terrain_mesh_checkbox)
        layout.addWidget(terrain_only_button)
        layout.addWidget(export_button)
        layout.addWidget(clear_button)
        layout.addWidget(self.mesh_summary_label)
        return group

    def _build_path_planning_group(self) -> QGroupBox:
        group = CollapsibleGroupBox("安全路径规划（点击标题折叠）")
        layout = QVBoxLayout(group)
        select_row = QHBoxLayout()
        start_button = QPushButton("选择起点")
        start_button.clicked.connect(self.start_route_start_selection)
        end_button = QPushButton("选择终点")
        end_button.clicked.connect(self.start_route_end_selection)
        select_row.addWidget(start_button)
        select_row.addWidget(end_button)

        settings_form = QFormLayout()
        self.route_max_slope_spin = QDoubleSpinBox()
        self.route_max_slope_spin.setRange(1.0, 89.0)
        self.route_max_slope_spin.setValue(35.0)
        self.route_max_slope_spin.setSuffix("°")
        self.route_slope_weight_spin = QDoubleSpinBox()
        self.route_slope_weight_spin.setRange(0.0, 20.0)
        self.route_slope_weight_spin.setValue(3.0)
        self.route_slope_weight_spin.setSingleStep(0.5)
        self.route_safety_weight_spin = QDoubleSpinBox()
        self.route_safety_weight_spin.setRange(0.0, 20.0)
        self.route_safety_weight_spin.setValue(2.0)
        self.route_safety_weight_spin.setSingleStep(0.5)
        settings_form.addRow("最大可通行坡度：", self.route_max_slope_spin)
        settings_form.addRow("坡度代价权重：", self.route_slope_weight_spin)
        settings_form.addRow("安全性代价权重：", self.route_safety_weight_spin)

        self.show_route_nodes_checkbox = QCheckBox("显示自动生成的可通行节点")
        self.show_route_nodes_checkbox.setChecked(False)
        self.show_route_nodes_checkbox.toggled.connect(self._show_scene)
        calculate_button = QPushButton("计算安全路径（Dijkstra）")
        calculate_button.clicked.connect(self.calculate_safe_route)
        clear_button = QPushButton("清除起点、终点和路径")
        clear_button.clicked.connect(self.clear_route)
        self.route_summary_label = QLabel("请先生成或导入地形网格")
        self.route_summary_label.setWordWrap(True)
        explanation = QLabel(
            "当前安全性由坡度、重建置信度和预测区域惩罚共同计算；"
            "地面材质风险接口已预留。"
        )
        explanation.setWordWrap(True)

        layout.addLayout(select_row)
        layout.addLayout(settings_form)
        layout.addWidget(self.show_route_nodes_checkbox)
        layout.addWidget(calculate_button)
        layout.addWidget(clear_button)
        layout.addWidget(self.route_summary_label)
        layout.addWidget(explanation)
        return group

    def _create_actions(self) -> None:
        open_action = QAction("打开模型", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.choose_model)
        save_action = QAction("保存工程", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        open_project_action = QAction("打开工程", self)
        open_project_action.triggered.connect(self.open_project)
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        reset_action = QAction("适应视图", self)
        reset_action.setShortcut("F")
        reset_action.triggered.connect(self.reset_camera)
        top_action = QAction("俯视", self)
        top_action.triggered.connect(self.view_top)
        density_action = QAction("分析点云密度", self)
        density_action.setShortcut("Ctrl+D")
        density_action.triggered.connect(self.analyze_density)
        density_settings_action = QAction("设置密度宽松程度", self)
        density_settings_action.triggered.connect(self.set_density_thresholds)
        original_action = QAction("显示原始颜色", self)
        original_action.triggered.connect(self.show_original_color)
        logic_action = QAction("查看标注逻辑", self)
        logic_action.triggered.connect(self.show_density_logic)
        cancel_picking_action = QAction("取消选点", self)
        cancel_picking_action.setShortcut("Esc")
        cancel_picking_action.triggered.connect(self.cancel_picking)
        self.addAction(cancel_picking_action)

        file_menu = self.menuBar().addMenu("文件")
        file_menu.addAction(open_action)
        file_menu.addAction(open_project_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        view_menu = self.menuBar().addMenu("视图")
        view_menu.addAction(reset_action)
        view_menu.addAction(top_action)
        analysis_menu = self.menuBar().addMenu("分析")
        analysis_menu.addAction(density_action)
        analysis_menu.addAction(density_settings_action)
        analysis_menu.addAction(original_action)
        analysis_menu.addSeparator()
        analysis_menu.addAction(logic_action)
        toolbar = self.addToolBar("常用工具")
        for action in (open_action, save_action, reset_action, top_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(density_action)
        toolbar.addAction(original_action)

    def choose_model(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择三维模型",
            "F:/Desktop/map_rescue/demo01/workspace1/dense/0",
            "PLY 模型 (*.ply)",
        )
        if filename:
            self.load_model(Path(filename))

    def load_model(self, path: Path) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage(f"正在读取 {path.name}……")
        QApplication.processEvents()
        try:
            self.scene = SceneModel.load(path)
            self.context_mesh = None
            self.show_source_checkbox.setChecked(True)
            self.show_full_map_checkbox.setChecked(False)
            self.show_full_map_checkbox.setEnabled(False)
            self.full_map_summary_label.setText("尚未加载完整网格")
            self._invalidate_derived_data(clear_roi=True)
            self._show_scene()
            self._show_summary()
            note = "（显示时已自动简化）" if self.scene.simplified else ""
            self.statusBar().showMessage(f"已加载：{path.name}{note}")
        except Exception as exc:
            QMessageBox.critical(self, "无法加载模型", str(exc))
            self.statusBar().showMessage("模型加载失败")
        finally:
            QApplication.restoreOverrideCursor()

    def _show_scene(self) -> None:
        self.plotter.disable_picking()
        self.plotter.clear()
        self.plotter.add_axes()
        self.plotter.show_grid(color="#66727d")
        if self.scene is not None:
            kwargs: dict[str, object] = {}
            if self.color_mode == "density" and "DensityRGB" in self.scene.display.point_data:
                kwargs.update(scalars="DensityRGB", rgb=True)
            else:
                rgb_name = self.scene.rgb_array_name()
                if rgb_name:
                    kwargs.update(scalars=rgb_name, rgb=True)
            if self.show_source_checkbox.isChecked():
                if self.scene.is_point_cloud:
                    self.plotter.add_mesh(
                        self.scene.display, name="base-model", style="points",
                        point_size=2, render_points_as_spheres=False, **kwargs,
                    )
                else:
                    self.plotter.add_mesh(
                        self.scene.display, name="base-model", smooth_shading=True,
                        show_edges=False, **kwargs,
                    )
        if (
            self.context_mesh is not None
            and self.show_full_map_checkbox.isChecked()
        ):
            full_map_kwargs: dict[str, object] = {}
            rgb_name = self.context_mesh.rgb_array_name()
            if rgb_name:
                full_map_kwargs.update(scalars=rgb_name, rgb=True)
            self.plotter.add_mesh(
                self.context_mesh.display,
                name="full-map-mesh",
                smooth_shading=True,
                show_edges=False,
                **full_map_kwargs,
            )
        self._draw_roi()
        self._draw_terrain_prediction()
        self._draw_checked_regions_highlight()
        self._draw_selected_region_highlight()
        self._draw_region_guide_points()
        self._draw_terrain_mesh()
        self._draw_path_planning()
        self.plotter.reset_camera()
        self.plotter.render()

    def choose_full_map_mesh(self) -> None:
        if self.scene is None:
            QMessageBox.information(self, "尚未加载点云", "请先打开 fused.ply。")
            return
        candidate = self.scene.path.parent / "meshed-poisson.ply"
        start_path = str(candidate if candidate.is_file() else self.scene.path.parent)
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择完整 Poisson 三角网格",
            start_path,
            "PLY 网格 (*.ply)",
        )
        if filename:
            self.load_full_map_mesh(Path(filename))

    def load_full_map_mesh(self, path: Path) -> None:
        if self.scene is None:
            raise RuntimeError("请先加载用于校准的 fused.ply。")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage(
            "正在加载并简化完整网格预览，模型较大时可能需要几十秒……"
        )
        QApplication.processEvents()
        try:
            context_mesh = SceneModel.load(path)
            context_mesh.apply_transform(self.scene.calibration_matrix)
            context_mesh.unit_name = self.scene.unit_name
            self.context_mesh = context_mesh
            self.show_full_map_checkbox.setEnabled(True)
            self.show_full_map_checkbox.setChecked(True)
            self.show_source_checkbox.setChecked(False)
            simplified = "（预览已简化，导出仍用完整模型）" if context_mesh.simplified else ""
            self.full_map_summary_label.setText(
                f"文件：{path.name}\n"
                f"完整顶点：{context_mesh.original.n_points:,}\n"
                f"完整三角面：{context_mesh.original.n_faces:,}\n"
                f"坐标单位：{context_mesh.unit_name}\n{simplified}"
            )
            self._show_scene()
            self.statusBar().showMessage(
                f"完整三维地图已加载：{context_mesh.original.n_faces:,}个三角面，"
                f"单位为{context_mesh.unit_name}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "加载完整三维地图失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def export_full_map_mesh(self) -> None:
        if self.context_mesh is None:
            QMessageBox.information(
                self, "尚无完整网格", "请先点击“加载完整 Poisson 网格”。"
            )
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出已校准的完整三维地图",
            "F:/Desktop/work/rescuemap/exports/full_map_calibrated.ply",
            "PLY 网格 (*.ply)",
        )
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() != ".ply":
            path = path.with_suffix(".ply")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("正在导出完整精度地图……")
        QApplication.processEvents()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            rgb_name = None
            for name in ("RGB", "RGBA", "rgb", "rgba", "Colors", "colors"):
                if name in self.context_mesh.original.point_data:
                    rgb_name = name
                    break
            self.context_mesh.original.save(
                path,
                binary=True,
                texture=rgb_name,
                recompute_normals=False,
            )
            QMessageBox.information(
                self,
                "完整地图已导出",
                f"文件：\n{path}\n\n"
                f"单位：{self.context_mesh.unit_name}\n"
                "该文件已经使用与点云相同的Z轴和尺度校准。\n\n"
                "在Blender中请使用：文件 → 导入 → Stanford (.ply)。",
            )
            self.statusBar().showMessage(f"已导出完整三维地图：{path}")
        except Exception as exc:
            QMessageBox.critical(self, "导出完整地图失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _show_summary(self) -> None:
        while self.info_form.rowCount():
            self.info_form.removeRow(0)
        if self.scene is None:
            self.info_form.addRow(QLabel("尚未加载模型"))
            return
        for label, value in self.scene.summary().items():
            value_label = QLabel(value)
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.info_form.addRow(f"{label}：", value_label)
        if self.density_result is not None:
            self._append_density_summary(self.density_result)

    def _invalidate_derived_data(self, *, clear_roi: bool) -> None:
        self.density_result = None
        self.scene_point_tree = None
        self.color_mode = "original"
        self.terrain_grid = None
        self.terrain_settings = None
        self.candidate_results.clear()
        self.terrain_mesh = None
        self.terrain_mesh_settings = None
        self.terrain_mesh_path = None
        self.show_terrain_mesh_checkbox.setChecked(True)
        self.show_terrain_mesh_checkbox.setEnabled(False)
        self.mesh_summary_label.setText("尚未生成地形网格")
        self._invalidate_path_planning()
        self.region_combo.clear()
        self.region_list.clear()
        self.region_summary_label.setText("尚未生成预测区域")
        self.candidate_validation_label.setText("尚未评价多个候选地形")
        self.color_legend_widget.hide()
        if clear_roi:
            self.terrain_roi = None

    def _start_picking(self, mode: str, target: int, message: str) -> None:
        if self.scene is None and not (
            mode in ("path-start", "path-end") and self.terrain_mesh is not None
        ):
            QMessageBox.information(self, "尚未加载模型", "请先打开 fused.ply。")
            return
        self.plotter.disable_picking()
        self.plotter.enable_trackball_style()
        self.picking_mode = mode
        self.picking_target = target
        self.picked_points = []
        left_clicking = mode in ("path-start", "path-end")
        self.cancel_picking_button.setText("取消选点（Esc）")
        self._show_picking_help(
            f"选点进行中：{message}\n"
            + (
                "左键单击完成选择；按Esc取消。"
                if left_clicking
                else "右键添加点；左键拖动仍可旋转地图。"
            )
        )
        self.statusBar().showMessage(f"{message}；按Esc可随时取消")
        self.plotter.enable_point_picking(
            callback=self._on_point_picked, left_clicking=left_clicking,
            show_message=False, show_point=False, clear_on_no_selection=False,
        )

    def cancel_picking(self) -> None:
        was_rectangle = self.picking_mode in ("region-box", "fill-box")
        self.plotter.disable_picking()
        self.plotter.enable_trackball_style()
        self.picking_mode = None
        self.guide_target_region_id = None
        self.manual_fill_target_region_id = None
        self.picked_points = []
        self.mouse_press_position = None
        self.rectangle_start_position = None
        self.selection_rubber_band.hide()
        self.plotter.remove_actor(
            "selection-points", reset_camera=False, render=False
        )
        self.picking_help_widget.hide()
        self.cancel_picking_button.setText("取消选点（Esc）")
        self.plotter.render()
        self.statusBar().showMessage(
            "框选完成，已恢复左键旋转地图"
            if was_rectangle else "已取消选点；可继续旋转和查看地图"
        )

    def _on_point_picked(self, point) -> None:
        if point is None or self.picking_mode is None:
            return
        picked = np.asarray(point, dtype=float)
        if picked.shape != (3,) or not np.all(np.isfinite(picked)):
            return
        self.picked_points.append(picked)
        labels = [f"P{i + 1}" for i in range(len(self.picked_points))]
        self.plotter.add_point_labels(
            np.asarray(self.picked_points), labels, name="selection-points",
            point_color="cyan", text_color="white", point_size=12,
            always_visible=True,
        )
        self.statusBar().showMessage(
            f"已选择 {len(self.picked_points)}/{self.picking_target} 个点"
        )
        if len(self.picked_points) < self.picking_target:
            return
        mode = self.picking_mode
        points = np.asarray(self.picked_points)
        self.plotter.disable_picking()
        self.picking_mode = None
        self.picking_help_widget.hide()
        self.plotter.remove_actor(
            "selection-points", reset_camera=False, render=False
        )
        if mode == "ground":
            self._finish_ground_calibration(points)
        elif mode == "scale":
            self._finish_scale_calibration(points)
        elif mode == "roi":
            self._finish_roi_selection(points)
        elif mode == "region":
            self._finish_map_region_selection(points[0])
        elif mode == "fill-guides":
            self._finish_fill_guide_selection(points)
        elif mode == "path-start":
            self._finish_route_point(points[0], is_start=True)
        elif mode == "path-end":
            self._finish_route_point(points[0], is_start=False)

    def start_ground_calibration(self) -> None:
        self._start_picking("ground", 3, "请依次点击三个地面点；点选顺序决定Z轴方向")

    def _apply_calibration_transform(self, transform: np.ndarray) -> None:
        if self.scene is None:
            raise RuntimeError("尚未加载点云。")
        self.scene.apply_transform(transform)
        if self.context_mesh is not None:
            self.context_mesh.apply_transform(transform)

    def _finish_ground_calibration(self, points: np.ndarray) -> None:
        try:
            transform = ground_alignment_transform(points)
            self._apply_calibration_transform(transform)
            self._invalidate_derived_data(clear_roi=True)
            self._show_scene()
            self._show_summary()
            self.view_top()
            QMessageBox.information(
                self, "坐标系已校正",
                "所选平面已变换到Z=0，第一个点成为原点。若模型上下颠倒，请点击“翻转Z方向”。",
            )
        except Exception as exc:
            QMessageBox.critical(self, "坐标校正失败", str(exc))

    def start_scale_calibration(self) -> None:
        self._start_picking("scale", 2, "请选择实际距离已知的两个点")

    def _finish_scale_calibration(self, points: np.ndarray) -> None:
        model_distance = float(np.linalg.norm(points[1] - points[0]))
        if model_distance <= 1e-12:
            QMessageBox.warning(self, "尺度标定失败", "两个点距离过近，请重新选择。")
            return
        real_length, ok = QInputDialog.getDouble(
            self, "输入实际长度",
            f"模型距离：{model_distance:.6f}\n请输入现实长度（米）：",
            1.0, 0.000001, 1_000_000.0, 6,
        )
        if not ok:
            return
        scale = real_length / model_distance
        self._apply_calibration_transform(uniform_scale_transform(scale))
        self.scene.unit_name = "米"  # type: ignore[union-attr]
        if self.context_mesh is not None:
            self.context_mesh.unit_name = "米"
        self._invalidate_derived_data(clear_roi=True)
        self._show_scene()
        self._show_summary()
        self.statusBar().showMessage(
            f"尺度标定完成：{model_distance:.6f} → {real_length:.6f}米，比例 {scale:.6f}"
        )

    def flip_vertical(self) -> None:
        if self.scene is None:
            return
        self._apply_calibration_transform(flip_vertical_transform())
        self._invalidate_derived_data(clear_roi=True)
        self._show_scene()
        self._show_summary()
        self.statusBar().showMessage("已翻转Z方向")

    def start_roi_selection(self) -> None:
        self._start_picking("roi", 2, "请选择地形矩形范围的两个对角点")

    def start_map_region_selection(self) -> None:
        if self.terrain_grid is None:
            QMessageBox.information(self, "没有预测区域", "请先生成空白区域预测。")
            return
        self._show_picking_help(
            "区域选择无需进入选点模式：直接左键单击黄色/彩色预测点即可；"
            "左键拖动仍然旋转地图。"
        )
        self.cancel_picking_button.setText("关闭提示（Esc）")
        self.statusBar().showMessage("请直接左键单击预测点；按Esc可隐藏提示")

    def start_rectangle_region_selection(self) -> None:
        if self.terrain_grid is None:
            QMessageBox.information(self, "没有预测区域", "请先生成空白区域预测。")
            return
        self.cancel_picking()
        self.picking_mode = "region-box"
        self.cancel_picking_button.setText("完成框选，恢复旋转（Esc）")
        self._show_picking_help(
            "框选模式：按住鼠标左键拖出矩形，矩形内预测点所属区域会加入批量勾选；"
            "可以连续框选，按Esc或点击取消按钮退出。"
        )
        self.statusBar().showMessage("左键拖框选择预测区域；按Esc退出框选模式")
        self.plotter.enable_rectangle_picking(
            callback=self._on_rectangle_region_selection,
            show_message=False,
            start=True,
            show_frustum=False,
            color="#22d3ee",
        )

    def _on_rectangle_region_selection(self, selection) -> None:
        if self.terrain_grid is None:
            return
        cloud = self.terrain_grid.prediction_polydata()
        extractor = vtkExtractSelectedFrustum()
        extractor.SetInputData(cloud)
        extractor.SetFrustum(selection.frustum)
        extractor.PreserveTopologyOff()
        extractor.Update()
        selected = pv.wrap(extractor.GetOutput())
        if selected.n_points == 0 or "RegionId" not in selected.point_data:
            self.statusBar().showMessage("框内没有预测点；可继续框选或按Esc退出")
            return
        region_ids = sorted(
            set(map(int, np.asarray(selected.point_data["RegionId"]).tolist()))
        )
        self.region_list.blockSignals(True)
        for row in range(self.region_list.count()):
            item = self.region_list.item(row)
            region_id = int(item.data(Qt.ItemDataRole.UserRole))
            if region_id in region_ids:
                item.setCheckState(Qt.CheckState.Checked)
        self.region_list.blockSignals(False)
        self._draw_checked_regions_highlight()
        self.statusBar().showMessage(
            f"本次框选加入{len(region_ids)}个区域；完成后点击“恢复旋转”或按Esc"
        )

    def start_manual_fill_rectangle(self) -> None:
        region_id = self._current_region_id()
        if self.terrain_grid is None or region_id is None:
            QMessageBox.information(self, "没有预测区域", "请先选择一个预测区域。")
            return
        guide_points = self.terrain_grid.guide_points_for(region_id)
        if len(guide_points) < 3:
            QMessageBox.information(
                self,
                "请先选择参考地表",
                "人工补充范围可能覆盖同一XY位置的其他点层。\n"
                "请先为当前区域选择至少3个周围参考点，再框选补充范围。",
            )
            return
        strategy = str(self.fill_strategy_combo.currentData())
        if strategy == FILL_AUTO:
            strategy = FILL_PLANE
            self.fill_strategy_combo.setCurrentIndex(
                self.fill_strategy_combo.findData(FILL_PLANE)
            )
        amplitude = self._strategy_amplitude(strategy)
        if amplitude is None:
            return
        self.cancel_picking()
        self.manual_fill_target_region_id = region_id
        self.manual_fill_strategy = strategy
        self.manual_fill_amplitude = float(amplitude)
        self.picking_mode = "fill-box"
        self.view_top()
        self.cancel_picking_button.setText("完成补充框选，恢复旋转（Esc）")
        self._show_picking_help(
            f"人工补充范围：当前区域{region_id}，策略“{FILL_STRATEGY_NAMES[strategy]}”。\n"
            "已自动切换到俯视图；按住左键拖出矩形。可连续框选，按Esc完成。"
        )
        self.statusBar().showMessage("左键拖框指定需要新建预测点的XY范围；按Esc完成")
        self.plotter.enable_rectangle_picking(
            callback=self._on_manual_fill_rectangle,
            show_message=False,
            start=True,
            show_frustum=False,
            color="#d946ef",
        )

    def _on_manual_fill_rectangle(self, selection) -> None:
        if (
            self.terrain_grid is None
            or self.manual_fill_target_region_id is None
        ):
            return
        region_id = self.manual_fill_target_region_id
        guide_points = self.terrain_grid.guide_points_for(region_id)
        reference_z = float(np.median(guide_points[:, 2]))
        try:
            bounds = self._rectangle_xy_bounds(selection.viewport, reference_z)
            count = self.terrain_grid.add_manual_fill_rectangle(region_id, bounds)
            self.candidate_results.pop(region_id, None)
            self.terrain_grid.apply_fill_strategy(
                region_id,
                self.manual_fill_strategy,
                self.manual_fill_amplitude,
            )
            self.terrain_mesh = None
            self.terrain_mesh_settings = None
            self._invalidate_path_planning()
            self.mesh_summary_label.setText("人工补充范围已改变，请重新生成地形网格")
            self._refresh_region_list_text()
            self._update_region_summary()
            self._draw_terrain_prediction()
            self._draw_checked_regions_highlight()
            self._draw_selected_region_highlight()
            self._draw_region_guide_points()
            self.plotter.render()
            self.statusBar().showMessage(
                f"区域{region_id}新增/覆盖{count}个规则栅格；可继续框选或按Esc完成"
            )
        except Exception as exc:
            QMessageBox.warning(self, "人工补充范围无效", str(exc))

    def _rectangle_xy_bounds(
        self,
        viewport: tuple[float, float, float, float],
        reference_z: float,
    ) -> tuple[float, float, float, float]:
        """Project a top-view screen rectangle onto a horizontal XY plane."""
        renderer = self.plotter.renderer
        x0, y0, x1, y1 = map(float, viewport)
        world_points: list[np.ndarray] = []
        for x, y in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
            endpoints: list[np.ndarray] = []
            for display_z in (0.0, 1.0):
                renderer.SetDisplayPoint(x, y, display_z)
                renderer.DisplayToWorld()
                homogeneous = np.asarray(renderer.GetWorldPoint(), dtype=float)
                if abs(float(homogeneous[3])) <= 1e-12:
                    raise ValueError("无法把屏幕框选范围转换为地形坐标。")
                endpoints.append(homogeneous[:3] / homogeneous[3])
            direction = endpoints[1] - endpoints[0]
            if abs(float(direction[2])) <= 1e-12:
                raise ValueError("当前视角不是有效俯视角，请重新进入框选功能。")
            factor = (reference_z - endpoints[0][2]) / direction[2]
            world_points.append(endpoints[0] + factor * direction)
        xy = np.asarray(world_points)[:, :2]
        return (
            float(np.min(xy[:, 0])),
            float(np.max(xy[:, 0])),
            float(np.min(xy[:, 1])),
            float(np.max(xy[:, 1])),
        )

    def _finish_map_region_selection(
        self, point: np.ndarray, *, silent: bool = False
    ) -> None:
        if self.terrain_grid is None:
            return
        prediction_mask = self.terrain_grid.prediction_mask()
        effective_labels = self.terrain_grid.effective_region_labels()
        ix, iy = np.where(prediction_mask)
        prediction_points = np.column_stack(
            (
                self.terrain_grid.x_centers[ix],
                self.terrain_grid.y_centers[iy],
                self.terrain_grid.predicted_height[ix, iy],
            )
        )
        if len(prediction_points) == 0:
            return
        distances = np.linalg.norm(prediction_points - point, axis=1)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) > self.terrain_grid.cell_size * 3.0:
            if not silent:
                QMessageBox.information(
                    self,
                    "没有点中预测区域",
                    "点击位置离预测点较远，请重新点击黄色预测点。",
                )
            return
        region_id = int(effective_labels[ix[nearest], iy[nearest]])
        self._select_region_id(region_id)

    def _select_region_id(self, region_id: int) -> None:
        if self.terrain_grid is None:
            return
        if region_id < 1 or region_id > self.terrain_grid.region_count:
            return
        combo_index = self.region_combo.findData(region_id)
        if combo_index >= 0:
            self.region_combo.setCurrentIndex(combo_index)
        for row in range(self.region_list.count()):
            item = self.region_list.item(row)
            if int(item.data(Qt.ItemDataRole.UserRole)) == region_id:
                self.region_list.setCurrentRow(row)
                self.region_list.scrollToItem(item)
                break
        self._update_region_summary()
        self.picking_help_widget.hide()
        self.statusBar().showMessage(f"已在地图中选择预测区域 {region_id}")

    def start_fill_guide_selection(self) -> None:
        region_id = self._current_region_id()
        if self.terrain_grid is None or region_id is None:
            QMessageBox.information(self, "没有预测区域", "请先选择一个预测区域。")
            return
        if not np.any(self.terrain_grid.region_mask(region_id)):
            self._populate_regions()
            region_id = self._current_region_id()
            if region_id is None or not np.any(
                self.terrain_grid.region_mask(region_id)
            ):
                QMessageBox.information(
                    self,
                    "区域已经变化",
                    "预测区域已经重新生成，请重新在三维地图中选择目标区域。",
                )
                return
        count, ok = QInputDialog.getInt(
            self,
            "选择参考地表点",
            "准备在正确的地表层上选择几个参考点？\n"
            "建议从空白区域四周选择6～12个点，尽量覆盖多个方向。",
            8,
            3,
            30,
            1,
        )
        if not ok:
            return
        self.guide_target_region_id = region_id
        self._start_picking(
            "fill-guides",
            count,
            f"请在区域{region_id}周围正确的同一地表层上选择{count}个参考点",
        )

    def _finish_fill_guide_selection(self, points: np.ndarray) -> None:
        if (
            self.scene is None
            or self.terrain_grid is None
            or self.guide_target_region_id is None
        ):
            return
        # Keep the guide points attached to the region chosen before picking.
        # Automatically guessing another region from the picked-point centroid
        # made the operation unpredictable when upper and lower surfaces
        # overlapped in XY.  This is the stable behaviour used by the earlier
        # version of the tool.
        region_id = self.guide_target_region_id
        self.guide_target_region_id = None
        if not np.any(self.terrain_grid.region_mask(region_id)):
            self._populate_regions()
            QMessageBox.warning(
                self,
                "区域已经变化",
                "选点期间预测区域发生了变化，请重新选择目标区域。",
            )
            return
        if self.scene_point_tree is None:
            self.scene_point_tree = cKDTree(np.asarray(self.scene.original.points))
        _, indices = self.scene_point_tree.query(points, k=1)
        snapped_points = np.asarray(self.scene.original.points)[np.asarray(indices)]
        try:
            self.terrain_grid.set_guide_points(region_id, snapped_points)
            self.candidate_results.pop(region_id, None)
            requested_strategy = str(self.fill_strategy_combo.currentData())
            fitted_strategy = (
                FILL_QUADRATIC
                if requested_strategy == FILL_QUADRATIC and len(snapped_points) >= 6
                else FILL_PLANE
            )
            self.terrain_grid.apply_fill_strategy(region_id, fitted_strategy)
            self.fill_strategy_combo.setCurrentIndex(
                self.fill_strategy_combo.findData(fitted_strategy)
            )
            self.terrain_mesh = None
            self.terrain_mesh_settings = None
            self._invalidate_path_planning()
            self.mesh_summary_label.setText("参考地表已改变，请重新生成地形网格")
            self._refresh_region_list_text()
            self._update_region_summary()
            self._show_scene()
            self.statusBar().showMessage(
                f"区域{region_id}已使用{len(snapped_points)}个参考点拟合"
                f"{FILL_STRATEGY_NAMES[fitted_strategy]}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "参考点拟合失败", str(exc))

    def _finish_roi_selection(self, points: np.ndarray) -> None:
        xmin, xmax = sorted((float(points[0, 0]), float(points[1, 0])))
        ymin, ymax = sorted((float(points[0, 1]), float(points[1, 1])))
        if xmax - xmin <= 1e-9 or ymax - ymin <= 1e-9:
            QMessageBox.warning(self, "范围无效", "请选择XY方向上不同的两个点。")
            return
        self.terrain_roi = (xmin, xmax, ymin, ymax)
        self.terrain_grid = None
        self.terrain_settings = None
        self.candidate_results.clear()
        self.terrain_mesh = None
        self.terrain_mesh_settings = None
        self._invalidate_path_planning()
        self.mesh_summary_label.setText("尚未生成地形网格")
        self.region_combo.clear()
        self.region_list.clear()
        self._show_scene()
        self.statusBar().showMessage(
            f"地形范围：X {xmin:.3f}～{xmax:.3f}，Y {ymin:.3f}～{ymax:.3f}"
        )

    def _draw_roi(self) -> None:
        if self.terrain_roi is None or self.scene is None:
            return
        xmin, xmax, ymin, ymax = self.terrain_roi
        z = float(self.scene.original.bounds[4])
        corners = np.array(
            [[xmin, ymin, z], [xmax, ymin, z], [xmax, ymax, z], [xmin, ymax, z]]
        )
        self.plotter.add_mesh(
            pv.lines_from_points(corners, close=True), name="terrain-roi",
            color="cyan", line_width=4,
        )

    def generate_terrain_prediction(self) -> None:
        if self.scene is None or not self.scene.is_point_cloud:
            QMessageBox.information(self, "需要点云", "请打开 fused.ply。")
            return
        if self.terrain_roi is None:
            QMessageBox.information(self, "尚未选择范围", "请先选择地形矩形范围。")
            return
        xmin, xmax, ymin, ymax = self.terrain_roi
        default_size = max(xmax - xmin, ymax - ymin) / 100.0
        cell_size, ok = QInputDialog.getDouble(
            self, "设置地形栅格",
            f"请输入栅格边长（当前单位：{self.scene.unit_name}）：",
            max(default_size, 0.000001), 0.000001,
            max(xmax - xmin, ymax - ymin), 6,
        )
        if not ok:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("正在生成高度栅格并进行二维空白补全……")
        QApplication.processEvents()
        try:
            algorithm = str(self.algorithm_combo.currentData())
            self.terrain_grid = build_terrain_grid(
                self.scene.original.points, bounds_xy=self.terrain_roi,
                cell_size=cell_size, min_points=3,
                max_prediction_distance_cells=4.0,
                algorithm=algorithm,
            )
            self.candidate_results.clear()
            self.terrain_settings = {
                "cell_size": cell_size,
                "min_points": 3,
                "max_prediction_distance_cells": 4.0,
                "algorithm": algorithm,
            }
            self.terrain_mesh = None
            self.terrain_mesh_settings = None
            self._invalidate_path_planning()
            self.mesh_summary_label.setText("尚未生成地形网格")
            self._populate_regions()
            self._show_scene()
            self.statusBar().showMessage(
                f"预测完成（{algorithm}）：{self.terrain_grid.region_count}个候选区域；最终是否补拍由人工决定"
            )
        except Exception as exc:
            QMessageBox.critical(self, "地形预测失败", str(exc))
            self.statusBar().showMessage("地形预测失败")
        finally:
            QApplication.restoreOverrideCursor()

    def _populate_regions(self) -> None:
        self.region_combo.blockSignals(True)
        self.region_combo.clear()
        self.region_list.blockSignals(True)
        self.region_list.clear()
        if self.terrain_grid is not None:
            for region_id in range(1, self.terrain_grid.region_count + 1):
                if not np.any(self.terrain_grid.region_mask(region_id)):
                    continue
                self.region_combo.addItem(f"区域 {region_id}", region_id)
                item = QListWidgetItem(self._region_list_text(region_id))
                item.setData(Qt.ItemDataRole.UserRole, region_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.region_list.addItem(item)
        self.region_combo.blockSignals(False)
        self.region_list.blockSignals(False)
        self._update_region_summary()

    def _region_list_text(self, region_id: int) -> str:
        if self.terrain_grid is None:
            return f"区域 {region_id}"
        info = self.terrain_grid.region_summary(region_id)
        display = {
            PREDICTED: "未判断",
            ACCEPTED: "接受",
            UNCERTAIN: "待定",
            RESHOOT: "补拍",
        }[str(info["decision"])]
        strategy = FILL_STRATEGY_NAMES[str(info["fill_strategy"])]
        return f"区域 {region_id}｜{info['cells']}格｜{strategy}｜{display}"

    def _refresh_region_list_text(self) -> None:
        if self.terrain_grid is None:
            return
        self.region_list.blockSignals(True)
        self.region_combo.blockSignals(True)
        for index in range(self.region_list.count() - 1, -1, -1):
            item = self.region_list.item(index)
            region_id = int(item.data(Qt.ItemDataRole.UserRole))
            if not np.any(self.terrain_grid.region_mask(region_id)):
                self.region_list.takeItem(index)
                combo_index = self.region_combo.findData(region_id)
                if combo_index >= 0:
                    self.region_combo.removeItem(combo_index)
                continue
            item.setText(self._region_list_text(region_id))
        self.region_list.blockSignals(False)
        self.region_combo.blockSignals(False)

    def _select_region_from_list(
        self, current: QListWidgetItem | None, previous: QListWidgetItem | None
    ) -> None:
        del previous
        if current is None:
            return
        region_id = int(current.data(Qt.ItemDataRole.UserRole))
        combo_index = self.region_combo.findData(region_id)
        if combo_index >= 0:
            self.region_combo.setCurrentIndex(combo_index)
        self._update_region_summary()

    def select_unreviewed_regions(self) -> None:
        if self.terrain_grid is None:
            return
        self.region_list.blockSignals(True)
        for index in range(self.region_list.count()):
            item = self.region_list.item(index)
            region_id = int(item.data(Qt.ItemDataRole.UserRole))
            checked = self.terrain_grid.decision_for(region_id) == PREDICTED
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
        self.region_list.blockSignals(False)
        self._draw_checked_regions_highlight()

    def clear_region_checks(self) -> None:
        self.region_list.blockSignals(True)
        for index in range(self.region_list.count()):
            self.region_list.item(index).setCheckState(Qt.CheckState.Unchecked)
        self.region_list.blockSignals(False)
        self._draw_checked_regions_highlight()

    def _checked_region_ids(self) -> list[int]:
        result: list[int] = []
        for index in range(self.region_list.count()):
            item = self.region_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                result.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return result

    def apply_batch_decision(self, decision: str) -> None:
        if self.terrain_grid is None:
            QMessageBox.information(self, "没有预测区域", "请先生成空白区域预测。")
            return
        region_ids = self._checked_region_ids()
        if not region_ids:
            QMessageBox.information(
                self, "尚未勾选", "请先在批量处理列表中勾选一个或多个区域。"
            )
            return
        for region_id in region_ids:
            self.terrain_grid.set_decision(region_id, decision)
        self.terrain_mesh = None
        self.terrain_mesh_settings = None
        self._invalidate_path_planning()
        self.mesh_summary_label.setText("人工状态已改变，请重新生成地形网格")
        self._refresh_region_list_text()
        self.clear_region_checks()
        self._update_region_summary()
        self._show_scene()
        self.statusBar().showMessage(
            f"已将{len(region_ids)}个区域批量设为{decision}"
        )

    def _draw_terrain_prediction(self) -> None:
        if self.terrain_grid is None:
            self.color_legend_widget.hide()
            return
        cloud = self.terrain_grid.prediction_polydata()
        if cloud.n_points == 0:
            self.color_legend_widget.hide()
            return
        self.plotter.add_mesh(
            cloud, name="terrain-prediction", style="points", point_size=7,
            render_points_as_spheres=True, scalars="DecisionRGB", rgb=True,
            opacity=0.72,
        )
        self.color_legend_widget.show()
        self._position_color_legend()

    def _draw_checked_regions_highlight(self, item=None) -> None:
        del item
        if self.terrain_grid is None:
            self.plotter.remove_actor(
                "checked-terrain-regions", reset_camera=False, render=False
            )
            return
        region_ids = self._checked_region_ids()
        if not region_ids:
            self.plotter.remove_actor(
                "checked-terrain-regions", reset_camera=False, render=False
            )
            self.plotter.render()
            return
        mask = np.isin(
            self.terrain_grid.effective_region_labels(), region_ids
        ) & self.terrain_grid.prediction_mask()
        ix, iy = np.where(mask)
        points = np.column_stack(
            (
                self.terrain_grid.x_centers[ix],
                self.terrain_grid.y_centers[iy],
                self.terrain_grid.predicted_height[ix, iy],
            )
        )
        self.plotter.add_mesh(
            pv.PolyData(points),
            name="checked-terrain-regions",
            style="points",
            color="#22d3ee",
            point_size=9,
            render_points_as_spheres=True,
            opacity=0.95,
            reset_camera=False,
        )
        self.plotter.render()

    def _draw_selected_region_highlight(self) -> None:
        region_id = self._current_region_id()
        if self.terrain_grid is None or region_id is None:
            self.plotter.remove_actor(
                "selected-terrain-region", reset_camera=False, render=False
            )
            self.plotter.remove_actor(
                "selected-terrain-surface", reset_camera=False, render=False
            )
            return
        mask = self.terrain_grid.region_mask(region_id)
        ix, iy = np.where(mask)
        if len(ix) == 0:
            return
        points = np.column_stack(
            (
                self.terrain_grid.x_centers[ix],
                self.terrain_grid.y_centers[iy],
                self.terrain_grid.predicted_height[ix, iy],
            )
        )
        preview = self.terrain_grid.region_surface_polydata(region_id)
        if preview.n_cells > 0:
            self.plotter.add_mesh(
                preview,
                name="selected-terrain-surface",
                color="#d946ef",
                opacity=0.24,
                show_edges=False,
                smooth_shading=False,
                reset_camera=False,
            )
        else:
            self.plotter.remove_actor(
                "selected-terrain-surface", reset_camera=False, render=False
            )
        self.plotter.add_mesh(
            pv.PolyData(points),
            name="selected-terrain-region",
            style="points",
            color="#d946ef",
            point_size=10,
            render_points_as_spheres=True,
            opacity=1.0,
            reset_camera=False,
        )
        self.plotter.render()

    def _draw_region_guide_points(self) -> None:
        region_id = self._current_region_id()
        if self.terrain_grid is None or region_id is None:
            self.plotter.remove_actor(
                "terrain-guide-points", reset_camera=False, render=False
            )
            return
        guide_points = self.terrain_grid.guide_points_for(region_id)
        if len(guide_points) == 0:
            self.plotter.remove_actor(
                "terrain-guide-points", reset_camera=False, render=False
            )
            return
        self.plotter.add_mesh(
            pv.PolyData(guide_points),
            name="terrain-guide-points",
            style="points",
            color="#84cc16",
            point_size=20,
            render_points_as_spheres=True,
            opacity=1.0,
            reset_camera=False,
        )
        self.plotter.render()

    def _current_region_id(self) -> int | None:
        data = self.region_combo.currentData()
        return int(data) if data is not None else None

    def _update_region_summary(self) -> None:
        region_id = self._current_region_id()
        if self.terrain_grid is None or region_id is None:
            self.region_summary_label.setText("尚未生成预测区域")
            self.candidate_validation_label.setText("尚未评价多个候选地形")
            self._draw_selected_region_highlight()
            self._draw_region_guide_points()
            return
        info = self.terrain_grid.region_summary(region_id)
        unit = self.scene.unit_name if self.scene else "单位"
        strategy = str(info["fill_strategy"])
        parameter_text = ""
        if strategy in (FILL_CONVEX, FILL_CONCAVE):
            parameter_text = f"（强度 {info['fill_parameter']:.4f} {unit}）"
        self.region_summary_label.setText(
            f"格子数：{info['cells']}\n面积：{info['area']:.4f} {unit}²\n"
            f"最远离观测点：{info['max_gap']:.4f} {unit}\n"
            f"人工参考点：{info['guide_points']}个\n"
            f"填充策略：{FILL_STRATEGY_NAMES[strategy]}{parameter_text}\n"
            f"人工状态：{info['decision']}"
        )
        result = self.candidate_results.get(region_id)
        if result is None:
            self.candidate_validation_label.setText("尚未评价多个候选地形")
        else:
            lines = [result.context_message]
            for score in result.scores:
                photo_text = (
                    f"{score.photometric_score:.3f}"
                    if score.photometric_score is not None
                    else "无证据"
                )
                lines.append(
                    f"{FILL_STRATEGY_NAMES[score.strategy]}：总分 {score.final_confidence:.3f}，"
                    f"几何 {score.geometry_score:.3f}，模型一致 {score.model_agreement_score:.3f}，"
                    f"纹理 {score.texture_strength:.3f}，图像 {photo_text}，"
                    f"有效视角 {score.effective_views}"
                )
            lines.append(
                f"推荐：{FILL_STRATEGY_NAMES[result.recommended_strategy]}"
                + ("（已使用图像证据）" if result.image_evidence_available else "（仅几何评价）")
            )
            self.candidate_validation_label.setText("\n".join(lines))
        self._draw_selected_region_highlight()
        self._draw_region_guide_points()

    def evaluate_current_candidates(self) -> None:
        region_id = self._current_region_id()
        if self.scene is None or self.terrain_grid is None or region_id is None:
            QMessageBox.information(self, "没有预测区域", "请先生成并选择一个预测区域。")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("正在生成三个候选曲面并进行反投影评价……")
        QApplication.processEvents()
        try:
            result = evaluate_region_candidates(
                self.terrain_grid,
                region_id,
                model_path=self.scene.path,
                calibration_matrix=self.scene.calibration_matrix,
            )
            self.candidate_results[region_id] = result
            self._update_region_summary()
            self.statusBar().showMessage(
                f"区域{region_id}评价完成，推荐“"
                f"{FILL_STRATEGY_NAMES[result.recommended_strategy]}”"
            )
        except Exception as exc:
            QMessageBox.critical(self, "候选地形评价失败", str(exc))
            self.statusBar().showMessage("候选地形评价失败")
        finally:
            QApplication.restoreOverrideCursor()

    def apply_recommended_candidate(self) -> None:
        region_id = self._current_region_id()
        if self.terrain_grid is None or region_id is None:
            QMessageBox.information(self, "没有预测区域", "请先生成并选择一个预测区域。")
            return
        result = self.candidate_results.get(region_id)
        if result is None:
            QMessageBox.information(
                self, "尚未评价", "请先点击“评价三个候选地形（几何＋图像）”。"
            )
            return
        try:
            strategy = result.recommended_strategy
            self.terrain_grid.apply_fill_strategy(region_id, strategy)
            combo_index = self.fill_strategy_combo.findData(strategy)
            if combo_index >= 0:
                self.fill_strategy_combo.setCurrentIndex(combo_index)
            self.terrain_mesh = None
            self.terrain_mesh_settings = None
            self._invalidate_path_planning()
            self.mesh_summary_label.setText("推荐候选已应用，请重新生成地形网格")
            self._refresh_region_list_text()
            self._update_region_summary()
            self._show_scene()
            self.statusBar().showMessage(
                f"已对区域{region_id}应用“{FILL_STRATEGY_NAMES[strategy]}”"
            )
        except Exception as exc:
            QMessageBox.critical(self, "应用推荐候选失败", str(exc))

    def _strategy_amplitude(self, strategy: str) -> float | None:
        if strategy not in (FILL_CONVEX, FILL_CONCAVE):
            return 0.0
        unit = self.scene.unit_name if self.scene else "单位"
        value, ok = QInputDialog.getDouble(
            self,
            "设置人工起伏强度",
            f"请输入相对周围拟合平面的最大{'隆起' if strategy == FILL_CONVEX else '凹陷'}"
            f"（{unit}）：",
            0.05 if unit == "米" else 0.1,
            0.0,
            1_000_000.0,
            6,
        )
        return value if ok else None

    def _apply_fill_strategy(self, region_ids: list[int]) -> None:
        if self.terrain_grid is None or not region_ids:
            return
        strategy = str(self.fill_strategy_combo.currentData())
        amplitude = self._strategy_amplitude(strategy)
        if amplitude is None:
            return
        try:
            for region_id in region_ids:
                self.terrain_grid.apply_fill_strategy(
                    region_id, strategy, amplitude
                )
            self.terrain_mesh = None
            self.terrain_mesh_settings = None
            self._invalidate_path_planning()
            self.mesh_summary_label.setText("填充形状已改变，请重新生成地形网格")
            self._refresh_region_list_text()
            self._update_region_summary()
            self._show_scene()
            self.statusBar().showMessage(
                f"已对{len(region_ids)}个区域应用“{FILL_STRATEGY_NAMES[strategy]}”"
            )
        except Exception as exc:
            QMessageBox.critical(self, "应用填充策略失败", str(exc))

    def apply_fill_strategy_to_current(self) -> None:
        region_id = self._current_region_id()
        if region_id is None:
            QMessageBox.information(self, "没有预测区域", "请先生成空白区域预测。")
            return
        self._apply_fill_strategy([region_id])

    def apply_fill_strategy_to_checked(self) -> None:
        region_ids = self._checked_region_ids()
        if not region_ids:
            QMessageBox.information(
                self, "尚未勾选", "请先在批量处理列表中勾选一个或多个区域。"
            )
            return
        self._apply_fill_strategy(region_ids)

    def set_region_decision(self, decision: str) -> None:
        region_id = self._current_region_id()
        if self.terrain_grid is None or region_id is None:
            QMessageBox.information(self, "没有预测区域", "请先生成空白区域预测。")
            return
        self.terrain_grid.set_decision(region_id, decision)
        self.terrain_mesh = None
        self.terrain_mesh_settings = None
        self._invalidate_path_planning()
        self.mesh_summary_label.setText("人工状态已改变，请重新生成地形网格")
        self._refresh_region_list_text()
        self._update_region_summary()
        self._show_scene()
        self.statusBar().showMessage(f"区域{region_id}的人工状态已设为{decision}")

    def generate_terrain_mesh(self) -> None:
        if self.terrain_grid is None:
            QMessageBox.information(
                self, "尚无地形预测", "请先框选地形范围并生成空白区域预测。"
            )
            return
        max_slope, ok = QInputDialog.getDouble(
            self,
            "生成地形网格",
            "允许连接的最大坡度（度）：\n"
            "仅筛除包含真实观测点的过陡连接。\n"
            "完全由已接受预测点组成的表面会连续建网，是否可走由路径规划判断。",
            75.0,
            1.0,
            89.0,
            1,
        )
        if not ok:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("正在连接真实格子和人工接受的预测格子……")
        QApplication.processEvents()
        try:
            self.terrain_mesh = build_terrain_mesh(
                self.terrain_grid, max_slope_degrees=max_slope
            )
            self._invalidate_path_planning()
            self.terrain_mesh_settings = {"max_slope_degrees": max_slope}
            self.terrain_mesh_path = None
            self.show_terrain_mesh_checkbox.setEnabled(True)
            self.show_terrain_mesh_checkbox.setChecked(True)
            self._update_mesh_summary()
            self._show_scene()
            self.statusBar().showMessage(
                f"地形网格已生成：{self.terrain_mesh.n_points:,}个顶点，"
                f"{self.terrain_mesh.n_cells:,}个三角面"
            )
        except Exception as exc:
            self.terrain_mesh = None
            self.terrain_mesh_settings = None
            self._invalidate_path_planning()
            QMessageBox.critical(self, "生成地形网格失败", str(exc))
            self.statusBar().showMessage("地形网格生成失败")
        finally:
            QApplication.restoreOverrideCursor()

    def _draw_terrain_mesh(self) -> None:
        if (
            self.terrain_mesh is None
            or self.terrain_mesh.n_cells == 0
            or not self.show_terrain_mesh_checkbox.isChecked()
        ):
            return
        mesh_kwargs: dict[str, object] = {}
        if "FaceRGB" in self.terrain_mesh.cell_data:
            mesh_kwargs.update(scalars="FaceRGB", preference="cell", rgb=True)
        elif "SourceRGB" in self.terrain_mesh.point_data:
            mesh_kwargs.update(scalars="SourceRGB", preference="point", rgb=True)
        else:
            rgb_name = next(
                (
                    name
                    for name in ("RGB", "RGBA", "rgb", "rgba", "Colors", "colors")
                    if name in self.terrain_mesh.point_data
                    and np.asarray(self.terrain_mesh.point_data[name]).ndim == 2
                    and np.asarray(self.terrain_mesh.point_data[name]).shape[1] in (3, 4)
                ),
                None,
            )
            if rgb_name is not None:
                mesh_kwargs.update(scalars=rgb_name, preference="point", rgb=True)
            else:
                mesh_kwargs.update(color="#55b95f")
        self.plotter.add_mesh(
            self.terrain_mesh,
            name="terrain-mesh",
            show_edges=True,
            edge_color="#27313a",
            line_width=1,
            smooth_shading=False,
            opacity=0.92,
            **mesh_kwargs,
        )

    def _update_mesh_summary(self) -> None:
        if self.terrain_mesh is None:
            self.mesh_summary_label.setText("尚未生成地形网格")
            return
        lines = [
            f"顶点：{self.terrain_mesh.n_points:,}",
            f"三角面/单元：{self.terrain_mesh.n_cells:,}",
        ]
        if "FaceSource" in self.terrain_mesh.cell_data:
            face_source = np.asarray(self.terrain_mesh.cell_data["FaceSource"])
            predicted_faces = int(np.count_nonzero(face_source == 1))
            observed_faces = int(len(face_source) - predicted_faces)
            lines.extend(
                (
                    f"真实三角面：{observed_faces:,}",
                    f"含预测点三角面：{predicted_faces:,}",
                )
            )
        if self.terrain_mesh_settings is not None:
            slope = self.terrain_mesh_settings.get("max_slope_degrees")
            if slope is not None and np.isfinite(slope):
                lines.append(f"最大连接坡度：{slope:.1f}°")
        if "SlopeRejectedFaces" in self.terrain_mesh.field_data:
            rejected = int(
                np.asarray(self.terrain_mesh.field_data["SlopeRejectedFaces"]).ravel()[0]
            )
            lines.append(f"因观测连接过陡而跳过：{rejected:,}面")
        if "AcceptedPredictionFaces" in self.terrain_mesh.field_data:
            retained = int(
                np.asarray(
                    self.terrain_mesh.field_data["AcceptedPredictionFaces"]
                ).ravel()[0]
            )
            lines.append(f"连续保留的已接受预测面：{retained:,}")
        if self.terrain_mesh_path is not None:
            lines.append(f"文件：{self.terrain_mesh_path.name}")
        self.mesh_summary_label.setText("\n".join(lines))

    def import_terrain_mesh(self) -> None:
        start_path = "F:/Desktop/work/rescuemap/exports"
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "导入规划地形网格",
            start_path,
            "地形网格 (*.vtp *.ply);;VTP科研字段网格 (*.vtp);;PLY网格 (*.ply)",
        )
        if not filename:
            return
        path = Path(filename)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage(f"正在导入地形网格：{path.name}")
        QApplication.processEvents()
        try:
            loaded = pv.read(path)
            if not isinstance(loaded, pv.PolyData):
                loaded = loaded.extract_surface().triangulate()
            if loaded.n_points < 3 or loaded.n_cells < 1:
                raise ValueError("文件中没有可用于路径规划的三角地形表面。")
            self.terrain_mesh = loaded
            self._invalidate_path_planning()
            self.terrain_mesh_path = path.resolve()
            slope = None
            if "MaxSlopeDegrees" in loaded.field_data:
                values = np.asarray(loaded.field_data["MaxSlopeDegrees"]).ravel()
                if len(values):
                    slope = float(values[0])
            self.terrain_mesh_settings = (
                {"max_slope_degrees": slope} if slope is not None else None
            )
            self.show_terrain_mesh_checkbox.setEnabled(True)
            self.show_terrain_mesh_checkbox.setChecked(True)
            self._update_mesh_summary()
            self._show_scene()
            self.statusBar().showMessage(f"已导入规划地形网格：{path}")
        except Exception as exc:
            QMessageBox.critical(self, "导入地形网格失败", str(exc))
            self.statusBar().showMessage("地形网格导入失败")
        finally:
            QApplication.restoreOverrideCursor()

    def show_only_terrain_mesh(self) -> None:
        if self.terrain_mesh is None:
            QMessageBox.information(
                self, "尚无地形网格", "请先生成或导入用于路径规划的地形网格。"
            )
            return
        self.show_source_checkbox.setChecked(False)
        self.show_full_map_checkbox.setChecked(False)
        self.show_terrain_mesh_checkbox.setEnabled(True)
        self.show_terrain_mesh_checkbox.setChecked(True)
        self._show_scene()
        self.statusBar().showMessage("当前仅显示规划地形网格，可继续选择路径节点")

    def clear_terrain_mesh(self) -> None:
        self.terrain_mesh = None
        self.terrain_mesh_settings = None
        self.terrain_mesh_path = None
        self.show_terrain_mesh_checkbox.setChecked(True)
        self.show_terrain_mesh_checkbox.setEnabled(False)
        self.mesh_summary_label.setText("尚未生成地形网格")
        self._invalidate_path_planning()
        self._show_scene()
        self.statusBar().showMessage("已清除网格预览；原始点云与预测结果未改变")

    def _invalidate_path_planning(self) -> None:
        self.route_start_point = None
        self.route_end_point = None
        self.route_result = None
        if hasattr(self, "route_summary_label"):
            self.route_summary_label.setText(
                "请先生成或导入地形网格"
                if self.terrain_mesh is None
                else "请选择起点和终点"
            )

    def start_route_start_selection(self) -> None:
        if self.terrain_mesh is None:
            QMessageBox.information(self, "尚无地形网格", "请先生成或导入地形网格。")
            return
        self._start_picking(
            "path-start", 1, "请在地形网格上单击路线起点"
        )

    def start_route_end_selection(self) -> None:
        if self.terrain_mesh is None:
            QMessageBox.information(self, "尚无地形网格", "请先生成或导入地形网格。")
            return
        self._start_picking(
            "path-end", 1, "请在地形网格上单击路线终点"
        )

    def _finish_route_point(self, point: np.ndarray, *, is_start: bool) -> None:
        if self.terrain_mesh is None:
            return
        mesh_points = np.asarray(self.terrain_mesh.points, dtype=float)
        _, vertex_index = cKDTree(mesh_points).query(point, k=1)
        snapped = mesh_points[int(vertex_index)].copy()
        if is_start:
            self.route_start_point = snapped
        else:
            self.route_end_point = snapped
        self.route_result = None
        start_text = "已选择" if self.route_start_point is not None else "未选择"
        end_text = "已选择" if self.route_end_point is not None else "未选择"
        self.route_summary_label.setText(
            f"起点：{start_text}\n终点：{end_text}\n等待计算路径"
        )
        self._draw_path_planning()
        self.plotter.render()
        self.statusBar().showMessage(
            "已设置路线起点" if is_start else "已设置路线终点"
        )

    def calculate_safe_route(self) -> None:
        if self.terrain_mesh is None:
            QMessageBox.information(self, "尚无地形网格", "请先生成或导入地形网格。")
            return
        if self.route_start_point is None or self.route_end_point is None:
            QMessageBox.information(self, "起终点不完整", "请先分别选择起点和终点。")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("正在生成可通行节点并执行Dijkstra路径规划……")
        QApplication.processEvents()
        try:
            self.route_result = plan_safe_path(
                self.terrain_mesh,
                self.route_start_point,
                self.route_end_point,
                max_slope_degrees=self.route_max_slope_spin.value(),
                slope_weight=self.route_slope_weight_spin.value(),
                safety_weight=self.route_safety_weight_spin.value(),
            )
            self.route_start_point = self.route_result.points[0].copy()
            self.route_end_point = self.route_result.points[-1].copy()
            unit = self.scene.unit_name if self.scene is not None else "模型单位"
            result = self.route_result
            self.route_summary_label.setText(
                f"可通行节点：{len(result.traversable_vertices):,}\n"
                f"路径节点：{len(result.vertex_indices):,}\n"
                f"三维长度：{result.length_3d:.3f} {unit}\n"
                f"水平长度：{result.horizontal_length:.3f} {unit}\n"
                f"累计上升/下降：{result.ascent:.3f}/{result.descent:.3f} {unit}\n"
                f"最大路径坡度：{result.max_slope_degrees:.1f}°\n"
                f"平均安全风险：{result.mean_safety_risk:.3f}\n"
                f"综合代价：{result.total_cost:.3f}"
            )
            self._draw_path_planning()
            self.plotter.render()
            self.statusBar().showMessage(
                f"Dijkstra规划完成：路径长度 {result.length_3d:.3f} {unit}"
            )
        except Exception as exc:
            self.route_result = None
            QMessageBox.critical(self, "路径规划失败", str(exc))
            self.statusBar().showMessage("路径规划失败")
        finally:
            QApplication.restoreOverrideCursor()

    def clear_route(self) -> None:
        self._invalidate_path_planning()
        self._show_scene()
        self.statusBar().showMessage("已清除路径规划结果")

    def _draw_path_planning(self) -> None:
        for actor_name in (
            "route-start", "route-end", "route-line", "route-nodes"
        ):
            self.plotter.remove_actor(
                actor_name, reset_camera=False, render=False
            )
        if self.terrain_mesh is None:
            return
        if self.route_start_point is not None:
            self.plotter.add_mesh(
                pv.PolyData(np.asarray([self.route_start_point])),
                name="route-start", style="points", color="#22c55e",
                point_size=18, render_points_as_spheres=True,
                reset_camera=False,
            )
        if self.route_end_point is not None:
            self.plotter.add_mesh(
                pv.PolyData(np.asarray([self.route_end_point])),
                name="route-end", style="points", color="#ef4444",
                point_size=18, render_points_as_spheres=True,
                reset_camera=False,
            )
        if self.route_result is None:
            return
        if (
            self.show_route_nodes_checkbox.isChecked()
            and len(self.route_result.traversable_points)
        ):
            node_points = self.route_result.traversable_points
            if len(node_points) > 100_000:
                step = int(np.ceil(len(node_points) / 100_000))
                node_points = node_points[::step]
            self.plotter.add_mesh(
                pv.PolyData(node_points), name="route-nodes", style="points",
                color="#38bdf8", point_size=3,
                render_points_as_spheres=False, opacity=0.55,
                reset_camera=False,
            )
        if len(self.route_result.points) >= 2:
            route_line = pv.lines_from_points(self.route_result.points)
            diagonal = float(self.terrain_mesh.length) if self.terrain_mesh is not None else 1.0
            radius = max(diagonal * 0.0025, 1e-6)
            self.plotter.add_mesh(
                route_line.tube(radius=radius), name="route-line",
                color="#ff3df2", smooth_shading=True,
                reset_camera=False,
            )

    def export_terrain_mesh(self) -> None:
        if self.terrain_mesh is None:
            QMessageBox.information(self, "尚无网格", "请先点击“生成地形网格”。")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出地形网格（随后在Blender中使用“文件→导入”）",
            "F:/Desktop/work/rescuemap/exports/terrain_mesh.ply",
            "PLY 网格 (*.ply)",
        )
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() != ".ply":
            path = path.with_suffix(".ply")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.terrain_mesh.save(
                path, binary=True, texture="SourceRGB", recompute_normals=False
            )
            data_path = path.with_name(f"{path.stem}_data.vtp")
            self.terrain_mesh.save(data_path, binary=True, recompute_normals=False)
            QMessageBox.information(
                self,
                "导出完成",
                f"Blender查看文件：\n{path}\n\n"
                f"完整科研字段文件：\n{data_path}\n\n"
                "绿色来自真实观测，黄色包含人工接受的预测。\n\n"
                "注意：PLY不是.blend工程，不能用Blender的“文件→打开”。\n"
                "请在Blender中选择：文件 → 导入 → Stanford (.ply)。",
            )
            self.statusBar().showMessage(f"地形网格已导出：{path}")
        except Exception as exc:
            QMessageBox.critical(self, "导出网格失败", str(exc))

    def analyze_density(self) -> None:
        if self.scene is None:
            QMessageBox.information(self, "尚未加载点云", "请先打开 fused.ply。")
            return
        if not self.scene.is_point_cloud:
            QMessageBox.information(
                self, "请使用原始点云",
                "密度分析应对fused.ply进行；Poisson网格密度不代表重建密度。",
            )
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("正在计算每个点的局部密度……")
        QApplication.processEvents()
        try:
            result = analyze_point_density(
                self.scene.original.points,
                green_median_multiplier=self.green_multiplier,
                red_median_multiplier=self.red_multiplier,
            )
            self.density_result = result
            self.scene.original.point_data["DensityScore"] = result.score
            self.scene.original.point_data["DensityClass"] = result.classes
            self.scene.original.point_data["DensityRGB"] = result.colors
            self.color_mode = "density"
            self._show_scene()
            self._show_summary()
            self.statusBar().showMessage(
                f"密度分析完成：红{result.sparse_count:,}，黄{result.medium_count:,}，绿{result.dense_count:,}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "密度分析失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def set_density_thresholds(self) -> None:
        green, ok = QInputDialog.getDouble(
            self, "绿色阈值", "绿色上限＝全局中位近邻距离的倍数：",
            self.green_multiplier, 0.1, 20.0, 2,
        )
        if not ok:
            return
        red, ok = QInputDialog.getDouble(
            self, "红色阈值", "红色下限＝全局中位近邻距离的倍数：",
            self.red_multiplier, green + 0.01, 100.0, 2,
        )
        if not ok:
            return
        self.green_multiplier = green
        self.red_multiplier = red
        self.statusBar().showMessage("密度标准已更新；请重新运行密度分析")

    def show_original_color(self) -> None:
        if self.scene is None:
            return
        self.color_mode = "original"
        self._show_scene()
        self.statusBar().showMessage("当前显示原始模型颜色")

    def _append_density_summary(self, result: DensityResult) -> None:
        self.info_form.addRow("近邻数量：", QLabel(str(result.k_neighbors)))
        self.info_form.addRow("绿色间距上限：", QLabel(f"{result.dense_distance:.6f}"))
        self.info_form.addRow("红色间距下限：", QLabel(f"{result.sparse_distance:.6f}"))
        self.info_form.addRow(
            "红/黄/绿点：",
            QLabel(f"{result.sparse_count:,} / {result.medium_count:,} / {result.dense_count:,}"),
        )

    def show_density_logic(self) -> None:
        QMessageBox.information(
            self, "点云密度标注逻辑",
            "每个点取最近16个邻居的平均距离。默认：\n\n"
            "绿色：不超过全局中位数的1.5倍；\n"
            "黄色：介于1.5倍与3倍之间；\n"
            "红色：超过全局中位数的3倍。\n\n"
            "可在“分析→设置密度宽松程度”中修改。密度不等同于最终可信度。",
        )

    def save_project(self) -> None:
        if self.scene is None:
            QMessageBox.information(self, "没有工程", "请先打开模型。")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存 RescueMap 工程",
            "F:/Desktop/work/rescuemap/projects/project.json",
            "RescueMap 工程 (*.json)",
        )
        if not filename:
            return
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        terrain = None
        if self.terrain_grid is not None and self.terrain_settings is not None:
            terrain = {
                **self.terrain_settings,
                "decisions": self.terrain_grid.decisions,
                "fill_strategies": self.terrain_grid.fill_strategies,
                "fill_parameters": self.terrain_grid.fill_parameters,
                "guide_points": self.terrain_grid.guide_points,
                "manual_fill_bounds": self.terrain_grid.manual_fill_bounds,
            }
        data = {
            "version": 7,
            "model_path": str(self.scene.path),
            "calibration_matrix": self.scene.calibration_matrix.tolist(),
            "unit_name": self.scene.unit_name,
            "terrain_roi": list(self.terrain_roi) if self.terrain_roi else None,
            "terrain": terrain,
            "terrain_mesh": self.terrain_mesh_settings,
            "terrain_mesh_path": (
                str(self.terrain_mesh_path) if self.terrain_mesh_path is not None else None
            ),
            "full_map_mesh_path": (
                str(self.context_mesh.path) if self.context_mesh is not None else None
            ),
            "density": {
                "green_multiplier": self.green_multiplier,
                "red_multiplier": self.red_multiplier,
            },
            "path_planning": {
                "start_point": (
                    self.route_start_point.tolist()
                    if self.route_start_point is not None else None
                ),
                "end_point": (
                    self.route_end_point.tolist()
                    if self.route_end_point is not None else None
                ),
                "max_slope_degrees": self.route_max_slope_spin.value(),
                "slope_weight": self.route_slope_weight_spin.value(),
                "safety_weight": self.route_safety_weight_spin.value(),
            },
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.statusBar().showMessage(f"工程已保存：{path}")

    def open_project(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "打开 RescueMap 工程", "F:/Desktop/work/rescuemap/projects",
            "RescueMap 工程 (*.json)",
        )
        if not filename:
            return
        try:
            data = json.loads(Path(filename).read_text(encoding="utf-8"))
            self.scene = SceneModel.load(Path(data["model_path"]))
            self.scene.apply_transform(np.asarray(data["calibration_matrix"], dtype=float))
            self.scene.unit_name = data.get("unit_name", "COLMAP单位")
            self.scene_point_tree = None
            self.context_mesh = None
            self.show_source_checkbox.setChecked(True)
            self.show_full_map_checkbox.setChecked(False)
            self.show_full_map_checkbox.setEnabled(False)
            self.full_map_summary_label.setText("尚未加载完整网格")
            density = data.get("density", {})
            self.green_multiplier = float(density.get("green_multiplier", 1.5))
            self.red_multiplier = float(density.get("red_multiplier", 3.0))
            roi = data.get("terrain_roi")
            self.terrain_roi = tuple(map(float, roi)) if roi else None
            self.density_result = None
            self.color_mode = "original"
            self.terrain_grid = None
            self.terrain_settings = None
            self.candidate_results.clear()
            self.terrain_mesh = None
            self.terrain_mesh_settings = None
            self.terrain_mesh_path = None
            self._invalidate_path_planning()
            self.show_terrain_mesh_checkbox.setChecked(True)
            self.show_terrain_mesh_checkbox.setEnabled(False)
            terrain = data.get("terrain")
            if terrain and self.terrain_roi:
                self.terrain_settings = {
                    "cell_size": float(terrain["cell_size"]),
                    "min_points": int(terrain["min_points"]),
                    "max_prediction_distance_cells": float(terrain["max_prediction_distance_cells"]),
                    "algorithm": str(terrain.get("algorithm", "linear")),
                }
                algorithm_index = self.algorithm_combo.findData(self.terrain_settings["algorithm"])
                if algorithm_index >= 0:
                    self.algorithm_combo.setCurrentIndex(algorithm_index)
                self.terrain_grid = build_terrain_grid(
                    self.scene.original.points,
                    bounds_xy=self.terrain_roi,
                    **self.terrain_settings,
                )
                for region_id, rectangles in terrain.get(
                    "manual_fill_bounds", {}
                ).items():
                    region_id = int(region_id)
                    if region_id <= self.terrain_grid.region_count:
                        for bounds in rectangles:
                            self.terrain_grid.add_manual_fill_rectangle(
                                region_id, tuple(map(float, bounds))
                            )
                for region_id, guide_points in terrain.get("guide_points", {}).items():
                    region_id = int(region_id)
                    if region_id <= self.terrain_grid.region_count:
                        self.terrain_grid.set_guide_points(
                            region_id, np.asarray(guide_points, dtype=float)
                        )
                fill_parameters = terrain.get("fill_parameters", {})
                for region_id, strategy in terrain.get("fill_strategies", {}).items():
                    region_id = int(region_id)
                    if region_id <= self.terrain_grid.region_count:
                        amplitude = float(
                            fill_parameters.get(str(region_id),
                                                fill_parameters.get(region_id, 0.0))
                        )
                        self.terrain_grid.apply_fill_strategy(
                            region_id, str(strategy), amplitude
                        )
                for region_id, decision in terrain.get("decisions", {}).items():
                    region_id = int(region_id)
                    if region_id <= self.terrain_grid.region_count:
                        self.terrain_grid.set_decision(region_id, decision)
                mesh_settings = data.get("terrain_mesh")
                if mesh_settings:
                    self.terrain_mesh_settings = {
                        "max_slope_degrees": float(mesh_settings["max_slope_degrees"])
                    }
                    self.terrain_mesh = build_terrain_mesh(
                        self.terrain_grid, **self.terrain_mesh_settings
                    )
                    self.show_terrain_mesh_checkbox.setEnabled(True)
                    self.show_terrain_mesh_checkbox.setChecked(True)
            terrain_mesh_path = data.get("terrain_mesh_path")
            if terrain_mesh_path:
                imported = pv.read(Path(terrain_mesh_path))
                if not isinstance(imported, pv.PolyData):
                    imported = imported.extract_surface().triangulate()
                if imported.n_points < 3 or imported.n_cells < 1:
                    raise ValueError("工程中的地形网格不包含有效三角面。")
                self.terrain_mesh = imported
                self.terrain_mesh_path = Path(terrain_mesh_path).resolve()
                self.show_terrain_mesh_checkbox.setEnabled(True)
                self.show_terrain_mesh_checkbox.setChecked(True)
            full_map_path = data.get("full_map_mesh_path")
            if full_map_path:
                context_mesh = SceneModel.load(Path(full_map_path))
                context_mesh.apply_transform(self.scene.calibration_matrix)
                context_mesh.unit_name = self.scene.unit_name
                self.context_mesh = context_mesh
                self.show_full_map_checkbox.setEnabled(True)
                self.show_full_map_checkbox.setChecked(True)
                self.show_source_checkbox.setChecked(False)
                simplified = (
                    "（预览已简化，导出仍用完整模型）"
                    if context_mesh.simplified else ""
                )
                self.full_map_summary_label.setText(
                    f"文件：{context_mesh.path.name}\n"
                    f"完整顶点：{context_mesh.original.n_points:,}\n"
                    f"完整三角面：{context_mesh.original.n_faces:,}\n"
                    f"坐标单位：{context_mesh.unit_name}\n{simplified}"
                )
            self._invalidate_path_planning()
            path_planning = data.get("path_planning", {})
            self.route_max_slope_spin.setValue(
                float(path_planning.get("max_slope_degrees", 35.0))
            )
            self.route_slope_weight_spin.setValue(
                float(path_planning.get("slope_weight", 3.0))
            )
            self.route_safety_weight_spin.setValue(
                float(path_planning.get("safety_weight", 2.0))
            )
            start_point = path_planning.get("start_point")
            end_point = path_planning.get("end_point")
            if self.terrain_mesh is not None and start_point is not None:
                self.route_start_point = np.asarray(start_point, dtype=float)
            if self.terrain_mesh is not None and end_point is not None:
                self.route_end_point = np.asarray(end_point, dtype=float)
            if self.route_start_point is not None or self.route_end_point is not None:
                self.route_summary_label.setText(
                    "已恢复起点/终点，请点击“计算安全路径（Dijkstra）”"
                )
            self._populate_regions()
            self._update_mesh_summary()
            self._show_scene()
            self._show_summary()
            self.statusBar().showMessage(f"工程已打开：{filename}")
        except Exception as exc:
            QMessageBox.critical(self, "打开工程失败", str(exc))

    def reset_camera(self) -> None:
        self.plotter.reset_camera()
        self.plotter.render()

    def view_top(self) -> None:
        self.plotter.view_xy()
        self.plotter.reset_camera()
        self.plotter.render()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.colmap_runner.is_running:
            QMessageBox.information(
                self,
                "任务仍在运行",
                "照片处理或重建仍在运行。请等待当前步骤完成，或先点击取消。",
            )
            event.ignore()
            return
        self._hide_processing_wait_dialog()
        self.plotter.close()
        super().closeEvent(event)
