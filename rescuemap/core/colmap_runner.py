from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QProcess, QThread, QTimer, Signal, Slot

from rescuemap.core.background_completion import complete_dynamic_backgrounds
from rescuemap.core.clarity_enhancement import (
    clarify_selected_regions,
    detect_blurry_regions,
)
from rescuemap.core.dynamic_mask import generate_geometry_masks
from rescuemap.core.image_preprocess import enhance_low_light_dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def bundled_colmap_path() -> Path | None:
    """Return the COLMAP engine shipped beside the application, if present."""
    if getattr(sys, "frozen", False):
        application_directory = Path(sys.executable).resolve().parent
    else:
        application_directory = Path(__file__).resolve().parents[2]
    candidates = (
        application_directory / "engines" / "colmap" / "bin" / "colmap.exe",
        application_directory / "engines" / "colmap" / "COLMAP.bat",
    )
    return next((path for path in candidates if path.is_file()), None)


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    workspace: Path
    point_cloud: Path
    poisson_mesh: Path


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    workspace: Path
    static_images: Path
    masks: Path


@dataclass(frozen=True, slots=True)
class _Stage:
    title: str
    command: str
    arguments: tuple[str, ...]
    kind: str = "process"


class _FunctionWorker(QObject):
    finished = Signal()
    failed = Signal(str)

    def __init__(self, function) -> None:
        super().__init__()
        self.function = function

    @Slot()
    def run(self) -> None:
        try:
            self.function()
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit()


def resolve_colmap_executable(selected_path: Path) -> Path:
    """Resolve a user-selected COLMAP file or installation directory."""
    selected_path = selected_path.expanduser().resolve()
    candidates: list[Path]
    if selected_path.is_dir():
        candidates = [
            selected_path / "bin" / "colmap.exe",
            selected_path / "colmap.exe",
            selected_path / "COLMAP.exe",
        ]
    elif selected_path.suffix.lower() in {".bat", ".cmd"}:
        candidates = [
            selected_path.parent / "bin" / "colmap.exe",
            selected_path.parent / "colmap.exe",
        ]
    else:
        candidates = [selected_path]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "没有找到 colmap.exe。请选择 COLMAP 安装目录，或其中的 bin\\colmap.exe。"
    )


def create_run_workspace(project_directory: Path) -> Path:
    project_directory = project_directory.expanduser().resolve()
    project_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = project_directory / f"reconstruction_{stamp}"
    suffix = 2
    while candidate.exists():
        candidate = project_directory / f"reconstruction_{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    (candidate / "sparse").mkdir()
    (candidate / "dense").mkdir()
    return candidate


def _quality_parameters(quality: str) -> tuple[str, str]:
    values = {
        "fast": ("4096", "1600"),
        "standard": ("8192", "2400"),
        "high": ("16384", "3200"),
    }
    if quality not in values:
        raise ValueError(f"未知重建质量：{quality}")
    return values[quality]


def build_colmap_stages(
    executable: Path,
    image_directory: Path,
    workspace: Path,
    quality: str,
    dynamic_suppression: bool = False,
    low_light_enhancement: bool = False,
    preprocess_only: bool = False,
) -> list[_Stage]:
    max_features, max_image_size = _quality_parameters(quality)
    database = workspace / "database.db"
    sparse = workspace / "sparse"
    sparse_model = sparse / "0"
    dense = workspace / "dense"
    fused = dense / "fused.ply"
    mesh = dense / "meshed-poisson.ply"
    exe = str(executable)

    processing_image_directory = (
        workspace / "enhanced_images" if low_light_enhancement else image_directory
    )
    enhancement_stages = (
        [
            _Stage("统一增强夜间图像", "", (), "enhance_images"),
            _Stage("人工检查夜间增强结果", "", (), "review_enhancement"),
        ]
        if low_light_enhancement else []
    )
    formal_image_directory = processing_image_directory
    formal_mask_directory: Path | None = None
    preview_stages: list[_Stage] = []
    if dynamic_suppression:
        preview_root = workspace / "dynamic_preview"
        preview_database = preview_root / "database.db"
        preview_sparse = preview_root / "sparse"
        preview_root.mkdir(parents=True, exist_ok=True)
        preview_sparse.mkdir(parents=True, exist_ok=True)
        formal_image_directory = workspace / "static_images"
        formal_mask_directory = workspace / "dynamic_masks"
        preview_stages = [
            _Stage(
                "快速提取低分辨率特征",
                exe,
                (
                    "feature_extractor",
                    "--database_path", str(preview_database),
                    "--image_path", str(processing_image_directory),
                    "--ImageReader.single_camera", "1",
                    "--SiftExtraction.max_num_features", "2048",
                    "--FeatureExtraction.max_image_size", "1200",
                ),
            ),
            _Stage(
                "快速匹配相邻图像",
                exe,
                (
                    "sequential_matcher",
                    "--database_path", str(preview_database),
                    "--SequentialMatching.overlap", "2",
                ),
            ),
            _Stage(
                "估计预览相机位姿",
                exe,
                (
                    "mapper",
                    "--database_path", str(preview_database),
                    "--image_path", str(processing_image_directory),
                    "--output_path", str(preview_sparse),
                ),
                "mapper_preview",
            ),
            _Stage("分析跨帧几何并生成动态掩膜", "", (), "generate_masks"),
            _Stage("人工复核自动动态框", "", (), "review_initial"),
            _Stage("多视角恢复静态背景", "", (), "complete_backgrounds"),
            _Stage("人工复核恢复后的静态图片", "", (), "review_completed"),
            _Stage("生成最终静态图片数据集", "", (), "complete_final"),
            _Stage("自动检测最终数据集模糊区域", "", (), "detect_blur"),
            _Stage("人工复核待清晰化区域", "", (), "review_clarity"),
            _Stage("自适应清晰化所选区域", "", (), "clarify_final"),
            _Stage("确认最终静态图片数据集", "", (), "review_final_static"),
        ]

    feature_arguments = [
        "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(formal_image_directory),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_num_features", max_features,
    ]
    if formal_mask_directory is not None:
        feature_arguments.extend(("--ImageReader.mask_path", str(formal_mask_directory)))

    formal_stages = [
        _Stage(
            "正式提取静态图像特征" if dynamic_suppression else "提取图像特征",
            exe,
            tuple(feature_arguments),
        ),
        _Stage(
            "匹配图像特征",
            exe,
            ("exhaustive_matcher", "--database_path", str(database)),
        ),
        _Stage(
            "生成稀疏模型",
            exe,
            (
                "mapper",
                "--database_path", str(database),
                "--image_path", str(formal_image_directory),
                "--output_path", str(sparse),
            ),
            "mapper_final",
        ),
        _Stage(
            "校正图像",
            exe,
            (
                "image_undistorter",
                "--image_path", str(formal_image_directory),
                "--input_path", str(sparse_model),
                "--output_path", str(dense),
                "--output_type", "COLMAP",
                "--max_image_size", max_image_size,
            ),
        ),
        _Stage(
            "计算稠密深度",
            exe,
            (
                "patch_match_stereo",
                "--workspace_path", str(dense),
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.max_image_size", max_image_size,
                "--PatchMatchStereo.geom_consistency", "true",
            ),
        ),
        _Stage(
            "融合稠密点云",
            exe,
            (
                "stereo_fusion",
                "--workspace_path", str(dense),
                "--workspace_format", "COLMAP",
                "--input_type", "geometric",
                "--StereoFusion.max_image_size", max_image_size,
                "--output_path", str(fused),
            ),
        ),
        _Stage(
            "生成 Poisson 网格",
            exe,
            ("poisson_mesher", "--input_path", str(fused), "--output_path", str(mesh)),
        ),
    ]
    preprocessing_stages = enhancement_stages + preview_stages
    return preprocessing_stages if preprocess_only else preprocessing_stages + formal_stages


class ColmapRunner(QObject):
    """Run a complete COLMAP reconstruction without blocking the Qt UI."""

    log_received = Signal(str)
    stage_changed = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)
    running_changed = Signal(bool)
    review_requested = Signal(str, str, str)
    enhancement_review_requested = Signal(str, str)
    final_review_requested = Signal(str, str)
    preprocessing_completed = Signal(object)
    photo_progress = Signal(int, int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self._stages: list[_Stage] = []
        self._stage_index = -1
        self._workspace: Path | None = None
        self._image_directory: Path | None = None
        self._processing_image_directory: Path | None = None
        self._dynamic_strength = "conservative"
        self._cancelled = False
        self._active = False
        self._preprocess_only = False
        self._resume_from_static = False
        self._background_thread: QThread | None = None
        self._background_worker: _FunctionWorker | None = None
        self._background_failure_prefix = "照片处理失败"

    @property
    def is_running(self) -> bool:
        return self._active

    def start(
        self,
        colmap_path: Path,
        image_directory: Path,
        project_directory: Path,
        quality: str = "standard",
        dynamic_suppression: bool = False,
        dynamic_strength: str = "conservative",
        low_light_enhancement: bool = False,
        preprocess_only: bool = False,
    ) -> Path:
        if self.is_running:
            raise RuntimeError("已有重建任务正在运行。")
        executable = resolve_colmap_executable(colmap_path)
        image_directory = image_directory.expanduser().resolve()
        if not image_directory.is_dir():
            raise FileNotFoundError(f"找不到照片目录：{image_directory}")
        image_count = sum(
            1
            for path in image_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if image_count < 2:
            raise ValueError("照片目录中至少需要两张受支持的图片。")
        if preprocess_only and not dynamic_suppression:
            raise ValueError("照片预处理需要开启动态区域检测。")

        workspace = create_run_workspace(project_directory)
        self._workspace = workspace
        self._image_directory = image_directory
        self._processing_image_directory = (
            workspace / "enhanced_images" if low_light_enhancement else image_directory
        )
        self._dynamic_strength = dynamic_strength
        self._stages = build_colmap_stages(
            executable,
            image_directory,
            workspace,
            quality,
            dynamic_suppression,
            low_light_enhancement,
            preprocess_only,
        )
        self._stage_index = -1
        self._cancelled = False
        self._active = True
        self._preprocess_only = preprocess_only
        self._resume_from_static = False
        self.log_received.emit(
            f"COLMAP：{executable}\n照片：{image_directory}\n"
            f"照片数量：{image_count}\n本次工作区：{workspace}\n"
            f"动态干扰自动抑制：{'开启' if dynamic_suppression else '关闭'}\n"
            + (
                f"动态抑制强度：{dynamic_strength}\n"
                if dynamic_suppression
                else ""
            )
            + f"夜间统一增强：{'开启' if low_light_enhancement else '关闭'}\n"
            + f"任务类型：{'仅生成最终静态图片数据集' if preprocess_only else '完整三维重建'}\n"
        )
        self.running_changed.emit(True)
        self._start_next_stage()
        return workspace

    def resume_preprocessing(
        self,
        colmap_path: Path,
        image_directory: Path,
        workspace: Path,
        start_kind: str,
    ) -> Path:
        """Resume an existing preprocessing workspace at a chosen stage."""
        if self.is_running:
            raise RuntimeError("已有重建任务正在运行。")
        executable = resolve_colmap_executable(colmap_path)
        image_directory = image_directory.expanduser().resolve()
        workspace = workspace.expanduser().resolve()
        if not image_directory.is_dir():
            raise FileNotFoundError(f"找不到原始照片目录：{image_directory}")
        if not workspace.is_dir():
            raise FileNotFoundError(f"找不到已有预处理工程：{workspace}")
        processing_directory = (
            workspace / "enhanced_images"
            if (workspace / "enhanced_images").is_dir()
            else image_directory
        )
        stages = build_colmap_stages(
            executable,
            image_directory,
            workspace,
            "fast",
            dynamic_suppression=True,
            low_light_enhancement=False,
            preprocess_only=True,
        )
        start_index = next(
            (index for index, stage in enumerate(stages) if stage.kind == start_kind),
            None,
        )
        if start_index is None:
            raise ValueError(f"未知的继续步骤：{start_kind}")
        if start_kind != "generate_masks" and not (workspace / "dynamic_masks").is_dir():
            raise FileNotFoundError("该工程尚未生成 dynamic_masks，不能从所选步骤继续。")
        if start_kind in {"review_completed", "complete_final", "detect_blur",
                           "review_clarity", "clarify_final", "review_final_static"} \
                and not (workspace / "static_images").is_dir():
            raise FileNotFoundError("该工程尚未生成 static_images，不能从所选步骤继续。")
        if start_kind in {"generate_masks", "complete_backgrounds", "complete_final"} \
                and not (workspace / "dynamic_preview" / "database.db").is_file():
            raise FileNotFoundError("该工程缺少快速重建数据库，无法继续自动处理。")

        # The working dynamic mask is reduced as pixels are successfully
        # stitched.  Restore the immutable, human-reviewed snapshot when the
        # user explicitly goes back to the first review or stitching stage.
        reviewed_masks = workspace / "reviewed_dynamic_masks"
        if start_kind in {"review_initial", "complete_backgrounds"}:
            if reviewed_masks.is_dir():
                shutil.copytree(
                    reviewed_masks, workspace / "dynamic_masks", dirs_exist_ok=True
                )
            elif (workspace / "review_initial_annotations.json").is_file():
                self._restore_annotation_boxes(
                    processing_directory,
                    workspace / "review_initial_annotations.json",
                    workspace / "dynamic_masks",
                )
                shutil.copytree(
                    workspace / "dynamic_masks", reviewed_masks, dirs_exist_ok=True
                )
        if start_kind in {"review_completed", "complete_final"} \
                and not (workspace / "refinement_masks").is_dir() \
                and (workspace / "review_completed_annotations.json").is_file():
            self._restore_annotation_boxes(
                workspace / "static_images",
                workspace / "review_completed_annotations.json",
                workspace / "refinement_masks",
            )
        if start_kind in {"review_clarity", "clarify_final"} \
                and not (workspace / "clarity_masks").is_dir() \
                and (workspace / "review_clarity_annotations.json").is_file():
            self._restore_annotation_boxes(
                workspace / "static_images",
                workspace / "review_clarity_annotations.json",
                workspace / "clarity_masks",
            )

        self._workspace = workspace
        self._image_directory = image_directory
        self._processing_image_directory = processing_directory
        self._dynamic_strength = "standard"
        self._stages = stages
        self._stage_index = start_index - 1
        self._cancelled = False
        self._active = True
        self._preprocess_only = True
        self._resume_from_static = (
            start_kind in {"review_completed", "complete_final", "detect_blur",
                           "review_clarity", "clarify_final", "review_final_static"}
            and (workspace / "static_images").is_dir()
        )
        self.log_received.emit(
            f"继续已有照片预处理工程：{workspace}\n"
            f"原始照片：{image_directory}\n"
            f"当前处理照片：{processing_directory}\n"
            f"从步骤“{stages[start_index].title}”继续。\n"
        )
        self.running_changed.emit(True)
        self._start_next_stage()
        return workspace

    def cancel(self) -> None:
        if not self.is_running:
            return
        self._cancelled = True
        self.log_received.emit("\n正在取消重建任务……\n")
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
        elif self._background_thread is not None and self._background_thread.isRunning():
            self.log_received.emit("当前图片步骤会安全结束后停止，请稍候。\n")
        else:
            self._active = False
            self.running_changed.emit(False)
            self.failed.emit("重建任务已取消，已生成的中间文件仍保留在工作区中。")

    def resume_after_review(self, accepted: bool) -> None:
        """Continue a pipeline paused for a modal human review."""
        if not self._active:
            return
        if not accepted:
            self.cancel()
            return
        if (
            self._workspace is not None
            and 0 <= self._stage_index < len(self._stages)
            and self._stages[self._stage_index].kind == "review_initial"
        ):
            source = self._workspace / "dynamic_masks"
            target = self._workspace / "reviewed_dynamic_masks"
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
                self.log_received.emit(
                    "第一轮人工动态框已独立保存；后续拼接不会删除这份标注。\n"
                )
        self.log_received.emit("人工复核结果已保存，继续处理。\n")
        QTimer.singleShot(0, self._start_next_stage)

    def resume_after_enhancement_review(self, choice: str) -> None:
        if not self._active:
            return
        if choice == "cancel":
            self.cancel()
            return
        if choice == "original":
            assert self._image_directory is not None
            self._switch_processing_images(self._image_directory)
            self.log_received.emit("已放弃夜间增强，本次重建继续使用原始照片。\n")
        else:
            self.log_received.emit("已确认夜间增强结果。\n")
        QTimer.singleShot(0, self._start_next_stage)

    def resume_after_final_review(self, choice: str) -> None:
        if not self._active:
            return
        if choice == "cancel":
            self.cancel()
            return
        if choice == "revise":
            review_index = next(
                index for index, stage in enumerate(self._stages)
                if stage.kind == "review_completed"
            )
            self._stage_index = review_index - 1
            self.log_received.emit("返回第二次动态框复核，可继续修正后重新生成最终数据集。\n")
        else:
            assert self._workspace is not None
            self._accept_refined_pixels(self._workspace)
            self.log_received.emit(
                "最终静态图片数据集已确认，保存并返回主界面。\n"
                if self._preprocess_only
                else "最终静态图片数据集已确认，开始正式三维重建。\n"
            )
        QTimer.singleShot(0, self._start_next_stage)

    def resume_after_clarity_review(self, choice: str) -> None:
        """Apply reviewed clarity boxes, skip them, or cancel preprocessing."""
        if not self._active:
            return
        if choice == "cancel":
            self.cancel()
            return
        if choice == "skip":
            clarify_index = next(
                index for index, stage in enumerate(self._stages)
                if stage.kind == "clarify_final"
            )
            self._stage_index = clarify_index
            self.log_received.emit("已跳过最终数据集清晰化，直接进入最终确认。\n")
        else:
            self.log_received.emit("清晰化选区已保存，开始自适应处理。\n")
        QTimer.singleShot(0, self._start_next_stage)

    def _switch_processing_images(self, directory: Path) -> None:
        assert self._processing_image_directory is not None
        old = str(self._processing_image_directory)
        new = str(directory)
        updated = []
        for stage in self._stages:
            arguments = tuple(new if argument == old else argument for argument in stage.arguments)
            updated.append(_Stage(stage.title, stage.command, arguments, stage.kind))
        self._stages = updated
        self._processing_image_directory = directory

    def _start_next_stage(self) -> None:
        self._stage_index += 1
        if self._stage_index >= len(self._stages):
            self._finish_successfully()
            return
        stage = self._stages[self._stage_index]
        self.stage_changed.emit(
            self._stage_index + 1, len(self._stages), stage.title
        )
        self.log_received.emit(
            f"\n===== {self._stage_index + 1}/{len(self._stages)} "
            f"{stage.title} =====\n"
        )
        if self._workspace is not None:
            self.process.setWorkingDirectory(str(self._workspace))
        if stage.kind == "generate_masks":
            self._start_python_stage(
                self._generate_dynamic_masks, "自动生成动态区域掩膜失败"
            )
            return
        if stage.kind == "enhance_images":
            self._start_python_stage(self._enhance_images, "夜间图像增强失败")
            return
        if stage.kind == "review_enhancement":
            assert self._workspace is not None
            assert self._image_directory is not None
            self.enhancement_review_requested.emit(
                str(self._image_directory), str(self._workspace / "enhanced_images")
            )
            return
        if stage.kind in {"review_initial", "review_completed", "review_clarity"}:
            assert self._workspace is not None
            assert self._processing_image_directory is not None
            image_directory = (
                self._processing_image_directory
                if stage.kind == "review_initial" else self._workspace / "static_images"
            )
            mask_directory = self._workspace / "dynamic_masks"
            if stage.kind == "review_completed":
                mask_directory = self._workspace / "refinement_masks"
                self._prepare_refinement_masks(image_directory, mask_directory)
            elif stage.kind == "review_clarity":
                mask_directory = self._workspace / "clarity_masks"
            self.review_requested.emit(
                stage.kind, str(image_directory), str(mask_directory)
            )
            return
        if stage.kind == "detect_blur":
            self._start_python_stage(
                self._detect_final_blur, "最终数据集模糊区域检测失败"
            )
            return
        if stage.kind == "clarify_final":
            self._start_python_stage(
                self._clarify_final_dataset, "最终数据集清晰化失败"
            )
            return
        if stage.kind == "review_final_static":
            assert self._workspace is not None
            assert self._processing_image_directory is not None
            self.final_review_requested.emit(
                str(self._processing_image_directory),
                str(self._workspace / "static_images"),
            )
            return
        if stage.kind in {"complete_backgrounds", "complete_final"}:
            final = stage.kind == "complete_final"
            self._start_python_stage(
                lambda: self._complete_dynamic_backgrounds(final=final),
                "多视角背景恢复失败",
            )
            return
        self.process.start(stage.command, list(stage.arguments))

    def _start_python_stage(self, function, failure_prefix: str) -> None:
        thread = QThread(self)
        worker = _FunctionWorker(function)
        self._background_failure_prefix = failure_prefix
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(self._python_stage_succeeded)
        worker.failed.connect(self._python_stage_failed)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_background_task)
        self._background_thread = thread
        self._background_worker = worker
        thread.start()

    def _python_stage_succeeded(self) -> None:
        if not self._active:
            return
        if self._cancelled:
            self._active = False
            self.running_changed.emit(False)
            self.failed.emit("任务已取消，已生成的中间文件仍保留在工作区中。")
            return
        QTimer.singleShot(0, self._start_next_stage)

    def _python_stage_failed(self, message: str) -> None:
        if not self._active:
            return
        self._active = False
        self.running_changed.emit(False)
        if self._cancelled:
            self.failed.emit("任务已取消，已生成的中间文件仍保留在工作区中。")
        else:
            self.failed.emit(f"{self._background_failure_prefix}：{message}")

    def _clear_background_task(self) -> None:
        self._background_thread = None
        self._background_worker = None

    def _read_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput())
        if data:
            self.log_received.emit(data.decode("utf-8", errors="replace"))

    def _process_finished(self, exit_code: int, _exit_status) -> None:
        self._read_output()
        if self._cancelled:
            self._active = False
            self.running_changed.emit(False)
            self.failed.emit("重建任务已取消，已生成的中间文件仍保留在工作区中。")
            return
        if exit_code != 0:
            stage = self._stages[self._stage_index]
            self._active = False
            self.running_changed.emit(False)
            self.failed.emit(
                f"“{stage.title}”执行失败（退出代码 {exit_code}）。请查看重建日志。"
            )
            return
        stage = self._stages[self._stage_index]
        if stage.kind in {"mapper_preview", "mapper_final"}:
            try:
                self._select_sparse_model(stage.kind)
            except Exception as exc:
                self._active = False
                self.running_changed.emit(False)
                self.failed.emit(str(exc))
                return
        self._start_next_stage()

    def _select_sparse_model(self, mapper_kind: str) -> None:
        """Use the largest registered sparse model when mapper creates several."""
        assert self._workspace is not None
        sparse_root = (
            self._workspace / "dynamic_preview" / "sparse"
            if mapper_kind == "mapper_preview"
            else self._workspace / "sparse"
        )
        models = [
            path
            for path in sparse_root.iterdir()
            if path.is_dir()
            and (path / "cameras.bin").is_file()
            and (path / "images.bin").is_file()
            and (path / "points3D.bin").is_file()
        ]
        if not models:
            if mapper_kind == "mapper_preview":
                self.log_received.emit(
                    "警告：快速阶段没有形成完整稀疏模型；将使用已经完成的两视图"
                    "几何验证继续生成保守掩膜。\n"
                )
                return
            raise RuntimeError(
                "稀疏重建没有生成有效模型。通常是照片重叠率不足、纹理太少，"
                "或相邻照片视角变化过大。"
            )
        selected = max(models, key=lambda path: (path / "images.bin").stat().st_size)
        if mapper_kind == "mapper_final":
            undistort_index = next(
                index
                for index, stage in enumerate(self._stages)
                if stage.arguments and stage.arguments[0] == "image_undistorter"
            )
            stage = self._stages[undistort_index]
            arguments = list(stage.arguments)
            input_index = arguments.index("--input_path") + 1
            arguments[input_index] = str(selected)
            self._stages[undistort_index] = _Stage(
                stage.title, stage.command, tuple(arguments), stage.kind
            )
        label = "预览" if mapper_kind == "mapper_preview" else "正式"
        self.log_received.emit(f"选用{label}稀疏模型：{selected.name}\n")

    def _generate_dynamic_masks(self) -> None:
        assert self._workspace is not None
        assert self._processing_image_directory is not None
        summary = generate_geometry_masks(
            self._workspace / "dynamic_preview" / "database.db",
            self._processing_image_directory,
            self._workspace / "dynamic_masks",
            self._workspace / "static_images",
            strength=self._dynamic_strength,
            preview_directory=self._workspace / "dynamic_mask_previews",
            report_path=self._workspace / "dynamic_report.json",
        )
        self.log_received.emit(
            "动态区域自动检测完成："
            f"{summary.masked_image_count}/{summary.image_count} 张照片包含候选动态区域，"
            f"候选像素占比 {summary.masked_ratio * 100:.2f}%。\n"
            "即将打开人工复核窗口：左键拖框补充，右键单击红框删除。\n"
        )

    def _enhance_images(self) -> None:
        assert self._workspace is not None
        assert self._image_directory is not None
        summary = enhance_low_light_dataset(
            self._image_directory, self._workspace / "enhanced_images"
        )
        self.log_received.emit(
            f"已增强 {summary.image_count} 张夜间照片；全部照片共用亮度区间 "
            f"{summary.low_percentile:.1f}～{summary.high_percentile:.1f}，原图未修改。\n"
        )

    def _detect_final_blur(self) -> None:
        assert self._workspace is not None
        summary = detect_blurry_regions(
            self._workspace / "static_images",
            self._workspace / "clarity_masks",
            self._workspace / "clarity_detection_report.json",
            progress_callback=lambda current, total: self.photo_progress.emit(
                current, total, "自动检测相对周围明显模糊的区域"
            ),
            should_cancel=lambda: self._cancelled,
        )
        self.log_received.emit(
            f"清晰度检测完成：{summary.selected_images}/{summary.image_count} 张照片"
            f"包含候选模糊区域，共 {summary.selected_pixels} 个候选像素。\n"
        )

    def _clarify_final_dataset(self) -> None:
        assert self._workspace is not None
        summary = clarify_selected_regions(
            self._workspace / "static_images",
            self._workspace / "clarity_masks",
            self._workspace / "static_images_before_clarity",
            self._workspace / "clarity_enhancement_report.json",
            progress_callback=lambda current, total: self.photo_progress.emit(
                current, total, "强/弱纹理自适应清晰化"
            ),
            should_cancel=lambda: self._cancelled,
        )
        self.log_received.emit(
            f"最终数据集清晰化完成：处理 {summary.selected_images} 张照片；"
            f"强纹理延伸 {summary.strong_texture_pixels} 像素，"
            f"弱纹理主色稳定 {summary.weak_texture_pixels} 像素。\n"
        )

    @staticmethod
    def _restore_annotation_boxes(
        image_directory: Path, annotation_path: Path, mask_directory: Path
    ) -> None:
        """Recreate editable masks from saved box JSON in an older project."""
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        mask_directory.mkdir(parents=True, exist_ok=True)
        for record in payload.get("images", []):
            relative = Path(str(record.get("image", "")))
            image_path = image_directory / relative
            if not image_path.is_file():
                continue
            with Image.open(image_path) as opened:
                static_mask = np.full(
                    (opened.height, opened.width), 255, dtype=np.uint8
                )
            for box in record.get("boxes", []):
                if not isinstance(box, list) or len(box) != 4:
                    continue
                x0, y0, x1, y1 = (int(value) for value in box)
                x0 = max(0, min(static_mask.shape[1] - 1, x0))
                x1 = max(0, min(static_mask.shape[1] - 1, x1))
                y0 = max(0, min(static_mask.shape[0] - 1, y0))
                y1 = max(0, min(static_mask.shape[0] - 1, y1))
                if x1 >= x0 and y1 >= y0:
                    static_mask[y0:y1 + 1, x0:x1 + 1] = 0
            target = mask_directory / (str(relative) + ".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(static_mask).save(target)

    @staticmethod
    def _prepare_refinement_masks(
        image_directory: Path, mask_directory: Path
    ) -> None:
        """Carry unresolved first-review boxes into persistent refinement masks."""
        mask_directory.mkdir(parents=True, exist_ok=True)
        unresolved_root = mask_directory.parent / "dynamic_masks"
        for image_path in image_directory.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = image_path.relative_to(image_directory)
            target = mask_directory / (str(relative) + ".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            unresolved_path = unresolved_root / (str(relative) + ".png")
            if target.is_file():
                with Image.open(target) as opened:
                    refinement_static = np.asarray(opened.convert("L")).copy()
            else:
                with Image.open(image_path) as opened:
                    refinement_static = np.full(
                        (opened.height, opened.width), 255, dtype=np.uint8
                    )
            if unresolved_path.is_file():
                with Image.open(unresolved_path) as opened:
                    unresolved_static = np.asarray(opened.convert("L"))
                if unresolved_static.shape == refinement_static.shape:
                    # Black in either mask means the region must remain boxed.
                    # Pixels successfully stitched are white in the unresolved
                    # mask and therefore disappear unless manually retained.
                    refinement_static = np.minimum(
                        refinement_static, unresolved_static
                    )
            Image.fromarray(refinement_static.astype(np.uint8)).save(target)

    @staticmethod
    def _accept_refined_pixels(workspace: Path) -> None:
        """Allow human-approved repaired pixels in the formal COLMAP pass."""
        refinement_root = workspace / "refinement_masks"
        dynamic_root = workspace / "dynamic_masks"
        if not refinement_root.is_dir() or not dynamic_root.is_dir():
            return
        for refinement_path in refinement_root.rglob("*.png"):
            relative = refinement_path.relative_to(refinement_root)
            dynamic_path = dynamic_root / relative
            if not dynamic_path.is_file():
                continue
            with Image.open(refinement_path) as opened:
                selected = np.asarray(opened.convert("L")) <= 127
            if not selected.any():
                continue
            with Image.open(dynamic_path) as opened:
                dynamic_static = np.asarray(opened.convert("L")).copy()
            if dynamic_static.shape != selected.shape:
                continue
            dynamic_static[selected] = 255
            Image.fromarray(dynamic_static.astype(np.uint8)).save(dynamic_path)

    def _complete_dynamic_backgrounds(self, *, final: bool) -> None:
        assert self._workspace is not None
        assert self._processing_image_directory is not None
        # The first pass writes accepted background into static_images and
        # clears those pixels from the unresolved mask.  The final pass must
        # therefore start from that reviewed result; starting again from the
        # originals would silently put the removed moving objects back.
        source_directory = (
            self._workspace / "static_images"
            if final else self._processing_image_directory
        )
        mask_directory = self._workspace / (
            "refinement_masks" if final else "dynamic_masks"
        )
        if final:
            self._prepare_refinement_masks(source_directory, mask_directory)
        database = self._workspace / "dynamic_preview" / "database.db"
        preview_directory = self._workspace / (
            "final_background_previews" if final else "background_completion_previews"
        )
        if final:
            completion = complete_dynamic_backgrounds(
                database,
                source_directory,
                mask_directory,
                self._workspace / "static_images",
                preview_directory=preview_directory,
                report_path=self._workspace / "final_background_report.json",
                mode="refine",
                progress_callback=lambda current, total: self.photo_progress.emit(
                    current, total, "人工选区纹理补全与边界平滑",
                ),
                should_cancel=lambda: self._cancelled,
            )
            rounds = [completion]
        else:
            # Run a second forward/backward comparison only while more than
            # 20% of the original boxed pixels remain black.  Round two uses
            # the complete dataset produced by round one, so a background that
            # became visible in one frame can support its neighbours.
            rounds = []
            round_source = source_directory
            original_dynamic = 0
            for round_number in (1, 2):
                completion = complete_dynamic_backgrounds(
                    database,
                    round_source,
                    mask_directory,
                    self._workspace / "static_images",
                    preview_directory=preview_directory,
                    report_path=self._workspace /
                    f"background_completion_report_round{round_number}.json",
                    mode="multiview",
                    progress_callback=lambda current, total, number=round_number:
                    self.photo_progress.emit(
                        current, total, f"第 {number} 轮跨帧拼接静态背景"
                    ),
                    should_cancel=lambda: self._cancelled,
                )
                rounds.append(completion)
                if round_number == 1:
                    original_dynamic = completion.dynamic_pixels
                remaining_ratio = completion.unresolved_pixels / max(
                    original_dynamic, 1
                )
                self.log_received.emit(
                    f"第 {round_number} 轮拼接后，原始框选区域仍有 "
                    f"{remaining_ratio * 100:.1f}% 未恢复。\n"
                )
                if self._cancelled or remaining_ratio <= 0.20:
                    break
                round_source = self._workspace / "static_images"
            completion = rounds[-1]
            shutil.copyfile(
                completion.report_path,
                self._workspace / "background_completion_report.json",
            )
        if final:
            self.log_received.emit(
                "人工选区补全与平滑："
                f"处理 {completion.completed_images} 张照片，"
                f"共 {completion.completed_pixels} 个选中像素；"
                f"强纹理二维延伸 {completion.local_texture_pixels}，"
                f"弱纹理边界多数填充 {completion.small_hole_pixels}。\n"
            )
        else:
            original_dynamic = rounds[0].dynamic_pixels
            final_unresolved = rounds[-1].unresolved_pixels
            total_completed = max(0, original_dynamic - final_unresolved)
            self.log_received.emit(
                f"{len(rounds)} 轮跨帧背景拼接完成："
                f"累计恢复 {total_completed}/{original_dynamic} 个框选像素；"
                f"其余 {final_unresolved} 个像素保留原框并进入人工复核。\n"
            )

    def _process_error(self, error) -> None:
        if self._cancelled:
            return
        if error == QProcess.ProcessError.FailedToStart:
            self.running_changed.emit(False)
            self.failed.emit("COLMAP 无法启动，请重新选择 bin\\colmap.exe。")

    def _finish_successfully(self) -> None:
        assert self._workspace is not None
        if self._preprocess_only:
            static_images = self._workspace / "static_images"
            masks = self._workspace / "dynamic_masks"
            if not static_images.is_dir() or not any(static_images.rglob("*")):
                self._active = False
                self.running_changed.emit(False)
                self.failed.emit("照片预处理已结束，但没有生成最终静态图片数据集。")
                return
            self._active = False
            self.running_changed.emit(False)
            self.preprocessing_completed.emit(
                PreprocessingResult(self._workspace, static_images, masks)
            )
            return
        point_cloud = self._workspace / "dense" / "fused.ply"
        mesh = self._workspace / "dense" / "meshed-poisson.ply"
        if not point_cloud.is_file() or not mesh.is_file():
            self._active = False
            self.running_changed.emit(False)
            self.failed.emit("COLMAP 已执行完毕，但没有找到最终点云或 Poisson 网格。")
            return
        self._active = False
        self.running_changed.emit(False)
        self.completed.emit(
            ReconstructionResult(self._workspace, point_cloud, mesh)
        )
