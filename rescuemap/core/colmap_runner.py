from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from rescuemap.core.background_completion import complete_dynamic_backgrounds
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

    def cancel(self) -> None:
        if not self.is_running:
            return
        self._cancelled = True
        self.log_received.emit("\n正在取消重建任务……\n")
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
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
            self.log_received.emit("最终静态图片数据集已确认，开始正式三维重建。\n")
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
            try:
                self._generate_dynamic_masks()
            except Exception as exc:
                self._active = False
                self.running_changed.emit(False)
                self.failed.emit(f"自动生成动态区域掩膜失败：{exc}")
                return
            QTimer.singleShot(0, self._start_next_stage)
            return
        if stage.kind == "enhance_images":
            try:
                self._enhance_images()
            except Exception as exc:
                self._active = False
                self.running_changed.emit(False)
                self.failed.emit(f"夜间图像增强失败：{exc}")
                return
            QTimer.singleShot(0, self._start_next_stage)
            return
        if stage.kind == "review_enhancement":
            assert self._workspace is not None
            assert self._image_directory is not None
            self.enhancement_review_requested.emit(
                str(self._image_directory), str(self._workspace / "enhanced_images")
            )
            return
        if stage.kind in {"review_initial", "review_completed"}:
            assert self._workspace is not None
            assert self._processing_image_directory is not None
            image_directory = (
                self._processing_image_directory
                if stage.kind == "review_initial"
                else self._workspace / "static_images"
            )
            self.review_requested.emit(
                stage.kind, str(image_directory), str(self._workspace / "dynamic_masks")
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
            try:
                self._complete_dynamic_backgrounds(final=stage.kind == "complete_final")
            except Exception as exc:
                self._active = False
                self.running_changed.emit(False)
                self.failed.emit(f"多视角背景恢复失败：{exc}")
                return
            QTimer.singleShot(0, self._start_next_stage)
            return
        self.process.start(stage.command, list(stage.arguments))

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
        completion = complete_dynamic_backgrounds(
            self._workspace / "dynamic_preview" / "database.db",
            source_directory,
            self._workspace / "dynamic_masks",
            self._workspace / "static_images",
            preview_directory=self._workspace / (
                "final_background_previews" if final else "background_completion_previews"
            ),
            report_path=self._workspace / (
                "final_background_report.json" if final else "background_completion_report.json"
            ),
        )
        self.log_received.emit(
            f"{'最终' if final else '首次'}多视角背景恢复："
            f"{completion.completed_images} 张照片获得可信补全，"
            f"补全 {completion.completed_pixels}/{completion.dynamic_pixels} 个动态像素，"
            f"其中强纹理延伸 {completion.strong_texture_pixels}，"
            f"弱纹理投影验证 {completion.weak_texture_pixels}；"
            f"其余 {completion.unresolved_pixels} 个像素保持屏蔽。\n"
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
