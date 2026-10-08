from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from rescuemap.core.dynamic_mask import generate_geometry_masks


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
) -> list[_Stage]:
    max_features, max_image_size = _quality_parameters(quality)
    database = workspace / "database.db"
    sparse = workspace / "sparse"
    sparse_model = sparse / "0"
    dense = workspace / "dense"
    fused = dense / "fused.ply"
    mesh = dense / "meshed-poisson.ply"
    exe = str(executable)

    formal_image_directory = image_directory
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
                    "--image_path", str(image_directory),
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
                    "--image_path", str(image_directory),
                    "--output_path", str(preview_sparse),
                ),
                "mapper_preview",
            ),
            _Stage("分析跨帧几何并生成动态掩膜", "", (), "generate_masks"),
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
    return preview_stages + formal_stages


class ColmapRunner(QObject):
    """Run a complete COLMAP reconstruction without blocking the Qt UI."""

    log_received = Signal(str)
    stage_changed = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)
    running_changed = Signal(bool)

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
        self._dynamic_strength = "conservative"
        self._cancelled = False

    @property
    def is_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(
        self,
        colmap_path: Path,
        image_directory: Path,
        project_directory: Path,
        quality: str = "standard",
        dynamic_suppression: bool = False,
        dynamic_strength: str = "conservative",
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

        workspace = create_run_workspace(project_directory)
        self._workspace = workspace
        self._image_directory = image_directory
        self._dynamic_strength = dynamic_strength
        self._stages = build_colmap_stages(
            executable,
            image_directory,
            workspace,
            quality,
            dynamic_suppression,
        )
        self._stage_index = -1
        self._cancelled = False
        self.log_received.emit(
            f"COLMAP：{executable}\n照片：{image_directory}\n"
            f"照片数量：{image_count}\n本次工作区：{workspace}\n"
            f"动态干扰自动抑制：{'开启' if dynamic_suppression else '关闭'}\n"
            + (
                f"动态抑制强度：{dynamic_strength}\n"
                if dynamic_suppression
                else ""
            )
        )
        self.running_changed.emit(True)
        self._start_next_stage()
        return workspace

    def cancel(self) -> None:
        if not self.is_running:
            return
        self._cancelled = True
        self.log_received.emit("\n正在取消重建任务……\n")
        self.process.kill()

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
                self.running_changed.emit(False)
                self.failed.emit(f"自动生成动态区域掩膜失败：{exc}")
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
            self.running_changed.emit(False)
            self.failed.emit("重建任务已取消，已生成的中间文件仍保留在工作区中。")
            return
        if exit_code != 0:
            stage = self._stages[self._stage_index]
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
        assert self._image_directory is not None
        summary = generate_geometry_masks(
            self._workspace / "dynamic_preview" / "database.db",
            self._image_directory,
            self._workspace / "dynamic_masks",
            self._workspace / "static_images",
            strength=self._dynamic_strength,
            preview_directory=self._workspace / "dynamic_mask_previews",
            report_path=self._workspace / "dynamic_report.json",
        )
        self.log_received.emit(
            "动态区域掩膜生成完成："
            f"{summary.masked_image_count}/{summary.image_count} 张照片检测到高置信动态区域，"
            f"屏蔽像素占比 {summary.masked_ratio * 100:.2f}%\n"
            f"平均几何静态可见率 {summary.mean_static_visibility * 100:.2f}%\n"
            "掩膜预览与逐图统计已经保存；不确定区域继续保留。\n"
        )

    def _process_error(self, error) -> None:
        if self._cancelled:
            return
        if error == QProcess.ProcessError.FailedToStart:
            self.running_changed.emit(False)
            self.failed.emit("COLMAP 无法启动，请重新选择 bin\\colmap.exe。")

    def _finish_successfully(self) -> None:
        assert self._workspace is not None
        point_cloud = self._workspace / "dense" / "fused.ply"
        mesh = self._workspace / "dense" / "meshed-poisson.ply"
        if not point_cloud.is_file() or not mesh.is_file():
            self.running_changed.emit(False)
            self.failed.emit("COLMAP 已执行完毕，但没有找到最终点云或 Poisson 网格。")
            return
        self.running_changed.emit(False)
        self.completed.emit(
            ReconstructionResult(self._workspace, point_cloud, mesh)
        )
