"""Serializable architecture settings for the controlled V2 experiment."""
from dataclasses import asdict, dataclass
import math


def validated_plan_output_scale(value):
    """A fixed XY unit conversion, not a learned state or a postprocessor."""
    if isinstance(value, (str, bytes)):
        raise ValueError("plan_output_scale must contain two positive finite numbers")
    try:
        result = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("plan_output_scale must contain two positive finite numbers") from exc
    if len(result) != 2 or any(not math.isfinite(v) or v <= 0 for v in result):
        raise ValueError("plan_output_scale must contain two positive finite numbers")
    return result


@dataclass
class MotionDriveV2Config:
    channels: int = 128
    backbone_arch: str = "resnet50"
    grid_size: tuple[int, int] = (64, 48)
    x_range: tuple[float, float] = (-10.0, 70.0)
    y_range: tuple[float, float] = (-32.0, 32.0)
    heights: tuple[float, ...] = (0.0, 1.0, 2.0)
    n_history: int = 4
    motion_grid: tuple[int, int] = (12, 16)
    correlation_channels: int = 32
    correlation_radius: int = 2
    scene_attention_channels: int = 32
    scene_chunk_size: int = 512
    planner_layers: int = 2
    planner_heads: int = 4
    goal_on: bool = True
    state_on: bool = True
    plan_output_scale: tuple[float, float] = (1.0, 1.0)
    motion_input_mode: str = "legacy"
    # Inputs use exactly the normalization of the public ResNet checkpoint.
    # Dataset/serving code supplies normalized float images, not uint8 images.

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
        for name in ("grid_size", "motion_grid", "x_range", "y_range", "heights"):
            setattr(self, name, tuple(getattr(self, name)))
        if self.channels % self.planner_heads:
            raise ValueError("channels must be divisible by planner_heads")
        if self.n_history <= 0 or min(self.grid_size + self.motion_grid) <= 0:
            raise ValueError("history and spatial dimensions must be positive")
        if self.scene_chunk_size <= 0 or self.correlation_radius < 0:
            raise ValueError("invalid attention/correlation geometry")
        self.plan_output_scale = validated_plan_output_scale(self.plan_output_scale)
        if self.motion_input_mode not in ("legacy", "high_feature", "low_feature"):
            raise ValueError("motion_input_mode must be legacy, high_feature or low_feature")
