"""Explicit physical and optimization settings, with units in field names."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class DesignConfig:
    device: str
    nx: int
    ny: int
    pixel_um: float
    mesh_um: float
    waveguide_width_um: float
    port_pitch_um: float
    fixed_permittivity: float
    design_permittivity: tuple[float, float]
    source_band_nm: tuple[float, float]
    optimization_band_nm: tuple[float, float]
    optimization_points: int
    validation_band_nm: tuple[float, float]
    validation_points: int
    iterations: int
    seed: int
    learning_rate: float
    decay_rate: float
    decay_steps: int
    ratio_penalty: float
    penalty_start: int
    phase_weight: float

    def __post_init__(self):
        if self.device not in {"splitter", "decoder"}:
            raise ValueError("device must be 'splitter' or 'decoder'")
        for name in (
            "nx",
            "ny",
            "optimization_points",
            "validation_points",
            "iterations",
            "decay_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.optimization_points < 2 or self.validation_points < 2:
            raise ValueError("Each wavelength grid needs at least two points")
        for name in (
            "pixel_um",
            "mesh_um",
            "waveguide_width_um",
            "port_pitch_um",
            "fixed_permittivity",
            "learning_rate",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "source_band_nm",
            "optimization_band_nm",
            "validation_band_nm",
            "design_permittivity",
        ):
            values = tuple(getattr(self, name))
            if len(values) != 2 or not 0 < values[0] < values[1]:
                raise ValueError(f"{name} must contain two increasing positive values")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, values)
        if not 0 < self.decay_rate <= 1:
            raise ValueError("decay_rate must be in (0, 1]")
        if not 0 <= self.penalty_start or type(self.penalty_start) is not int:
            raise ValueError("penalty_start must be a nonnegative integer")
        if not 0 <= self.seed < 2**32 or type(self.seed) is not int:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if any(
            not np.isfinite(v) or v < 0 for v in (self.ratio_penalty, self.phase_weight)
        ):
            raise ValueError("Objective weights must be finite and nonnegative")
        if self.device == "decoder" and self.phase_weight != 0:
            raise ValueError("Decoder training has no phase-coherence term")
        if self.port_pitch_um + self.waveguide_width_um / 2 > self.design_y / 2:
            raise ValueError("Output waveguides must fit within the design height")
        if not (
            self.source_band_nm[0]
            <= self.optimization_band_nm[0]
            < self.optimization_band_nm[1]
            <= self.source_band_nm[1]
        ):
            raise ValueError("Optimization wavelengths must lie inside the source band")
        if self.design_permittivity[1] > self.fixed_permittivity:
            raise ValueError("Design permittivity must not exceed the fixed material")

    @property
    def design_x(self):
        return self.nx * self.pixel_um

    @property
    def design_y(self):
        return self.ny * self.pixel_um

    @property
    def parameter_shape(self):
        return (self.nx, (self.ny + 1) // 2 if self.device == "splitter" else self.ny)

    @property
    def cases(self):
        return ("middle",) if self.device == "splitter" else ("top", "middle", "bottom")

    def wavelengths_nm(self, validation=False):
        band = self.validation_band_nm if validation else self.optimization_band_nm
        count = self.validation_points if validation else self.optimization_points
        return np.linspace(*band, count)

    def penalty(self, step):
        return self.ratio_penalty if step >= self.penalty_start else 0.0

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> DesignConfig:
    """Read one YAML configuration; unknown keys fail instead of being ignored."""
    with Path(path).open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return DesignConfig(**settings)
