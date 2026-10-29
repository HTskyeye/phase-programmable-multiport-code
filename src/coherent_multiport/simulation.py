"""Differentiable 2D Tidy3D models, preserving the supplied geometry."""

from functools import reduce
from operator import add

import autograd.numpy as anp
import numpy as np
import tidy3d as td

from .config import DesignConfig

PORTS = ("top", "middle", "bottom")


def initial_parameters(cfg: DesignConfig):
    """Use the original NumPy MT19937 sequence without changing global RNG state."""
    return np.random.RandomState(cfg.seed).random_sample(cfg.parameter_shape)


def expand_parameters(params, cfg: DesignConfig):
    """Expand the splitter's independent half along y; retain autograd tracing."""
    if params.shape != cfg.parameter_shape:
        raise ValueError(f"Expected independent parameters {cfg.parameter_shape}")
    if cfg.device == "decoder":
        return params
    reflected = params[:, ::-1] if cfg.ny % 2 == 0 else params[:, -2::-1]
    return anp.concatenate((params, reflected), axis=1)


def build_simulation(params, cfg: DesignConfig, case="middle", *, validation=False):
    """Construct, but do not submit, a unit-amplitude single-port simulation.

    Design arrays use axis 0 = x and axis 1 = y. Outputs are top / middle /
    bottom. Optimization keeps only objective monitors; validation adds
    broadband mode/flux monitors and fields at the optimization wavelengths.
    """
    if case not in cfg.cases:
        raise ValueError(f"case must be one of {cfg.cases}")
    dx, dy, pixel = cfg.design_x, cfg.design_y, cfg.pixel_um
    ys = dict(zip(PORTS, (cfg.port_pitch_um, 0.0, -cfg.port_pitch_um)))
    fixed = td.Medium(permittivity=cfg.fixed_permittivity)
    # The original overlapping waveguide boxes are intentional. The design
    # medium is placed last so it takes precedence in the design rectangle.
    guides = [
        td.Box(center=(-3, ys[p], 0), size=(6.5, cfg.waveguide_width_um, td.inf))
        for p in cfg.cases
    ]
    guides += [
        td.Box(center=(5, ys[p], 0), size=(6.5, cfg.waveguide_width_um, td.inf))
        for p in PORTS
    ]
    frame = [
        td.Box(center=(-pixel / 2, 0, 0), size=(pixel, dy + 2 * pixel, td.inf)),
        td.Box(center=(dx + pixel / 2, 0, 0), size=(pixel, dy + 2 * pixel, td.inf)),
        td.Box(center=(dx / 2, (dy + pixel) / 2, 0), size=(dx, pixel, td.inf)),
        td.Box(center=(dx / 2, -(dy + pixel) / 2, 0), size=(dx, pixel, td.inf)),
    ]
    full = expand_parameters(params, cfg)
    eps_min, eps_max = cfg.design_permittivity
    eps = anp.clip(eps_min + (eps_max - eps_min) * full, 1, cfg.fixed_permittivity)
    coords = {
        "x": np.linspace(pixel / 2, dx - pixel / 2, cfg.nx),
        "y": np.linspace((-dy + pixel) / 2, (dy - pixel) / 2, cfg.ny),
        "z": [0],
    }
    medium = td.CustomMedium(
        permittivity=td.ScalarFieldDataArray(
            eps.reshape((cfg.nx, cfg.ny, 1)), coords=coords
        )
    )
    structures = [
        td.Structure(geometry=reduce(add, guides), medium=fixed),
        td.Structure(geometry=reduce(add, frame), medium=fixed),
        td.Structure(
            geometry=td.Box(center=(dx / 2, 0, 0), size=(dx, dy, td.inf)), medium=medium
        ),
    ]
    source_freqs = td.C_0 / (np.array(cfg.source_band_nm) * 1e-3)
    freq0, fwidth = np.mean(source_freqs), np.ptp(source_freqs)
    # 1.11 um is the original modal aperture (1.2 * 0.8 + 0.15),
    # distinct from the physical port pitch of 1.2 um.
    mode_span = 1.2 * 0.8 + cfg.waveguide_width_um
    source = td.ModeSource(
        center=(-1, ys[case], 0),
        size=(0, mode_span, td.inf),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth, phase=0),
        mode_spec=td.ModeSpec(num_modes=1),
        mode_index=0,
        direction="+",
        name=f"source_{case}",
    )
    freqs = td.C_0 / (cfg.wavelengths_nm(validation) * 1e-3)
    monitors = [
        td.ModeMonitor(
            center=(dx + 1, ys[port], 0),
            size=(0, mode_span, td.inf),
            freqs=freqs,
            mode_spec=td.ModeSpec(num_modes=1),
            name=f"terminal_{i}",
        )
        for i, port in enumerate(PORTS, 1)
    ]
    if validation:
        monitors += [
            td.FluxMonitor(
                center=(dx + 1, ys[port], 0),
                size=(0, mode_span, 0),
                freqs=freqs,
                name=f"flux_{i}",
            )
            for i, port in enumerate(PORTS, 1)
        ]
        monitors += [
            td.FieldMonitor(
                center=(0, 0, 0),
                size=(td.inf, td.inf, 0),
                freqs=td.C_0 / (cfg.wavelengths_nm() * 1e-3),
                name="field",
            )
        ]
    refine = td.MeshOverrideStructure(
        geometry=td.Box(
            center=(dx / 2, 0, 0),
            size=(dx + 2 * pixel + 0.4, dy + 2 * pixel + 0.4, td.inf),
        ),
        dl=(cfg.mesh_um, cfg.mesh_um, None),
    )
    return td.Simulation(
        center=(dx / 2, 0, 0),
        size=(dx + 3, dy + 2, 0),
        grid_spec=td.GridSpec.auto(
            wavelength=np.mean(cfg.source_band_nm) * 1e-3,
            min_steps_per_wvl=16,
            override_structures=[refine],
        ),
        structures=structures,
        sources=[source],
        monitors=monitors,
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=False),
        run_time=30 / fwidth,
    )
