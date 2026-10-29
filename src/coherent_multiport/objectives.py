"""Numerical merit functions and the differentiable cloud evaluation boundary."""

from pathlib import Path

import autograd.numpy as anp
import tidy3d as td
import tidy3d.web as web

from .simulation import build_simulation


def mode_amplitudes(data, freqs):
    """Return complex amplitudes with shape (output port, wavelength).

    Select scalar values before stacking to preserve Tidy3D/autograd tracing.
    Flux monitors cannot provide these complex amplitudes or their phases.
    """
    return anp.stack(
        [
            anp.stack(
                [
                    data[f"terminal_{i}"]
                    .amps.sel(direction="+", mode_index=0, f=float(freq))
                    .values
                    for freq in freqs
                ]
            )
            for i in range(1, 4)
        ]
    )


def merit(amplitudes, ratio_penalty, phase_weight=0.0):
    """Average the original equal-splitting objective over wavelengths.

    The 1e-18 guards remove the derivative singularity at exactly zero power
    or error. They were already present in the supplied splitter objective.
    """
    target = anp.ones((3, 1)) / 3
    target = target / (anp.sum(target) + 1e-12)
    powers = anp.abs(amplitudes) ** 2
    total = anp.sum(powers, axis=0)
    ratios = powers / (total + 1e-12)
    overlap = anp.sum(anp.sqrt(target * powers + 1e-18), axis=0)
    error = anp.sqrt(anp.sum((ratios - target) ** 2, axis=0) + 1e-18)
    coherence = anp.abs(anp.sum(amplitudes, axis=0)) ** 2 / (3 * total + 1e-12)
    return anp.mean(
        (overlap**2 - ratio_penalty * error + phase_weight * coherence)
        / (1 + phase_weight)
    )


def cloud_objective(params, cfg, step, output_dir):
    """Submit forward/adjoint work when differentiated by autograd."""
    simulations = {case: build_simulation(params, cfg, case) for case in cfg.cases}
    path = Path(output_dir) / "cloud" / f"step_{step + 1:03d}"
    path.mkdir(parents=True, exist_ok=True)
    batch = web.run_async(
        simulations=simulations,
        path_dir=path,
        folder_name=f"multiport_{cfg.device}",
        verbose=False,
    )
    freqs = td.C_0 / (cfg.wavelengths_nm() * 1e-3)
    return anp.mean(
        anp.stack(
            [
                merit(
                    mode_amplitudes(batch[case], freqs),
                    cfg.penalty(step),
                    cfg.phase_weight,
                )
                for case in cfg.cases
            ]
        )
    )
