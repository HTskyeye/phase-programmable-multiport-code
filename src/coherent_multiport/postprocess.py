"""Broadband validation and portable numerical/figure exports."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
import tidy3d.web as web

from .checkpoint import atomic_npz, load_checkpoint
from .objectives import mode_amplitudes
from .optimization import make_optimizer
from .simulation import build_simulation, initial_parameters


def preview(cfg, output_dir):
    """Export the initial geometry locally; no account or cloud call is used."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    simulation = build_simulation(initial_parameters(cfg), cfg, cfg.cases[0])
    ax = simulation.plot_eps(z=0, monitor_alpha=0, source_alpha=0)
    ax.set_title(f"{cfg.device.capitalize()}: initial permittivity")
    ax.figure.savefig(directory / "geometry_initial.png", dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    return simulation


def evaluate(cfg, output_dir, *, submit=False):
    """Run independent single-port validations of the final checkpoint geometry."""
    if not submit:
        raise ValueError(
            "Set submit=True only when ready to run paid cloud simulations"
        )
    directory = Path(output_dir)
    state = load_checkpoint(directory / "checkpoint.npz", cfg, make_optimizer(cfg))
    if state.completed != cfg.iterations:
        raise ValueError("Complete the configured optimization before final validation")
    simulations = {
        case: build_simulation(state.params, cfg, case, validation=True)
        for case in cfg.cases
    }
    data_dir = directory / "validation" / "cloud"
    data_dir.mkdir(parents=True, exist_ok=True)
    batch = web.run_async(
        simulations=simulations,
        path_dir=data_dir,
        folder_name=f"multiport_{cfg.device}_validation",
        verbose=True,
    )
    export_validation(batch, cfg, directory / "validation")
    ax = simulations[cfg.cases[0]].plot_eps(z=0, monitor_alpha=0, source_alpha=0)
    ax.set_title(f"{cfg.device.capitalize()}: optimized permittivity")
    ax.figure.savefig(
        directory / "validation" / "geometry_final.png", dpi=300, bbox_inches="tight"
    )
    plt.close(ax.figure)
    return batch


def export_validation(batch, cfg, output_dir):
    """Export T[output, input, wavelength] and separate integrated flux.

    Each input case has a unit-amplitude mode source. Phases are referenced to
    the source and monitor planes; no propagation-phase de-embedding is applied.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    wavelengths = cfg.wavelengths_nm(validation=True)
    freqs = td.C_0 / (wavelengths * 1e-3)
    transmission = np.stack(
        [np.asarray(mode_amplitudes(batch[case], freqs)) for case in cfg.cases], axis=1
    )
    flux = np.stack(
        [
            np.stack(
                [
                    np.array(
                        [
                            float(
                                batch[case][f"flux_{i}"].flux.sel(f=float(freq)).values
                            )
                            for freq in freqs
                        ]
                    )
                    for i in range(1, 4)
                ]
            )
            for case in cfg.cases
        ],
        axis=1,
    )
    powers = np.abs(transmission) ** 2
    total = powers.sum(axis=0)
    ratios = np.divide(
        powers,
        total[None, :, :],
        out=np.full_like(powers, np.nan),
        where=total[None, :, :] > 0,
    )
    atomic_npz(
        directory / "transmission.npz",
        wavelength_nm=wavelengths,
        transmission=transmission,
        mode_power=powers,
        flux_power=flux,
        total_mode_transmission=total,
        output_fractions=ratios,
        input_labels=np.array(cfg.cases),
        output_labels=np.array(["top", "middle", "bottom"]),
    )
    with (directory / "spectra.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "input",
                "wavelength_nm",
                "output",
                "amplitude_real",
                "amplitude_imag",
                "amplitude_abs",
                "phase_deg",
                "mode_power",
                "flux_power",
                "output_fraction",
            ]
        )
        for j, case in enumerate(cfg.cases):
            for k, wavelength in enumerate(wavelengths):
                for i in range(3):
                    amp = transmission[i, j, k]
                    writer.writerow(
                        [
                            case,
                            wavelength,
                            i + 1,
                            amp.real,
                            amp.imag,
                            abs(amp),
                            np.angle(amp, deg=True),
                            powers[i, j, k],
                            flux[i, j, k],
                            ratios[i, j, k],
                        ]
                    )
    for j, case in enumerate(cfg.cases):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), constrained_layout=True)
        for i in range(3):
            axes[0].plot(wavelengths, powers[i, j], label=f"O{i + 1}")
            axes[1].plot(wavelengths, flux[i, j], label=f"O{i + 1}")
        axes[0].plot(wavelengths, total[j], "k--", label="Total")
        for ax, label in zip(axes, ("Forward modal power", "Net integrated flux")):
            ax.set(
                xlabel="Wavelength (nm)", ylabel="Power / incident power", title=label
            )
            ax.legend(frameon=False)
        fig.suptitle(f"{cfg.device.capitalize()} - {case} input")
        fig.savefig(directory / f"spectra_{case}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        ax = batch[case].plot_field("field", "Ez", "abs", f=td.C_0 / 1.5)
        ax.set_title(f"{case.capitalize()} input: |Ez| at 1500 nm")
        ax.figure.savefig(
            directory / f"field_{case}_1500nm.png", dpi=300, bbox_inches="tight"
        )
        plt.close(ax.figure)
