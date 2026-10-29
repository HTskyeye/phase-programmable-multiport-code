from dataclasses import replace
from types import SimpleNamespace

from autograd import grad, value_and_grad
import autograd.numpy as anp
import jax
import numpy as np
import pytest
import tidy3d as td

from coherent_multiport.checkpoint import load_checkpoint
from coherent_multiport.objectives import cloud_objective, merit
from coherent_multiport.optimization import make_optimizer, optimize
from coherent_multiport.simulation import (
    build_simulation,
    expand_parameters,
    initial_parameters,
)


def test_research_defaults_and_symmetry(configs):
    splitter, decoder = configs["splitter"], configs["decoder"]
    np.testing.assert_array_equal(splitter.wavelengths_nm(), np.arange(1455, 1546, 5))
    np.testing.assert_array_equal(decoder.wavelengths_nm(), np.arange(1470, 1531, 5))
    assert splitter.iterations == 50 and decoder.iterations == 60
    assert splitter.parameter_shape == (26, 18) and decoder.parameter_shape == (36, 36)
    assert decoder.penalty(39) == 0 and decoder.penalty(40) == 0.1
    p = initial_parameters(splitter)
    full = expand_parameters(p, splitter)
    np.testing.assert_array_equal(full, full[:, ::-1])
    derivative = grad(lambda x: anp.sum(expand_parameters(x, splitter) ** 2))(p)
    np.testing.assert_allclose(derivative, 4 * p)
    # An odd-height grid must not duplicate the centre column.
    odd = replace(splitter, ny=37)
    q = initial_parameters(odd)
    assert expand_parameters(q, odd).shape == (26, 37)
    d = grad(lambda x: anp.sum(expand_parameters(x, odd)))(q)
    np.testing.assert_array_equal(d[:, -1], 1)
    np.testing.assert_array_equal(d[:, :-1], 2)


@pytest.mark.parametrize("name", ["splitter", "decoder"])
def test_geometry_and_monitor_grids(configs, name):
    cfg = configs[name]
    params = initial_parameters(cfg)
    for case in cfg.cases:
        sim = build_simulation(params, cfg, case)
        assert sim.size == (cfg.design_x + 3, cfg.design_y + 2, 0)
        assert len(sim.sources) == 1 and len(sim.monitors) == 3
        assert sim.sources[0].mode_index == 0 and sim.sources[0].direction == "+"
        assert sim.grid_spec.override_structures[0].dl == (0.01, 0.01, None)
        eps = sim.structures[-1].medium.permittivity.values[..., 0]
        np.testing.assert_allclose(eps, 4.7 + 6.3 * expand_parameters(params, cfg))
        np.testing.assert_allclose(
            sim.monitors[0].freqs, td.C_0 / (cfg.wavelengths_nm() * 1e-3)
        )
        validation = build_simulation(params, cfg, case, validation=True)
        assert len(validation.monitors) == 7
        assert (
            len(validation.monitors[0].freqs)
            == len(validation.monitors[3].freqs)
            == 501
        )
        np.testing.assert_array_equal(
            validation.monitors[0].freqs, validation.monitors[3].freqs
        )
    with pytest.raises(ValueError):
        build_simulation(params, cfg, "not_a_port")


def legacy_merit(a, penalty, weight):
    """Independent scalar transcription of the supplied notebook objective."""
    values = []
    for k in range(a.shape[1]):
        target = anp.array([1 / 3] * 3)
        target = target / (anp.sum(target) + 1e-12)
        power = anp.abs(a[:, k]) ** 2
        total = anp.sum(power)
        ratios = power / (total + 1e-12)
        overlap = anp.sum(anp.sqrt(target * power + 1e-18))
        error = anp.sqrt(anp.sum((ratios - target) ** 2) + 1e-18)
        coherence = anp.abs(anp.sum(a[:, k])) ** 2 / (3 * total + 1e-12)
        values.append(
            (overlap**2 - penalty * error + weight * coherence) / (1 + weight)
        )
    return anp.mean(anp.stack(values))


@pytest.mark.parametrize("penalty,weight", [(0, 0), (0.1, 0), (0.1, 1.5)])
def test_objective_value_and_gradient(penalty, weight):
    x = np.random.RandomState(12).uniform(0.1, 0.9, (3, 19))
    phases = np.random.RandomState(13).uniform(-2, 2, x.shape)
    factor = np.exp(1j * phases)
    new = lambda p: merit(p * factor, penalty, weight)
    old = lambda p: legacy_merit(p * factor, penalty, weight)
    np.testing.assert_allclose(new(x), old(x), rtol=1e-13)
    np.testing.assert_allclose(grad(new)(x), grad(old)(x), rtol=1e-11, atol=1e-12)
    # Independent directional finite-difference check.
    direction = np.random.RandomState(14).normal(size=x.shape)
    h = 1e-6
    fd = (new(x + h * direction) - new(x - h * direction)) / (2 * h)
    np.testing.assert_allclose(
        np.sum(grad(new)(x) * direction), fd, rtol=1e-7, atol=1e-9
    )


def test_differentiable_simulation_to_objective(configs, monkeypatch, tmp_path):
    """Trace through CustomMedium -> mock modal data -> objective, with no FDTD."""
    import coherent_multiport.objectives as module

    cfg = configs["decoder"]

    class Amplitude:
        def __init__(self, amplitude):
            self.amplitude = amplitude

        def sel(self, **kwargs):
            return SimpleNamespace(values=self.amplitude)

    def fake_run(simulations, **kwargs):
        result = {}
        for case, sim in simulations.items():
            eps = sim.structures[-1].medium.permittivity.values
            strength = anp.mean(eps) / 20
            result[case] = {
                f"terminal_{i}": SimpleNamespace(
                    amps=Amplitude(strength * (1 + 0.1j * i))
                )
                for i in range(1, 4)
            }
        return result

    monkeypatch.setattr(module.web, "run_async", fake_run)
    value, gradient = value_and_grad(lambda p: cloud_objective(p, cfg, 0, tmp_path))(
        initial_parameters(cfg)
    )
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert gradient.shape == cfg.parameter_shape and np.linalg.norm(gradient) > 0


@pytest.mark.parametrize("name", ["splitter", "decoder"])
def test_complete_adam_resume(configs, tmp_path, name):
    cfg = replace(configs[name], iterations=7, penalty_start=3)

    def calculate(params, step):
        target = 0.3 if step < 3 else 0.7
        return -float(np.sum((params - target) ** 2)), -2 * (params - target)

    whole = optimize(cfg, tmp_path / "whole", value_gradient_fn=calculate)
    optimize(cfg, tmp_path / "resumed", max_steps=3, value_gradient_fn=calculate)
    resumed = optimize(cfg, tmp_path / "resumed", value_gradient_fn=calculate)
    assert resumed.completed == 7
    np.testing.assert_array_equal(whole.params, resumed.params)
    np.testing.assert_array_equal(whole.scores, resumed.scores)
    np.testing.assert_array_equal(whole.gradients, resumed.gradients)
    for a, b in zip(
        jax.tree_util.tree_leaves(whole.opt_state),
        jax.tree_util.tree_leaves(resumed.opt_state),
    ):
        np.testing.assert_array_equal(a, b)
    archive = np.load(tmp_path / "resumed" / "checkpoint.npz", allow_pickle=False)
    assert archive["gradients"].shape[0] == 7
    assert archive["full_params"].shape == (cfg.nx, cfg.ny)
    with pytest.raises(ValueError, match="differ"):
        changed = replace(cfg, mesh_um=0.02)
        load_checkpoint(
            tmp_path / "resumed" / "checkpoint.npz", changed, make_optimizer(changed)
        )


def test_failure_keeps_completed_checkpoint(configs, tmp_path):
    cfg = replace(configs["decoder"], iterations=4)

    def calculate(p, step):
        if step == 2:
            raise RuntimeError("Simulated interrupted cloud call")
        return float(p.mean()), np.ones_like(p)

    with pytest.raises(RuntimeError):
        optimize(cfg, tmp_path, value_gradient_fn=calculate)
    state = load_checkpoint(tmp_path / "checkpoint.npz", cfg, make_optimizer(cfg))
    assert state.completed == 2
    assert len(state.gradients) == 2


def test_submission_requires_explicit_opt_in(configs, tmp_path):
    from coherent_multiport.postprocess import evaluate

    with pytest.raises(ValueError, match="submit"):
        optimize(configs["splitter"], tmp_path)
    with pytest.raises(ValueError, match="submit"):
        evaluate(configs["splitter"], tmp_path)
