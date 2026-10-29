"""Adam ascent with deterministic initialization and complete step checkpoints."""

from pathlib import Path
import json
import logging

from autograd import value_and_grad
import numpy as np
import optax

from .checkpoint import RunState, atomic_npz, load_checkpoint, save_checkpoint
from .objectives import cloud_objective
from .simulation import expand_parameters, initial_parameters

LOGGER = logging.getLogger(__name__)


def make_optimizer(cfg):
    schedule = optax.exponential_decay(
        init_value=cfg.learning_rate,
        transition_steps=cfg.decay_steps,
        decay_rate=cfg.decay_rate,
        transition_begin=0,
    )
    return optax.adam(schedule)


def optimize(cfg, output_dir, *, submit=False, max_steps=None, value_gradient_fn=None):
    """Resume automatically and perform at most ``max_steps`` additional updates.

    Cloud work requires ``submit=True``. ``value_gradient_fn(params, step)`` is
    a dependency-injection point for offline tests, returning (value, gradient).
    A failed or interrupted step leaves the last complete checkpoint intact.
    """
    if value_gradient_fn is None and not submit:
        raise ValueError(
            "Set submit=True only when ready to run paid cloud simulations"
        )
    if max_steps is not None and (type(max_steps) is not int or max_steps < 0):
        raise ValueError("max_steps must be a nonnegative integer")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.npz"
    optimizer = make_optimizer(cfg)
    if checkpoint.exists():
        state = load_checkpoint(checkpoint, cfg, optimizer)
    else:
        params = initial_parameters(cfg)
        state = RunState(
            params=params,
            opt_state=optimizer.init(params),
            parameter_history=[params.copy()],
        )
        save_checkpoint(checkpoint, state, cfg)
    if state.completed > cfg.iterations:
        raise ValueError(
            "Configured iterations are below the completed checkpoint step"
        )
    (directory / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2), encoding="utf-8"
    )
    stop = (
        cfg.iterations
        if max_steps is None
        else min(cfg.iterations, state.completed + max_steps)
    )
    for step in range(state.completed, stop):
        if value_gradient_fn is None:
            evaluate = value_and_grad(
                lambda p: cloud_objective(p, cfg, step, directory)
            )
            value, gradient = evaluate(state.params)
        else:
            value, gradient = value_gradient_fn(state.params, step)
        gradient = np.asarray(gradient)
        if gradient.shape != cfg.parameter_shape:
            raise ValueError("Objective returned a gradient with the wrong shape")
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError(
                "Non-finite objective or gradient; previous checkpoint preserved"
            )
        updates, opt_state = optimizer.update(-gradient, state.opt_state, state.params)
        params = np.asarray(np.clip(optax.apply_updates(state.params, updates), 0, 1))
        if not np.all(np.isfinite(params)):
            raise FloatingPointError(
                "Non-finite Adam update; previous checkpoint preserved"
            )
        state = RunState(
            params=params,
            opt_state=opt_state,
            completed=step + 1,
            scores=state.scores + [float(value)],
            gradients=state.gradients + [gradient.copy()],
            parameter_history=state.parameter_history + [params.copy()],
        )
        save_checkpoint(checkpoint, state, cfg)
        LOGGER.info(
            "%s update %d/%d: objective %.8f",
            cfg.device,
            step + 1,
            cfg.iterations,
            value,
        )
    export_design(state, cfg, directory)
    return state


def export_design(state, cfg, directory):
    """Export the final updated design, distinct from pre-update objective values."""
    directory = Path(directory)
    full = np.asarray(expand_parameters(state.params, cfg))
    eps_lo, eps_hi = cfg.design_permittivity
    atomic_npz(
        directory / "design.npz",
        params=state.params,
        full_params=full,
        permittivity=eps_lo + (eps_hi - eps_lo) * full,
        completed=np.array(state.completed),
        config_json=np.array(json.dumps(cfg.to_dict())),
    )
    history = np.column_stack((np.arange(1, state.completed + 1), state.scores))
    np.savetxt(
        directory / "history.csv",
        history,
        delimiter=",",
        header="update,objective_before_update",
        comments="",
        fmt=["%d", "%.12g"],
    )
    np.savetxt(directory / "parameters_full.csv", full, delimiter=",")
