"""Atomic, pickle-free checkpoints including the complete Optax state."""

from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import os
import tempfile

import jax
import numpy as np

from .simulation import expand_parameters

SCHEMA_VERSION = 1


@dataclass
class RunState:
    params: np.ndarray
    opt_state: object
    completed: int = 0
    scores: list = field(default_factory=list)
    gradients: list = field(default_factory=list)
    parameter_history: list = field(default_factory=list)


def run_signature(cfg):
    """Reject incompatible physics, objectives, optimizer settings or runtimes.

    Iteration budget and account details do not determine optimizer evolution.
    No account details are read or stored here.
    """
    settings = cfg.to_dict()
    settings.pop("iterations")
    metadata = {
        "schema": SCHEMA_VERSION,
        "settings": settings,
        "versions": {
            name: version(name)
            for name in ("tidy3d", "autograd", "numpy", "jax", "jaxlib", "optax")
        },
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }
    encoded = json.dumps(metadata, sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest(), metadata


def atomic_npz(path, **arrays):
    """Replace the destination only after a complete file has been flushed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(path, state, cfg):
    signature, metadata = run_signature(cfg)
    leaves, _ = jax.tree_util.tree_flatten(state.opt_state)
    arrays = {f"opt_state_{i:03d}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    arrays.update(
        schema=np.array(SCHEMA_VERSION),
        signature=np.array(signature),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        completed=np.array(state.completed),
        params=np.asarray(state.params),
        full_params=np.asarray(expand_parameters(state.params, cfg)),
        scores=np.asarray(state.scores),
        gradients=np.asarray(state.gradients).reshape((-1,) + cfg.parameter_shape),
        parameter_history=np.stack(state.parameter_history),
        opt_state_leaf_count=np.array(len(leaves)),
    )
    atomic_npz(path, **arrays)


def load_checkpoint(path, cfg, optimizer):
    """Restore moments and both Adam/schedule counters without resetting them."""
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["schema"]) != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint schema")
        signature, _ = run_signature(cfg)
        if str(archive["signature"]) != signature:
            raise ValueError(
                "Checkpoint settings/runtime differ; use the original configuration"
            )
        params = archive["params"].copy()
        if params.shape != cfg.parameter_shape:
            raise ValueError("Checkpoint parameter shape is incompatible")
        leaves, tree = jax.tree_util.tree_flatten(optimizer.init(params))
        if int(archive["opt_state_leaf_count"]) != len(leaves):
            raise ValueError("Checkpoint optimizer structure is incompatible")
        saved = [archive[f"opt_state_{i:03d}"].copy() for i in range(len(leaves))]
        if any(a.shape != np.shape(b) for a, b in zip(saved, leaves)):
            raise ValueError("Checkpoint optimizer array shapes are incompatible")
        state = RunState(
            params=params,
            opt_state=jax.tree_util.tree_unflatten(tree, saved),
            completed=int(archive["completed"]),
            scores=archive["scores"].tolist(),
            gradients=list(archive["gradients"].copy()),
            parameter_history=list(archive["parameter_history"].copy()),
        )
    if not (
        len(state.scores)
        == len(state.gradients)
        == state.completed
        == len(state.parameter_history) - 1
    ):
        raise ValueError("Checkpoint history length is inconsistent")
    if not np.all(np.isfinite(state.params)) or np.any((params < 0) | (params > 1)):
        raise ValueError("Checkpoint contains invalid parameters")
    return state
