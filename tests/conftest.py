"""Keep tests local, including accidental calls through the cloud wrappers."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "TIDY3D_BASE_DIR", str(Path(tempfile.gettempdir()) / "multiport_test_tidy3d")
)

import pytest
import tidy3d as td
import tidy3d.web as web

td.config.logging.level = "ERROR"


@pytest.fixture(autouse=True)
def no_cloud(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("A test attempted to submit a cloud simulation")

    monkeypatch.setattr(web, "run", reject)
    monkeypatch.setattr(web, "run_async", reject)


@pytest.fixture
def configs():
    from coherent_multiport.config import load_config

    root = Path(__file__).resolve().parents[1]
    return {
        name: load_config(root / "configs" / f"{name}.yaml")
        for name in ("splitter", "decoder")
    }
