import csv

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tidy3d as td
import xarray as xr
from types import SimpleNamespace

from coherent_multiport.postprocess import export_validation


@pytest.mark.parametrize("name", ["splitter", "decoder"])
def test_complex_export_order_and_flux(configs, tmp_path, name):
    cfg = configs[name]
    freqs = td.C_0 / (cfg.wavelengths_nm(True) * 1e-3)
    expected = np.empty((3, len(cfg.cases), 501), dtype=complex)

    class FakeData(dict):
        def plot_field(self, *args, **kwargs):
            _, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            return ax

    batch = {}
    for j, case in enumerate(cfg.cases):
        data = FakeData()
        for i in range(3):
            expected[i, j] = (i + 1) / 10 + 1j * ((j + 1) / 10 + np.arange(501) / 10000)
            data[f"terminal_{i + 1}"] = SimpleNamespace(
                amps=xr.DataArray(
                    expected[i, j][None, :, None],
                    dims=("direction", "f", "mode_index"),
                    coords={"direction": ["+"], "f": freqs, "mode_index": [0]},
                )
            )
            # Deliberately distinct from |T|^2 to detect mixing flux and mode data.
            data[f"flux_{i + 1}"] = SimpleNamespace(
                flux=xr.DataArray(
                    np.full(501, 0.123 + i / 100), dims=("f",), coords={"f": freqs}
                )
            )
        batch[case] = data
    export_validation(batch, cfg, tmp_path)
    with np.load(tmp_path / "transmission.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["transmission"], expected)
        np.testing.assert_allclose(saved["mode_power"], abs(expected) ** 2)
        np.testing.assert_allclose(saved["flux_power"][0], 0.123)
        np.testing.assert_allclose(saved["output_fractions"].sum(axis=0), 1)
        np.testing.assert_array_equal(saved["wavelength_nm"], cfg.wavelengths_nm(True))
        assert list(saved["input_labels"]) == list(cfg.cases)
    with (tmp_path / "spectra.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(cfg.cases) * 3 * 501
    assert float(rows[0]["amplitude_real"]) == 0.1
    assert len(list(tmp_path.glob("*.png"))) == 2 * len(cfg.cases)
