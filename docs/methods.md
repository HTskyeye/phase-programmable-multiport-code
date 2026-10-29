# Methods and file conventions

## Scope and physical model

These workflows reorganize the supplied splitter and decoder research notebooks into reusable modules. The equations, random seeds, geometry ordering, source settings, optimization schedules and parameter bounds are retained. The splitter uses 19 optimization wavelengths (1455–1545 nm in 5 nm steps); the decoder uses 13 (1470–1530 nm in 5 nm steps).

All lengths in the model are in micrometres. The simulation has zero z thickness and uses out-of-plane electric polarization with one propagating mode per port. It is a 2D model, not a calibrated 3D SOI device. The surrounding medium has relative permittivity 1; fixed waveguides and an 80 nm outer frame have relative permittivity 11.4. Within the design region, each bounded parameter rho maps linearly to epsilon = 4.7 + 6.3 rho. No binarization, spatial filter, or minimum-feature constraint is applied.

Arrays have axes (x, y). Splitter parameters cover half of the y extent and are mirrored about y = 0 before simulation; their gradients accumulate from both halves. Saved full arrays have shape (26, 36). Decoder parameters have shape (36, 36). Port order is top, middle, bottom, at y = +1.2, 0, -1.2. The splitter has only the middle input.

The design occupies x = 0 to design_x. Waveguide boxes retain their original overlapping extents and are added before the design medium. Sources lie at x = -1; output monitors lie at x = design_x + 1. The modal aperture is 1.11 micrometres. PML terminates x and y; the local grid override is 10 nm, with 16 automatic steps per wavelength elsewhere. Simulation duration is 30/source_frequency_width.

The splitter pulse covers 1450–1550 nm and validation samples 1440–1560 nm. The decoder pulse covers 1460–1540 nm and validation samples 1450–1550 nm. Validation uses 501 uniformly spaced wavelengths, including 1500 nm. Optimization uses only mode monitors needed by the objective; validation restores field and flux monitors and extends mode monitoring to the same 501 points. Removing unused optimization monitors does not change the device or source definition.

## Objectives and optimization

For complex output amplitudes a_j at one wavelength, define P_j = |a_j|², P_sum = sum(P_j), target fractions t_j = 1/3, and p_j = P_j/(P_sum + 1e-12). As in the original notebooks, normalize t by sum(t) + 1e-12 before use. Let

```text
A = sum_j sqrt(t_j * P_j + 1e-18)
E = sqrt(sum_j (p_j - t_j)^2 + 1e-18)
C = |sum_j a_j|^2 / (3 * P_sum + 1e-12)
J = (A^2 - lambda * E + w * C) / (1 + w)
```

The objective averages J over wavelengths and, for the decoder, over three independent single-port inputs. For the splitter, lambda = 0.1 and w = 1.5 throughout 50 updates. For the decoder, w = 0; lambda = 0 for updates 1–40 and 0.1 for updates 41–60. The coherence expression rewards balanced, phase-aligned outputs together; its value also depends on amplitude balance.

The supplied decoder used a phase-to-target lookup followed by hard/EMA target updates. For the author's constant-one lookup, every normalized target is always (1/3, 1/3, 1/3). This release expresses that fixed target directly, removing the external spreadsheet dependency. A nonconstant lookup would describe a different objective and is not supported here. The decoder also receives the splitter's 1e-18 square-root guards to keep derivatives finite at zero; this affects only numerical singularities at the stated precision.

Optax Adam maximizes the objective by applying negative gradients. The initial learning rate is 0.05 with continuous exponential decay 0.8^(step/5); Adam uses its defaults beta1 = 0.9, beta2 = 0.999 and epsilon = 1e-8. Updated parameters are clipped to [0, 1]. Initial random arrays match NumPy RandomState seeds 888 (splitter) and 777 (decoder). The code does not change JAX's precision mode; keep it unchanged when resuming.

## Checkpoints and restart semantics

`checkpoint.npz` is written atomically after each complete update and once before the first update. It contains parameters, full geometry, all Optax state leaves (including first/second moments and schedule counters), gradients, objective values and independent-parameter history. It is loaded with `allow_pickle=False`.

The stored signature checks physics, spectral grids, objective weights, optimizer settings, precision mode and core dependency versions. The requested total iteration budget is excluded so a run may be extended deliberately. Account and path settings are not part of the signature. Use one process per output directory. Use a new `--output` directory for a fresh optimization.

`completed` counts successful updates. `scores[k]` and `gradients[k]` were evaluated at `parameter_history[k]`; `params` is the updated geometry after `completed` updates. Final validation uses this last geometry, not a geometry selected by comparing objectives across the decoder's penalty change.

A failed cloud request does not advance the local optimizer. Rerunning the command restores the last complete update. An interrupted in-flight cloud step is not automatically recovered or cancelled and may incur charges if repeated; inspect its task status in Tidy3D first. Changing accounts does not require transferring old cloud data to continue optimization, but retain downloaded data if needed for later analysis. Legacy pickle checkpoints that lack optimizer state cannot provide an exact continuation and are not accepted.

## Exports and interpretation

| File | Meaning |
| --- | --- |
| `config.json` | Resolved settings for this run |
| `checkpoint.npz` | Complete optimizer restart state and histories |
| `design.npz` | Independent/full parameters, physical permittivity, completed count |
| `parameters_full.csv` | Full design parameter matrix; rows are x, columns are y |
| `history.csv` | Objective at each pre-update geometry |
| `validation/transmission.npz` | Complex amplitudes, mode powers, flux and output fractions |
| `validation/spectra.csv` | Long-form per-input, per-output spectral data |
| `validation/*.png` | Geometry, spectra and 1500 nm field plots |
| `cloud/`, `validation/cloud/` | Local Tidy3D simulation data and task records |

In the transmission archive, `transmission[output, input, wavelength]` is complex, with shape (3, 1, 501) for the splitter and (3, 3, 501) for the decoder. `input_labels` and `output_labels` specify port order; `wavelength_nm` is ascending. Independent input simulations use unit-amplitude mode sources, so `mode_power = abs(transmission)**2` is incident-normalized forward power. `total_mode_transmission` sums those three output powers. `output_fractions` divides by that sum and is NaN when the sum is zero.

`flux_power` is the net Poynting flux through each finite output aperture. It can differ from modal power because it integrates all fields and includes forward/backward contributions. Flux does not contain a complex phase and cannot reconstruct a complex transmission matrix on its own. Complex amplitudes retain source/monitor reference-plane phases; no de-embedding is applied.

The optimized decoder is simulated in its single-port equal-splitting orientation. Its saved transmission matrix is for that orientation. Phase-conjugate routing requires reciprocal excitation from the opposite side (or a consistently mirrored device and port convention) with a reference-wavelength code held fixed across the band. This package does not silently reinterpret the equal-splitting matrix as a routed-state matrix. Full forward routing and allocation searches are separate validation tasks.

Neither 1 minus total transmission nor missing output modal power is, by itself, an absorption measurement. Reflection, radiation, other modes and finite-aperture effects need separate diagnostics. Phase-only control of a fixed matrix does not guarantee that every desired output distribution is reachable.

## Validation limits

The offline tests check symmetry, configuration defaults, simulation construction, autograd connectivity, objective values/gradients, data ordering, checkpoint mismatch detection and interrupted/resumed optimizer equivalence. They block cloud API calls. These tests verify local implementation behavior, not electromagnetic convergence or experimental performance.

The tested direct dependencies are recorded in `requirements.txt`. Tidy3D's remote solver and cloud availability can change; retain cloud task records with any scientific result. No unpublished manuscript files, account credentials, precomputed optimized structures or experimental results are included in this repository.
