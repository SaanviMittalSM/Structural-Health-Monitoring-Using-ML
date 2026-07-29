# Structural Health Monitoring Using ML

Linear regression models that recover **strain** and **temperature** from Fiber Bragg Grating (FBG) sensor wavelength readings — a core technique in structural health monitoring, where FBG sensors are bonded to a structure and their wavelength shift under load/thermal change is used to infer strain and temperature.

## Dataset

`fbg_sensor_strain_temperature_data.csv` — 1,000 simulated samples with three FBG sensor wavelength readings (`FBG1`, `FBG2`, `FBG3`) and the corresponding ground-truth `Strain` and `Temperature`. The data is synthetically generated (seeded, reproducible) to isolate the sensor-fusion problem: a single FBG sensor's wavelength shift responds to *both* strain and temperature, so multiple sensors are needed to disambiguate the two.

## Notebooks

| Notebook | What it does |
|---|---|
| [`LR_base.ipynb`](LR_base.ipynb) | Baseline linear regression predicting Strain + Temperature jointly from raw FBG wavelengths. Compares a 2-sensor model (Model A) against a 3-sensor model (Model B), plus a variant of Model B trained as two independent single-target regressions. |
| [`LR-4_DeltaW.ipynb`](LR-4_DeltaW.ipynb) | Same comparison, but using **wavelength shift (Δλ)** relative to a baseline instead of raw wavelength — the more physically standard approach for FBG sensing. |
| [`LR_S+T.ipynb`](LR_S+T.ipynb) | Combines strain and temperature into a single weighted target and fits one regression model, then compares it against separate single-target models. Run twice — once on Δλ features, once on raw wavelength — to check whether the shift transform matters for this combined formulation. |
| [`R&L.ipynb`](R&L.ipynb) | Ridge, Lasso, and ElasticNet regression on the same 3-sensor features, comparing regularization strategies against the plain linear baseline. |
| [`ANN_S+T.ipynb`](ANN_S+T.ipynb) | A small feed-forward neural network (Keras/TensorFlow) predicting Strain + Temperature jointly, compared against the linear models. |
| [`PINN.ipynb`](PINN.ipynb) | A physics-informed neural network that adds a Bragg-equation-based physics loss term alongside the data loss, to see whether domain physics improves on the plain ANN. |

## Key Results

Using all three FBG sensors together is what makes this work — two sensors alone can't separate strain from temperature:

| Model | Strain R² | Temperature R² |
|---|---|---|
| 2 sensors (FBG1 + FBG2) | 0.38 | 0.38 |
| **3 sensors (FBG1 + FBG2 + FBG3)** | **0.997** | **0.995** |

The combined strain-temperature target model (`LR_S+T.ipynb`) reaches **R² = 0.998**, slightly outperforming the separate single-target models — and the Δλ vs. raw-wavelength feature choice makes no measurable difference once all three sensors are used.

With all three sensors, plain linear regression is already close to the ceiling, so the more complex models mostly match it rather than beat it: Ridge regression reaches **R² = 0.995–0.997**, the ANN reaches **R² = 0.994–0.998**, and the physics-informed NN is a mixed bag — strong on strain (**R² = 0.84**) but weak on temperature (**R² ≈ -0.04**, i.e. no better than predicting the mean), suggesting the physics loss term needs further tuning.

## Getting Started

### Requirements

- Python 3.10+
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `tensorflow`, `jupyter`

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn tensorflow jupyter
```

### Run it

```bash
jupyter notebook LR_base.ipynb
```

Each notebook reads `fbg_sensor_strain_temperature_data.csv` from the repo root, so no extra setup is needed — run all cells top to bottom. A few enhanced CSVs and plot images produced by earlier runs are committed for reference; re-running a notebook regenerates them in place.
