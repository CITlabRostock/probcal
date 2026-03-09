# probcal
Calibration of the outputs of neural networks

## Dataset

Several example datasets are included. Four are real-world datasets derived from Alzheimer’s disease studies. These datasets are fully anonymized and provided under neutral filenames to prevent any linkage to specific cohorts or studies. They each include around 500 samples.

Three additional datasets are synthetic and are explicitly designed to illustrate characteristic behaviors of the supported calibration methods. Synthetic dataset filenames reflect the calibration behavior they were generated to demonstrate. They each include 5000 samples.

Input data must be provided as a CSV file with a header row.

Each row represents one sample. The dataset must contain two columns:

`y_prob`: uncalibrated probability prediction of the base model, float in the range [0,1].

`y_true`: ground-truth binary label, must be either 0 or 1.

This framework currently supports binary classification only.

Missing values are not allowed.

Example:

```csv
y_prob,y_true
0.83,1
0.12,0
0.54,1
0.01,0
```

## Installation

Python 3.10 or newer is recommended. Install dependencies using:

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended but not required.

## Configuration

All experiments are controlled through a TOML configuration file.

The configuration defines the dataset path, calibration methods, evaluation metrics, bootstrap settings, and which plots are generated.

Only metrics and plots explicitly enabled in the configuration are computed.

Two examples configs are provided.

## Calibration Methods

The framework supports logit calibration, probit calibration, isotonic regression, and Bayesian Binning into Quantiles (BBQ).

The uncalibrated base model is evaluated as a baseline.

## Running

Run an experiment using:

```python
python path/to/probcal.py path/to/config.toml
```

The script loads the data, splits it into calibration and test sets, fits the selected calibration models, computes the enabled metrics, and generates optional plots.

## Evaluation

Supported classification metrics are accuracy and F1 score.

Supported calibration metrics are log loss, Brier score, expected calibration error (ECE), and maximum calibration error (MCE).

Classification thresholds are chosen using Youden’s J statistic on the calibration set.

## Output

Results are printed to the console.

Optional plots include:

calibrated prediction curves, calibration curves, reliability diagrams, bootstrap confidence intervals.

