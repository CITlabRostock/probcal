# config_utils.py
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


# ----------------------------
# TOML loading (Py>=3.11 + fallback)
# ----------------------------
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # Python <=3.10


ALLOWED_METHODS = {"logit", "probit", "isotonic", "bbq"}


# ----------------------------
# Defaults (optional sections)
# ----------------------------
DEFAULTS: dict[str, Any] = {
    "data": {
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True,
    },
    "bootstrap": {
        "n_bootstrap": 500,
        "ci": 0.95,
        "n_jobs": -1,
    },
    "parametric": {
        "use_regularization": False,
        "regularization_strength": 0.0,
    },
    "isotonic": {},
    "bbq": {
        "C": 5,
        "N_prime": 2.0,
    },
    "evaluation": {
        "compute_accuracy": True,
        "compute_f1": True,
        "compute_logloss": True,
        "compute_brier": True,
        "compute_ece": True,
        "compute_mce": True,
    },
    "prints": {
        "analyse_baseline": True,
        "print_youden_threshold_on_calibrated": False,
        "print_std_on_calibrated": False,
    },
    "plots": {
        "plot_baseline": True,
        "plot_calibrated_predictions": False,
        "plot_calibrated_predictions_with_data": False,
        "plot_calibrated_predictions_ci": False,
        "plot_calibration_curves": True,
        "plot_reliability_diagrams_scatter": True,
        "plot_reliability_diagrams_bar": False,
    },
}


# ----------------------------
# Public API
# ----------------------------
def load_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML config file and return it as a dict."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() != ".toml":
        raise ValueError(f"Config file must be a .toml file, got: {path.name}")

    with path.open("rb") as f:
        config = tomllib.load(f)

    if not isinstance(config, dict):
        raise ValueError("Config parsing failed: expected a TOML table at top level.")

    return config


def apply_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return a new config dict where optional sections/keys are filled with defaults.
    Existing user values are preserved.
    """
    cfg = deepcopy(dict(config))

    # Ensure required top-level keys exist (validation checks contents later)
    # (We do not fill these with defaults.)
    # - data_path
    # - label_column
    # - prob_column
    # - methods

    # Fill optional sections
    for section, defaults in DEFAULTS.items():
        user_section = cfg.get(section)

        if user_section is None:
            cfg[section] = deepcopy(defaults)
            continue

        if not isinstance(user_section, dict):
            raise TypeError(f"config.{section} must be a table/section, got {type(user_section).__name__}")

        merged = deepcopy(defaults)
        merged.update(user_section)  # user overrides defaults
        cfg[section] = merged

    return cfg


def validate_config(config: Mapping[str, Any]) -> None:
    """
    Validate the config dict. Assumes defaults have already been applied.

    Raises:
        ValueError / TypeError with a human-readable message if something is invalid.
    """
    # ---- Required keys ----
    for k in ["data_path", "label_column", "prob_column", "methods"]:
        if k not in config:
            raise ValueError(f"Missing required key: config.{k}")

    # ---- Types for required keys ----
    data_path = config["data_path"]
    label_column = config["label_column"]
    prob_column = config["prob_column"]
    methods = config["methods"]

    if not isinstance(data_path, str):
        raise TypeError(f"config.data_path must be str, got {type(data_path).__name__}")
    if not isinstance(label_column, str):
        raise TypeError(f"config.label_column must be str, got {type(label_column).__name__}")
    if not isinstance(prob_column, str):
        raise TypeError(f"config.prob_column must be str, got {type(prob_column).__name__}")

    # dataset file existence check (nice user feedback)
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # ---- Methods ----
    if not isinstance(methods, list) or len(methods) == 0:
        raise TypeError("config.methods must be a non-empty list of strings, e.g. ['logit', 'bbq']")

    for i, m in enumerate(methods):
        if not isinstance(m, str):
            raise TypeError(f"config.methods[{i}] must be a string, got {type(m).__name__}")
        if m not in ALLOWED_METHODS:
            raise ValueError(f"Unknown calibration method '{m}'. Allowed: {sorted(ALLOWED_METHODS)}")

    # ---- Data split section ----
    data_cfg = config["data"]
    _require_float_in_open_interval(data_cfg["test_size"], "config.data.test_size", 0.0, 1.0)

    _require_int(data_cfg["random_state"], "config.data.random_state")
    _require_bool(data_cfg["stratify"], "config.data.stratify")

    # ---- Bootstrap section ----
    boot = config["bootstrap"]
    _require_int(boot["n_bootstrap"], "config.bootstrap.n_bootstrap")
    if boot["n_bootstrap"] < 1:
        raise ValueError("config.bootstrap.n_bootstrap must be >= 1 (0 disables bootstrapping)")

    _require_float_in_open_interval(boot["ci"], "config.bootstrap.ci", 0.0, 1.0)
    _require_int(boot["n_jobs"], "config.bootstrap.n_jobs")
    if boot["n_jobs"] == 0:
        raise ValueError("config.bootstrap.n_jobs must not be 0")

    # ---- Parametric section ----
    par = config["parametric"]
    _require_bool(par["use_regularization"], "config.parametric.use_regularization")
    _require_float(par["regularization_strength"], "config.parametric.regularization_strength")

    if par["use_regularization"] and par["regularization_strength"] <= 0.0:
        raise ValueError(
            "config.parametric.regularization_strength must be > 0 when use_regularization=true"
        )

    # ---- BBQ section ----
    bbq = config["bbq"]
    _require_int(bbq["C"], "config.bbq.C")
    if bbq["C"] < 1:
        raise ValueError("config.bbq.C must be >= 1")

    _require_float(bbq["N_prime"], "config.bbq.N_prime")
    if bbq["N_prime"] <= 0.0:
        raise ValueError("config.bbq.N_prime must be > 0")

    # ---- Evaluation / Prints / Plots (bool flags) ----
    _require_bool_dict(config["evaluation"], "config.evaluation")
    _require_bool_dict(config["prints"], "config.prints")
    _require_bool_dict(config["plots"], "config.plots")


# ----------------------------
# Small type-check helpers
# ----------------------------
def _require_bool(x: Any, name: str) -> None:
    if not isinstance(x, bool):
        raise TypeError(f"{name} must be bool, got {type(x).__name__}")


def _require_int(x: Any, name: str) -> None:
    # bool is a subclass of int -> reject bool here
    if isinstance(x, bool) or not isinstance(x, int):
        raise TypeError(f"{name} must be int, got {type(x).__name__}")


def _require_float(x: Any, name: str) -> None:
    # allow ints? here we keep it strict: must be float
    if not isinstance(x, float):
        raise TypeError(f"{name} must be float, got {type(x).__name__}")


def _require_float_in_open_interval(x: Any, name: str, lo: float, hi: float) -> None:
    _require_float(x, name)
    if not (lo < x < hi):
        raise ValueError(f"{name} must be strictly between {lo} and {hi}, got {x}")


def _require_bool_dict(d: Any, name: str) -> None:
    if not isinstance(d, dict):
        raise TypeError(f"{name} must be a section/table, got {type(d).__name__}")
    for k, v in d.items():
        if not isinstance(v, bool):
            raise TypeError(f"{name}.{k} must be bool, got {type(v).__name__}")