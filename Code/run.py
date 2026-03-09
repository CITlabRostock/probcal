# run.py
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from config_utils import load_config, apply_defaults, validate_config

# Your existing imports (adjust module names to match your repo)
from data import load_data, analyse_data, predict_labels_from_threshold

from evaluation import (
    optimal_threshold_youden,
    evaluate_uncalibrated_model,
    evaluate_as_classifier,
    evaluate_as_calibrator,
)

from parametric import bootstrap_ci_parametric, predict_logit_probit
from isotonic import bootstrap_ci_isotonic, predict_isotonic
from bayesian import bootstrap_ci_bbq, fit_bbq, predict_bbq

from plots import (
    plot_probabilistic_prediction_all,
    plot_probabilistic_prediction_all_with_data,
    plot_probabilistic_prediction_all_with_ci_bootstrap,
    plot_calibration_curve,
    plot_reliability_diagrams_all_scatter,
    plot_reliability_diagrams_all_bars,
)


def run_experiment(config_path: str) -> None:
    # ---- Load + validate config ----
    config = load_config(config_path)
    config = apply_defaults(config)
    validate_config(config)

    methods_list = list(config["methods"])
    methods_set = set(methods_list)
    dataset_name = Path(config["data_path"]).stem

    data_cfg = config["data"]
    boot_cfg = config["bootstrap"]
    eval_cfg = config["evaluation"]
    print_cfg = config["prints"]
    plot_cfg = config["plots"]

    bbq_cfg = config["bbq"]

    # ---- Load dataset ----
    # Expected: X = y_prob in [0,1], t = y_true in {0,1}
    X, t = load_data(
        config["data_path"],
        prob_column=config["prob_column"],
        label_column=config["label_column"],
    )

    X = np.asarray(X).reshape(-1)
    t = np.asarray(t).astype(int).reshape(-1)
    eps = 1e-12
    X = np.clip(X, eps, 1.0 - eps)

    # ---- Baseline analysis + baseline evaluation on full dataset ----
    if print_cfg["analyse_baseline"] or plot_cfg["plot_baseline"]:
        print("Analysing Entire Dataset:")
        analyse_data(
            X, t, dataset_name,
            do_print=print_cfg["analyse_baseline"],
            do_plot=plot_cfg["plot_baseline"],
        )

    threshold_cnn_all = optimal_threshold_youden(t, X)
    preds_CNN_all = predict_labels_from_threshold(X, threshold_cnn_all)

    do_any_eval = any(eval_cfg.values())
    if print_cfg["analyse_baseline"] and do_any_eval:
        print("Evaluating Uncalibrated CNN Model on entire dataset:")
        evaluate_uncalibrated_model(
            t, preds_CNN_all, X,
            threshold=threshold_cnn_all,
            metrics=eval_cfg,
        )

    # ---- Train/test split ----
    stratify = t if data_cfg["stratify"] else None

    X_cal, X_test, t_cal, t_test = train_test_split(
        X,
        t,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=stratify,
    )

    # sort for plotting readability
    idx = np.argsort(X_cal)
    X_cal, t_cal = X_cal[idx], t_cal[idx]

    idx = np.argsort(X_test)
    X_test, t_test = X_test[idx], t_test[idx]

    # Baseline threshold on calibration set (for comparability)
    threshold_cnn = optimal_threshold_youden(t_cal, X_cal)

    # ---- Fit calibration models (bootstrapping always) ----
    # Regularization parameter for parametric methods
    par_cfg = config["parametric"]
    lambda_ = float(par_cfg["regularization_strength"]) if par_cfg["use_regularization"] else 0.0

    fitted = {}
    stds = {}
    thresholds = {}

    # LOGIT
    if "logit" in methods_set:
        logit_probs_cal, logit_weights, logit_weights_bootstrap, std_logit = bootstrap_ci_parametric(
            X_cal,
            t_cal,
            model_type="logit",
            n_bootstrap=boot_cfg["n_bootstrap"],
            ci=boot_cfg["ci"],
            lambda_=lambda_,
            seed=data_cfg["random_state"],
            n_jobs=boot_cfg["n_jobs"],
        )
        fitted["logit"] = {
            "weights": logit_weights,
            "weights_bootstrap": logit_weights_bootstrap,
            "probs_cal": logit_probs_cal,
        }
        stds["logit"] = std_logit
        thresholds["logit"] = optimal_threshold_youden(t_cal, logit_probs_cal)

    # PROBIT
    if "probit" in methods_set:
        probit_probs_cal, probit_weights, probit_weights_bootstrap, std_probit = bootstrap_ci_parametric(
            X_cal,
            t_cal,
            model_type="probit",
            n_bootstrap=boot_cfg["n_bootstrap"],
            ci=boot_cfg["ci"],
            lambda_=lambda_,
            seed=data_cfg["random_state"],
            n_jobs=boot_cfg["n_jobs"],
        )
        fitted["probit"] = {
            "weights": probit_weights,
            "weights_bootstrap": probit_weights_bootstrap,
            "probs_cal": probit_probs_cal,
        }
        stds["probit"] = std_probit
        thresholds["probit"] = optimal_threshold_youden(t_cal, probit_probs_cal)

    # ISOTONIC
    if "isotonic" in methods_set:
        mean_iso, lower_iso, upper_iso, std_iso = bootstrap_ci_isotonic(
            X_cal,
            t_cal,
            n_bootstrap=boot_cfg["n_bootstrap"],
            ci=boot_cfg["ci"],
            seed=data_cfg["random_state"],
            n_jobs=boot_cfg["n_jobs"],
        )
        fitted["isotonic"] = {
            "X_cal": X_cal,
            "mean": mean_iso,
            "lower": lower_iso,
            "upper": upper_iso,
        }
        stds["isotonic"] = std_iso
        thresholds["isotonic"] = optimal_threshold_youden(t_cal, mean_iso)

    # BBQ
    if "bbq" in methods_set:
        mean_bbq, lower_bbq, upper_bbq, std_bbq = bootstrap_ci_bbq(
            X_cal,
            t_cal,
            n_bootstrap=boot_cfg["n_bootstrap"],
            ci=boot_cfg["ci"],
            C=bbq_cfg["C"],
            N_prime=bbq_cfg["N_prime"],
            seed=data_cfg["random_state"],
            n_jobs=boot_cfg["n_jobs"],
        )

        # Fit BBQ model (for test-time prediction)
        _, bin_edges_list, bin_means_list, bbq_weights = fit_bbq(
            X_cal,
            t_cal,
            C=bbq_cfg["C"],
            N_prime=bbq_cfg["N_prime"],
        )

        fitted["bbq"] = {
            "mean": mean_bbq,
            "lower": lower_bbq,
            "upper": upper_bbq,
            "bin_edges_list": bin_edges_list,
            "bin_means_list": bin_means_list,
            "bbq_weights": bbq_weights,
        }
        stds["bbq"] = std_bbq
        thresholds["bbq"] = optimal_threshold_youden(t_cal, mean_bbq)

    # ---- Optional prints ----
    if print_cfg["print_youden_threshold_on_calibrated"]:
        print("\nOptimal Thresholds:")
        print(f"CNN (calibration set): {threshold_cnn}")
        print(f"CNN (entire dataset):  {threshold_cnn_all}")
        for name in methods_list:
            if name in thresholds:
                print(f"{name}: {thresholds[name]}")


    if print_cfg["print_std_on_calibrated"]:
        print("\nMean standard deviation of calibrated probabilities (bootstrap):")
        for name in methods_list:
            if name in stds:
                print(f"{name}: {stds[name]}")

    # ---- Predictions on test set ----
    probs_test = {}

    if "logit" in methods_set:
        probs_test["logit"] = predict_logit_probit(
            X_test, fitted["logit"]["weights"], model_type="logit"
        )

    if "probit" in methods_set:
        probs_test["probit"] = predict_logit_probit(
            X_test, fitted["probit"]["weights"], model_type="probit"
        )

    if "isotonic" in methods_set:
        probs_test["isotonic"] = predict_isotonic(
            fitted["isotonic"]["X_cal"],
            fitted["isotonic"]["mean"],
            X_test,
        )

    if "bbq" in methods_set:
        probs_test["bbq"] = predict_bbq(
            X_test,
            fitted["bbq"]["bin_edges_list"],
            fitted["bbq"]["bin_means_list"],
            fitted["bbq"]["bbq_weights"],
        )

    preds_CNN_test = predict_labels_from_threshold(X_test, threshold_cnn)

    # ---- Evaluation ----
    do_classifier_eval = eval_cfg.get("compute_accuracy", False) or eval_cfg.get("compute_f1", False)
    do_calibrator_eval = (
        eval_cfg.get("compute_logloss", False)
        or eval_cfg.get("compute_brier", False)
        or eval_cfg.get("compute_ece", False)
        or eval_cfg.get("compute_mce", False)
    )

    if do_classifier_eval:
        print("\nEvaluating selected Models as Classifiers on test data:")
        evaluate_as_classifier(
            X_test,
            t_test,
            thresholds.get("logit"),
            thresholds.get("probit"),
            thresholds.get("isotonic"),
            thresholds.get("bbq"),
            fitted.get("logit", {}).get("weights"),
            fitted.get("probit", {}).get("weights"),
            fitted.get("isotonic", {}).get("X_cal"),
            fitted.get("isotonic", {}).get("mean"),
            fitted.get("bbq", {}).get("bin_edges_list"),
            fitted.get("bbq", {}).get("bin_means_list"),
            fitted.get("bbq", {}).get("bbq_weights"),
            X_real=X_test,
            t_cnn=preds_CNN_test,
            metrics=eval_cfg,
            methods=methods_list,
        )

    if do_calibrator_eval:
        print("\nEvaluating selected Models as Calibrators on test data:")
        evaluate_as_calibrator(
            X_test,
            t_test,
            thresholds.get("logit"),
            thresholds.get("probit"),
            thresholds.get("isotonic"),
            thresholds.get("bbq"),
            fitted.get("logit", {}).get("weights"),
            fitted.get("probit", {}).get("weights"),
            fitted.get("isotonic", {}).get("X_cal"),
            fitted.get("isotonic", {}).get("mean"),
            fitted.get("bbq", {}).get("bin_edges_list"),
            fitted.get("bbq", {}).get("bin_means_list"),
            fitted.get("bbq", {}).get("bbq_weights"),
            X_real=X_test,
            threshold_cnn=threshold_cnn,
            metrics=eval_cfg,
            methods=methods_list,
        )

    # ---- Plots ----
    if plot_cfg["plot_calibrated_predictions"]:
        plot_probabilistic_prediction_all(
            X_cal,
            fitted.get("logit", {}).get("weights"),
            fitted.get("probit", {}).get("weights"),
            fitted.get("isotonic", {}).get("mean"),
            fitted.get("bbq", {}).get("mean"),
        )

    if plot_cfg["plot_calibrated_predictions_with_data"]:
        plot_probabilistic_prediction_all_with_data(
            X_cal,
            t_cal,
            fitted.get("logit", {}).get("weights"),
            fitted.get("probit", {}).get("weights"),
            fitted.get("isotonic", {}).get("mean"),
            fitted.get("bbq", {}).get("mean"),
        )

    if plot_cfg["plot_calibrated_predictions_ci"]:
        plot_probabilistic_prediction_all_with_ci_bootstrap(
            X_cal,
            fitted.get("logit", {}).get("weights_bootstrap"),
            fitted.get("probit", {}).get("weights_bootstrap"),
            fitted.get("isotonic", {}).get("mean"),
            fitted.get("isotonic", {}).get("lower"),
            fitted.get("isotonic", {}).get("upper"),
            fitted.get("bbq", {}).get("mean"),
            fitted.get("bbq", {}).get("lower"),
            fitted.get("bbq", {}).get("upper"),
            alpha=1.0 - boot_cfg["ci"],
            grid_size=500,
            gap_threshold=0.05,
        )

    if plot_cfg["plot_calibration_curves"] and (len(probs_test) > 0):
        plot_calibration_curve(
            t_test,
            probs_test.get("logit"),
            probs_test.get("probit"),
            probs_test.get("isotonic"),
            probs_test.get("bbq"),
            X_test,
        )

    if plot_cfg["plot_reliability_diagrams_scatter"]:
        plot_reliability_diagrams_all_scatter(
            X_test,
            t_test,
            thresholds.get("logit"),
            thresholds.get("probit"),
            thresholds.get("isotonic"),
            thresholds.get("bbq"),
            fitted.get("logit", {}).get("weights"),
            fitted.get("probit", {}).get("weights"),
            fitted.get("isotonic", {}).get("X_cal"),
            fitted.get("isotonic", {}).get("mean"),
            fitted.get("bbq", {}).get("bin_edges_list"),
            fitted.get("bbq", {}).get("bin_means_list"),
            fitted.get("bbq", {}).get("bbq_weights"),
            X_real=X_test,
            threshold_cnn=threshold_cnn,
            methods=methods_list,
        )

    if plot_cfg["plot_reliability_diagrams_bar"]:
        plot_reliability_diagrams_all_bars(
            X_test,
            t_test,
            thresholds.get("logit"),
            thresholds.get("probit"),
            thresholds.get("isotonic"),
            thresholds.get("bbq"),
            fitted.get("logit", {}).get("weights"),
            fitted.get("probit", {}).get("weights"),
            fitted.get("isotonic", {}).get("X_cal"),
            fitted.get("isotonic", {}).get("mean"),
            fitted.get("bbq", {}).get("bin_edges_list"),
            fitted.get("bbq", {}).get("bin_means_list"),
            fitted.get("bbq", {}).get("bbq_weights"),
            X_real=X_test,
            threshold_cnn=threshold_cnn,
            methods=methods_list,
        )