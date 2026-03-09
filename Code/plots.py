# Implementation of all the Plots

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from parametric import predict_logit_probit
from evaluation import get_n_bins, evaluate_calibration_for_reliability_plot

# Plot data points colored by class labels
def plot_data(X, t, name):
    plt.figure(figsize=(8, 6))

    plt.scatter(X[t == 1], t[t == 1], color='green', alpha=0.5, label='t=1')
    plt.scatter(X[t == 0], t[t == 0], color='red', alpha=0.5, label='t=0')

    plt.xlabel('Uncalibrated Data (X)')
    plt.ylabel('True Label (t)')
    plt.title(f'Plot of {name} Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# plot logit, probit, and isotonic regression without data
def plot_probabilistic_prediction_all(X, logit_weights_mean, probit_weights_mean, y_iso, y_bbq):
    x_grid = np.linspace(0, 1, 500)

    sorted_indices = np.argsort(X.flatten())
    X_sorted = X[sorted_indices]

    plt.figure(figsize=(8, 6))

    if logit_weights_mean is not None:
        logit_probs = predict_logit_probit(x_grid, logit_weights_mean, model_type="logit")
        plt.plot(x_grid, logit_probs, label="Logit", linewidth=2, color="blue")

    if probit_weights_mean is not None:
        probit_probs = predict_logit_probit(x_grid, probit_weights_mean, model_type="probit")
        plt.plot(x_grid, probit_probs, label="Probit", linewidth=2, color="green")

    if y_iso is not None:
        iso_probs = y_iso[sorted_indices]
        plt.plot(X_sorted, iso_probs, label="Isotonic", linestyle="--", linewidth=2, color="orange")

    if y_bbq is not None:
        bbq_probs = y_bbq[sorted_indices]
        plt.plot(X_sorted, bbq_probs, label="BBQ", linestyle="-.", linewidth=2, color="red")

    plt.title("Probabilistic Prediction Plot")
    plt.xlabel("Uncalibrated Data (X)")
    plt.ylabel("Predicted Probability (Y)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# plot logit, probit, and isotonic regression with data
def plot_probabilistic_prediction_all_with_data(X, t, logit_weights_mean, probit_weights_mean, y_iso, y_bbq):
    # Sort data for plotting isotonic/bbq + true labels
    sorted_indices = np.argsort(np.ravel(X))
    X_sorted = np.ravel(X)[sorted_indices]
    t_sorted = np.ravel(t)[sorted_indices]

    plt.figure(figsize=(8, 6))

    # Generate smooth grid over [0,1] for logit/probit
    x_grid = np.linspace(0, 1, 500)

    # Logit / Probit curves (only if available)
    if logit_weights_mean is not None:
        logit_probs = predict_logit_probit(x_grid, logit_weights_mean, model_type="logit")
        plt.plot(x_grid, logit_probs, label="Logit", linewidth=2, color="blue")

    if probit_weights_mean is not None:
        probit_probs = predict_logit_probit(x_grid, probit_weights_mean, model_type="probit")
        plt.plot(x_grid, probit_probs, label="Probit", linewidth=2, color="green")
    # Isotonic + BBQ curves on sorted X (only if available)
    if y_iso is not None:
        iso_probs = np.ravel(y_iso)[sorted_indices]
        plt.plot(X_sorted, iso_probs, label="Isotonic", linestyle="--", linewidth=2, color="orange")
    if y_bbq is not None:
        bbq_probs = np.ravel(y_bbq)[sorted_indices]
        plt.plot(X_sorted, bbq_probs, label="BBQ", linestyle="-.", linewidth=2, color="red")

    # True labels as scatter (always available)
    plt.scatter(X_sorted, t_sorted, color="black", alpha=0.3, label="True Labels", s=20)

    plt.title("Probabilistic Prediction with True Labels", fontsize=14)
    plt.xlabel("Uncalibrated Data (X)", fontsize=12)
    plt.ylabel("Calibrated Probability (Y) / True Label", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="gray")
    plt.tight_layout()
    plt.show()

# plot calibration curve
def plot_calibration_curve(t_true, logit_probs, probit_probs, isotonic_probs, bbq_probs, uncalibrated_probs=None):
    """
    Plots calibration curves for whichever methods are provided (not None).
    If uncalibrated_probs is provided, also plots the base model.
    """
    n_bins = get_n_bins(t_true)

    plt.figure(figsize=(8, 6))

    # Optional: base model
    if uncalibrated_probs is not None:
        base_fraction, base_mean_prob = calibration_curve(
            t_true, uncalibrated_probs, n_bins=n_bins, strategy="quantile"
        )
        plt.plot(base_mean_prob, base_fraction, marker="o", label="Uncalibrated CNN", linestyle=":", linewidth=2, color="purple")

    # Helper to add a curve only if probs exist
    def add_curve(probs, label, linestyle="-", color=None):
        if probs is None:
            return
        frac, mean_prob = calibration_curve(t_true, probs, n_bins=n_bins, strategy="quantile")
        plt.plot(mean_prob, frac, marker="o", label=label, linestyle=linestyle, linewidth=2, color=color)

    add_curve(logit_probs, "Logit", linestyle="-", color="blue")
    add_curve(probit_probs, "Probit", linestyle="-", color="green")
    add_curve(isotonic_probs, "Isotonic Regression", linestyle="--", color="orange")
    add_curve(bbq_probs, "BBQ Calibration", linestyle="-.", color="red")

    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfectly Calibrated")

    plt.title("Calibration Curves", fontsize=14)
    plt.xlabel("Mean Predicted Probability in Bin", fontsize=12)
    plt.ylabel("Fraction of Positives in Bin", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()

# Plot all probabilistic predictions with confidence intervals
def plot_probabilistic_prediction_all_with_ci_bootstrap(
    X,
    logit_weights_bootstrap,
    probit_weights_bootstrap,
    iso_mean, iso_lower, iso_upper,
    bbq_mean, bbq_lower, bbq_upper,
    alpha=0.05,
    grid_size=500,
    gap_threshold=0.05
):
    x_grid = np.linspace(0, 1, grid_size)

    X_flat = np.ravel(X)

    # Compute gap mask
    dists = np.min(np.abs(x_grid[:, None] - X_flat[None, :]), axis=1)
    mask_gap = dists > gap_threshold

    def compute_probs(weights_list, model_type):
        probs = np.array([predict_logit_probit(x_grid, w, model_type=model_type) for w in weights_list])
        lower = np.percentile(probs, 100 * (alpha / 2), axis=0)
        upper = np.percentile(probs, 100 * (1 - alpha / 2), axis=0)
        mean = np.mean(probs, axis=0)
        return mean, lower, upper

    def slice_into_regions(mask):
        edges = np.diff(mask.astype(int))
        idx = np.where(edges != 0)[0] + 1
        idx = np.r_[0, idx, len(mask)]
        for i in range(len(idx) - 1):
            yield mask[idx[i]], slice(idx[i], idx[i+1])

    def plot_with_gap(ax, title, mean, lower, upper, color):
        ax.plot(x_grid, mean, label=title, color=color)
        for is_gap, region in slice_into_regions(mask_gap):
            c = "red" if is_gap else color
            a = 0.5 if is_gap else 0.2
            ax.fill_between(x_grid[region], lower[region], upper[region], color=c, alpha=a)
        ax.set_title(f"{title} with CI")
        ax.set_xlabel("Uncalibrated Data (X)")
        ax.set_ylabel("Calibrated Probability (Y)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", linewidth=0.5)

    def mark_missing(ax, title):
        ax.set_title(f"{title} (not selected)")
        ax.axis("off")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    # Logit
    if logit_weights_bootstrap is not None:
        logit_mean, logit_lower, logit_upper = compute_probs(logit_weights_bootstrap, "logit")
        plot_with_gap(axes[0, 0], "Logit", logit_mean, logit_lower, logit_upper, color="blue")
    else:
        mark_missing(axes[0, 0], "Logit")

    # Probit
    if probit_weights_bootstrap is not None:
        probit_mean, probit_lower, probit_upper = compute_probs(probit_weights_bootstrap, "probit")
        plot_with_gap(axes[0, 1], "Probit", probit_mean, probit_lower, probit_upper, color="green")
    else:
        mark_missing(axes[0, 1], "Probit")

    # Isotonic
    ax = axes[1, 0]
    if iso_mean is not None and iso_lower is not None and iso_upper is not None:
        ax.plot(X_flat, iso_mean, label="Isotonic", color="orange")
        ax.fill_between(X_flat, iso_lower, iso_upper, color="orange", alpha=0.2)
        ax.set_title("Isotonic with CI")
        ax.set_xlabel("Uncalibrated Data (X)")
        ax.set_ylabel("Calibrated Probability (Y)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", linewidth=0.5)
    else:
        mark_missing(ax, "Isotonic")

    # BBQ
    ax = axes[1, 1]
    if bbq_mean is not None and bbq_lower is not None and bbq_upper is not None:
        ax.step(X_flat, bbq_mean, label="BBQ", color="red")
        ax.fill_between(X_flat, bbq_lower, bbq_upper, step="mid", color="red", alpha=0.2)
        ax.set_title("BBQ with CI")
        ax.set_xlabel("Uncalibrated Data (X)")
        ax.set_ylabel("Calibrated Probability (Y)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", linewidth=0.5)
    else:
        mark_missing(ax, "BBQ")

    plt.show()

# Plot reliability diagrams for all models using bar plot
def plot_reliability_diagrams_all_bars(
    X_test, t_test,
    threshold_logit, threshold_probit, threshold_iso, threshold_bbq,
    logit_weights, probit_weights,
    X_train, mean_iso,
    bin_edges_list, bin_means_list, bbq_weights,
    X_real=None, threshold_cnn=None,
    methods=None,
):
    results = evaluate_calibration_for_reliability_plot(
        X_test, t_test,
        threshold_logit, threshold_probit, threshold_iso, threshold_bbq,
        logit_weights, probit_weights,
        X_train, mean_iso,
        bin_edges_list, bin_means_list, bbq_weights,
        X_real=X_real, threshold_cnn=threshold_cnn
    )

    panels = ["logit", "probit", "isotonic", "bbq"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.flatten()

    COLOR_MAP = {
        "Logit": "blue",
        "Probit": "green",
        "Isotonic": "orange",
        "Bbq": "red",
        "CNN": "purple",
    }

    def make_bins(confs, accs, n_bins=10):
        confs = np.asarray(confs)
        accs = np.asarray(accs)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers, bin_heights, bin_widths = [], [], []

        for i in range(n_bins):
            left, right = bin_edges[i], bin_edges[i + 1]
            mask = (confs >= left) & (confs < right)
            if np.any(mask):
                bin_acc = np.mean(accs[mask])
                center = (left + right) / 2
                bin_centers.append(center)
                bin_heights.append(bin_acc)
                bin_widths.append(right - left)

        return bin_centers, bin_heights, bin_widths

    def plot_bar(ax, confs, accs, title):
        centers, heights, widths = make_bins(confs, accs)
        color = COLOR_MAP.get(title, "gray")

        ax.bar(
            centers,
            heights,
            width=widths,
            align="center",
            edgecolor="black",
            color=color,
            alpha=0.8,
        )
        for c, a in zip(centers, heights):
            ax.plot([c, c], [c, a], color="black", linewidth=2)

        ax.plot([0, 1], [0, 1], linestyle="--", color="black")
        ax.set_title(title)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def mark_missing(ax, title):
        ax.set_title(f"{title} (not selected)")
        ax.axis("off")

    for ax, name in zip(axes, panels):
        title = name.capitalize()
        if name in results:
            accs, confs = results[name]  # dict stores (accs, confs)
            plot_bar(ax, confs, accs, title)
        else:
            mark_missing(ax, title)

    plt.suptitle("Reliability Diagrams (Guo et al. 2017)", fontsize=14)
    plt.show()

    # Optional CNN plot
    if "cnn" in results:
        accs, confs = results["cnn"]
        fig_cnn, ax_cnn = plt.subplots(figsize=(8, 6), constrained_layout=True)
        plot_bar(ax_cnn, confs, accs, "CNN")
        plt.show()

# Plot reliability diagrams for all models using scatter plot
def plot_reliability_diagrams_all_scatter(
    X_test, t_test,
    threshold_logit, threshold_probit, threshold_iso, threshold_bbq,
    logit_weights, probit_weights,
    X_train, mean_iso,
    bin_edges_list, bin_means_list, bbq_weights,
    X_real=None, threshold_cnn=None,
    methods=None,
):
    results = evaluate_calibration_for_reliability_plot(
        X_test, t_test,
        threshold_logit, threshold_probit, threshold_iso, threshold_bbq,
        logit_weights, probit_weights,
        X_train, mean_iso,
        bin_edges_list, bin_means_list, bbq_weights,
        X_real=X_real, threshold_cnn=threshold_cnn
    )

    order = list(methods) if methods is not None else ["logit", "probit", "isotonic", "bbq"]
    panels = ["logit", "probit", "isotonic", "bbq"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.flatten()

    COLOR_MAP = {
        "Logit": "blue",
        "Probit": "green",
        "Isotonic": "orange",
        "Bbq": "red",
        "CNN": "purple",
    }

    def plot_scatter_with_gap_lines(ax, confs, accs, title):
        confs, accs = np.array(confs), np.array(accs)
        color = COLOR_MAP.get(title, "gray")

        for c, a in zip(confs, accs):
            ax.plot([c, c], [c, a], linewidth=1, color=color)

        ax.scatter(confs, accs, color=color)
        ax.plot([0, 1], [0, 1], linestyle="--", color="black")
        ax.set_title(title)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def mark_missing(ax, title):
        ax.set_title(f"{title} (not selected)")
        ax.axis("off")

    for ax, name in zip(axes, panels):
        title = name.capitalize()
        if name in results:
            accs, confs = results[name]
            plot_scatter_with_gap_lines(ax, confs, accs, title)
        else:
            mark_missing(ax, title)

    plt.suptitle("Reliability Diagrams for selected Models", fontsize=14)
    plt.show()

    # CNN separate (optional)
    if "cnn" in results:
        accs, confs = results["cnn"]
        fig_cnn, ax_cnn = plt.subplots(figsize=(8, 6), constrained_layout=True)
        plot_scatter_with_gap_lines(ax_cnn, confs, accs, "CNN")
        plt.show()