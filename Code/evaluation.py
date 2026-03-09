# file for evaluation metrics

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, brier_score_loss, roc_curve
from parametric import predict_logit_probit, predict_logit_probit_labels
from isotonic import predict_isotonic, predict_isotonic_labels
from bayesian import predict_bbq, predict_bbq_labels

# get n_bins for calibration metrics
def get_n_bins(t_true, min_bin_size=20, max_bins=25):
    """
    Calculate the number of bins for calibration metrics based on the size of t_true.
    Ensures at least min_bin_size samples per bin and no more than max_bins total bins.
    """
    n_samples = len(t_true)
    n_bins = min(max_bins, max(1, n_samples // min_bin_size))
    return n_bins

# Evaluate models as classifiers using accuracy and F1 score
def evaluate_as_classifier(
    X_test, t_test,
    threshold_logit, threshold_probit, threshold_iso, threshold_bbq,
    logit_weights,
    probit_weights,
    X_train, mean_iso,
    bin_edges_list, bin_means_list, bbq_weights,
    X_real=None, t_cnn=None,
    metrics=None,
    methods=None,
):
    """
    Evaluate selected calibration models as classifiers.
    Computes only metrics enabled in `metrics` and only for methods that are available.

    metrics: dict-like with keys:
        compute_accuracy, compute_f1 (bool)
    methods: optional iterable of method names to control ordering
    """
    if metrics is None:
        metrics = {"compute_accuracy": True, "compute_f1": True}

    compute_acc = bool(metrics.get("compute_accuracy", False))
    compute_f1 = bool(metrics.get("compute_f1", False))

    if not (compute_acc or compute_f1):
        return  # nothing to do

    order = list(methods) if methods is not None else ["logit", "probit", "isotonic", "bbq"]

    results = {}  # method -> dict of metric -> value

    # LOGIT
    if "logit" in order and logit_weights is not None and threshold_logit is not None:
        preds = predict_logit_probit_labels(X_test, logit_weights, model_type="logit", threshold=threshold_logit)
        res = {}
        if compute_acc:
            res["Acc"] = accuracy_score(t_test, preds)
        if compute_f1:
            res["F1"] = f1_score(t_test, preds)
        results["logit"] = res

    # PROBIT
    if "probit" in order and probit_weights is not None and threshold_probit is not None:
        preds = predict_logit_probit_labels(X_test, probit_weights, model_type="probit", threshold=threshold_probit)
        res = {}
        if compute_acc:
            res["Acc"] = accuracy_score(t_test, preds)
        if compute_f1:
            res["F1"] = f1_score(t_test, preds)
        results["probit"] = res

    # ISOTONIC
    if "isotonic" in order and X_train is not None and mean_iso is not None and threshold_iso is not None:
        preds = predict_isotonic_labels(X_train, mean_iso, X_test, threshold=threshold_iso)
        res = {}
        if compute_acc:
            res["Acc"] = accuracy_score(t_test, preds)
        if compute_f1:
            res["F1"] = f1_score(t_test, preds)
        results["isotonic"] = res

    # BBQ
    if "bbq" in order and bin_edges_list is not None and bin_means_list is not None and bbq_weights is not None and threshold_bbq is not None:
        preds = predict_bbq_labels(X_test, bin_edges_list, bin_means_list, bbq_weights, threshold=threshold_bbq)
        res = {}
        if compute_acc:
            res["Acc"] = accuracy_score(t_test, preds)
        if compute_f1:
            res["F1"] = f1_score(t_test, preds)
        results["bbq"] = res

    # CNN baseline (optional)
    if X_real is not None and t_cnn is not None:
        res = {}
        if compute_acc:
            res["Acc"] = accuracy_score(t_test, t_cnn)
        if compute_f1:
            res["F1"] = f1_score(t_test, t_cnn)
        results["cnn"] = res

    # Print
    print("Classifier Evaluation:")
    if not results:
        print("No classifier evaluations were run (missing model inputs).")
        return

    # Print each method line, only the enabled metrics
    for name in order + (["cnn"] if "cnn" in results else []):
        if name not in results:
            continue
        parts = []
        if compute_acc and "Acc" in results[name]:
            parts.append(f"Acc: {results[name]['Acc']:.4f}")
        if compute_f1 and "F1" in results[name]:
            parts.append(f"F1: {results[name]['F1']:.4f}")
        print(f"{name.upper():<9} - " + ", ".join(parts))

# ECE and MCE calculation (Guo et al. 2017)
def calibration_error(t_true, y_prob, n_bins=10, threshold=0.5): 
    """
    Computes ECE and MCE using Guo et al. (2017) definition with quantile binning.
    Returns ECE, MCE, per-bin Accuracies and Confidences.
    """

    # get binary predictions and confidence scores
    t_pred = (y_prob >= threshold).astype(int)
    p_conf = np.where(t_pred == 1, y_prob, 1 - y_prob)

    # Quantile binning
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y_prob, quantiles)
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0

    accs, confs, sizes = [], [], []

    # Iterate over bins to calculate accuracy and confidence
    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        in_bin = (y_prob >= left) & (y_prob < right) if i < n_bins - 1 else (y_prob >= left) & (y_prob <= right)

        # Skip empty bins (shouldnt happen with quantile binning but in case quantile gets switched to uniform at some point)
        if not np.any(in_bin):
            continue

        # Calculate accuracy and confidence for the bin
        acc = np.mean(t_pred[in_bin] == t_true[in_bin])
        conf = np.mean(p_conf[in_bin])
        size = np.sum(in_bin)

        accs.append(acc)
        confs.append(conf)
        sizes.append(size)

    accs = np.array(accs)
    confs = np.array(confs)
    sizes = np.array(sizes)

    # Normalize sizes to get weights for ECE
    weights = sizes / np.sum(sizes)
    ece = np.sum(weights * np.abs(accs - confs))
    mce = np.max(np.abs(accs - confs))

    return ece, mce, accs, confs

# Define a function to compute calibration metrics
def calibration_metrics(t_true, y_prob, n_bins=10, threshold=0.5):
    # Log loss and Brier score
    logloss = log_loss(t_true, y_prob)
    brier = brier_score_loss(t_true, y_prob)
    # ECE (valid because quantile bins are balanced and non empty)
    # MCE (maximum absolute calibration error)
    ece, mce, _, _ = calibration_error(t_true, y_prob, n_bins=n_bins, threshold=threshold)
    return logloss, brier, ece, mce

# Evaluates the calibration of the models as calibrators
def evaluate_as_calibrator(
    X_test, t_test,
    threshold_logit, threshold_probit, threshold_iso, threshold_bbq,
    logit_weights,
    probit_weights,
    X_train, mean_iso,
    bin_edges_list, bin_means_list, bbq_weights,
    X_real=None, threshold_cnn=None,
    metrics=None,
    methods=None,
):
    """
    Evaluate selected models as calibrators.
    Computes only metrics enabled in `metrics` and only for methods that are available.

    metrics: dict-like with keys:
        compute_logloss, compute_brier, compute_ece, compute_mce (bool)
    methods: optional iterable (e.g. methods_list from config) to control evaluation/print order
    """
    if metrics is None:
        metrics = {
            "compute_logloss": True,
            "compute_brier": True,
            "compute_ece": True,
            "compute_mce": True,
        }

    compute_logloss = bool(metrics.get("compute_logloss", False))
    compute_brier   = bool(metrics.get("compute_brier", False))
    compute_ece     = bool(metrics.get("compute_ece", False))
    compute_mce     = bool(metrics.get("compute_mce", False))

    if not (compute_logloss or compute_brier or compute_ece or compute_mce):
        return  # nothing to do

    # Ensure there are always 20 samples per bin and no more than 25 bins -> returns list of bin counts to iterate over
    n_bins = get_n_bins(t_test)

    order = list(methods) if methods is not None else ["logit", "probit", "isotonic", "bbq"]

    results = {}  # method -> dict of metric -> value

    def _compute_metrics_for_probs(probs, threshold):
        logloss, brier, ece, mce = calibration_metrics(
            t_test, probs, n_bins=n_bins, threshold=threshold
        )
        out = {}
        if compute_logloss:
            out["LogLoss"] = logloss
        if compute_brier:
            out["Brier"] = brier
        if compute_ece:
            out["ECE"] = ece
        if compute_mce:
            out["MCE"] = mce
        return out

    # LOGIT
    if "logit" in order and logit_weights is not None and threshold_logit is not None:
        logit_probs = predict_logit_probit(X_test, logit_weights, model_type="logit")
        results["logit"] = _compute_metrics_for_probs(logit_probs, threshold_logit)

    # PROBIT
    if "probit" in order and probit_weights is not None and threshold_probit is not None:
        probit_probs = predict_logit_probit(X_test, probit_weights, model_type="probit")
        results["probit"] = _compute_metrics_for_probs(probit_probs, threshold_probit)

    # ISOTONIC
    if "isotonic" in order and X_train is not None and mean_iso is not None and threshold_iso is not None:
        iso_probs = predict_isotonic(X_train, mean_iso, X_test)
        results["isotonic"] = _compute_metrics_for_probs(iso_probs, threshold_iso)

    # BBQ
    if (
        "bbq" in order
        and bin_edges_list is not None
        and bin_means_list is not None
        and bbq_weights is not None
        and threshold_bbq is not None
    ):
        bbq_probs = predict_bbq(X_test, bin_edges_list, bin_means_list, bbq_weights)
        results["bbq"] = _compute_metrics_for_probs(bbq_probs, threshold_bbq)

    # CNN baseline (optional)
    if X_real is not None and threshold_cnn is not None:
        results["cnn"] = _compute_metrics_for_probs(X_real, threshold_cnn)

    # Print
    print("Calibration Evaluation:")
    if not results:
        print("No calibration evaluations were run (missing model inputs).")
        return

    def _fmt_line(name, resdict):
        parts = []
        if "LogLoss" in resdict:
            parts.append(f"LogLoss: {resdict['LogLoss']:.4f}")
        if "Brier" in resdict:
            parts.append(f"Brier: {resdict['Brier']:.4f}")
        if "ECE" in resdict:
            parts.append(f"ECE: {resdict['ECE']:.4f}")
        if "MCE" in resdict:
            parts.append(f"MCE: {resdict['MCE']:.4f}")
        return f"{name.upper():<9} - " + ", ".join(parts)

    for name in order:
        if name in results:
            print(_fmt_line(name, results[name]))

    if "cnn" in results:
        print(_fmt_line("cnn", results["cnn"]))

# Optimal threshold for binary decision boundary using Youden's J statistic
def optimal_threshold_youden(t_true, y_proba):
    '''
    Computes the optimal classification threshold using Youden's J statistic:
        J = TPR - FPR
    It returns the threshold that maximizes the difference between True Positive Rate (TPR)
    and False Positive Rate (FPR), based on the ROC curve.
    
    Parameters:
        t_true  : Ground truth binary labels (0 or 1)
        y_proba : Predicted probabilities

    Returns:
        Optimal threshold maximizing TPR - FPR
    '''
    fpr, tpr, thresholds = roc_curve(t_true, y_proba)
    return thresholds[np.argmax(tpr - fpr)]

# Evaluate uncalibrated CNN output using classification and calibration metrics
def evaluate_uncalibrated_model(t_true, t_pred, X, threshold=0.5, metrics=None):
    """
    Evaluate CNN output before calibration using metrics controlled by `metrics`.
    """
    if metrics is None:
        metrics = {
            "compute_accuracy": True,
            "compute_f1": True,
            "compute_logloss": True,
            "compute_brier": True,
            "compute_ece": True,
            "compute_mce": True,
        }

    compute_acc     = bool(metrics.get("compute_accuracy", False))
    compute_f1      = bool(metrics.get("compute_f1", False))
    compute_logloss = bool(metrics.get("compute_logloss", False))
    compute_brier   = bool(metrics.get("compute_brier", False))
    compute_ece     = bool(metrics.get("compute_ece", False))
    compute_mce     = bool(metrics.get("compute_mce", False))

    if not (compute_acc or compute_f1 or compute_logloss or compute_brier or compute_ece or compute_mce):
        return  # nothing to do

    print("Uncalibrated CNN Performance:")

    # Classification metrics
    if compute_acc or compute_f1:
        if compute_acc:
            acc = accuracy_score(t_true, t_pred)
            print(f"Accuracy:       {acc:.4f}")
        if compute_f1:
            f1 = f1_score(t_true, t_pred)
            print(f"F1 Score:       {f1:.4f}")

    # Calibration metrics
    if compute_logloss or compute_brier or compute_ece or compute_mce:
        n_bins = get_n_bins(t_true)
        logloss, brier, ece, mce = calibration_metrics(t_true, X, n_bins=n_bins, threshold=threshold)

        if compute_logloss:
            print(f"Log Loss:       {logloss:.4f}")
        if compute_brier:
            print(f"Brier Score:    {brier:.4f}")
        if compute_ece:
            print(f"ECE (Expected): {ece:.4f}")
        if compute_mce:
            print(f"MCE (Max):      {mce:.4f}")

# Evaluate calibration for reliability plot
def evaluate_calibration_for_reliability_plot(
    X_test, t_test,
    threshold_logit, threshold_probit, threshold_iso, threshold_bbq,
    logit_weights, probit_weights,
    X_train, mean_iso,
    bin_edges_list, bin_means_list, bbq_weights,
    X_real=None, threshold_cnn=None,
):
    n_bins = get_n_bins(t_test)

    out = {}  # method -> (accs, confs)

    # Logit
    if logit_weights is not None and threshold_logit is not None:
        logit_probs = predict_logit_probit(X_test, logit_weights, model_type="logit")
        _, _, accs, confs = calibration_error(t_test, logit_probs, n_bins=n_bins, threshold=threshold_logit)
        out["logit"] = (accs, confs)

    # Probit
    if probit_weights is not None and threshold_probit is not None:
        probit_probs = predict_logit_probit(X_test, probit_weights, model_type="probit")
        _, _, accs, confs = calibration_error(t_test, probit_probs, n_bins=n_bins, threshold=threshold_probit)
        out["probit"] = (accs, confs)

    # Isotonic
    if X_train is not None and mean_iso is not None and threshold_iso is not None:
        iso_probs = predict_isotonic(X_train, mean_iso, X_test)
        _, _, accs, confs = calibration_error(t_test, iso_probs, n_bins=n_bins, threshold=threshold_iso)
        out["isotonic"] = (accs, confs)

    # BBQ
    if bin_edges_list is not None and bin_means_list is not None and bbq_weights is not None and threshold_bbq is not None:
        bbq_probs = predict_bbq(np.ravel(X_test), bin_edges_list, bin_means_list, bbq_weights)
        _, _, accs, confs = calibration_error(t_test, bbq_probs, n_bins=n_bins, threshold=threshold_bbq)
        out["bbq"] = (accs, confs)

    # CNN baseline (optional)
    if X_real is not None and threshold_cnn is not None:
        _, _, accs, confs = calibration_error(t_test, X_real, n_bins=n_bins, threshold=threshold_cnn)
        out["cnn"] = (accs, confs)

    return out