import numpy as np
from scipy.special import betaln, gammaln
from joblib import Parallel, delayed

# compute log marginal likelihood of Data in current bin
def compute_marginal_likelihood(k, n, alpha, beta, prior_strength_per_bin):
    # Validate inputs
    if not (0 <= k <= n):
        raise ValueError("k must be between 0 and n and k <= n must hold.")
    if alpha <= 0 or beta <= 0 or prior_strength_per_bin <= 0:
        raise ValueError("alpha, beta, and prior_strength_per_bin must be positive.")

    # Compute the log marginal likelihood with gamma global normalisation term from paper
    term_beta_ratio = betaln(alpha + k, beta + n - k) - betaln(alpha, beta)
    term_gamma = gammaln(prior_strength_per_bin) - gammaln(n + prior_strength_per_bin)
    return term_beta_ratio + term_gamma

# Fit BBQ calibration model
def fit_bbq(X, t, C=5, N_prime=2.0):
    """
    Fit BBQ using full marginal likelihood expression from the paper.

    Parameters:
    - X: Predicted probabilities of base model
    - t: True binary labels (0 or 1)
    - C: Controls bin count search range from n^(1/3)/C to n^(1/3)*C
    - N_prime: Total prior strength (distributed across bins)

    Returns:
    - final_prediction: Calibrated probabilities
    - bin_edges_list: List of bin edges per model
    - bin_means_list: List of posterior means per bin per model
    - model_weights: Posterior weight for each model
    """
    t = np.asarray(t)
    X = np.asarray(X)
    n = len(X)

    base = int(np.floor(n ** (1.0 / 3)))
    # Ensure base is at least 2 to avoid too few bins
    lower_bound = max(2, base // C)
    # Ensure there are at least 20 data points per bin
    upper_bound = min(n // 20, base * C + 1)

    # Ensures bin_counts is not empty
    if upper_bound < lower_bound:
        print("Warning: BBQ bin range fallback activated — using default range [1, 10].")
        lower_bound = 1
        upper_bound = 10
    
    # assuming bin_counts as a range from lower_bound to upper_bound
    bin_counts = list(range(lower_bound, upper_bound))

    model_preds = []
    model_log_evidences = []
    bin_edges_list = []
    bin_means_list = []

    # safeguard for p_b, alpha, beta
    eps = 1e-6

    # Loop through each bin count
    # This is the main loop that fits one BBQ model for each bin count
    for B in bin_counts:
        # Uniform Distribution of prior strength across bins
        prior_strength_per_bin = N_prime / B
        bin_edges = np.quantile(X, np.linspace(0, 1, B + 1))
        bin_edges = np.unique(bin_edges)
        bin_edges[-1] += 1e-8  # make last edge exclusive

        bin_indices = np.digitize(X, bin_edges, right=False) - 1
        bin_means = np.zeros(len(bin_edges) - 1)
        log_marginal_likelihood = 0.0

        # Iterate through each bin to compute posterior means and log marginal likelihood for that bin
        for i in range(len(bin_means)):
            mask = bin_indices == i
            # Samples in the current bin
            n_bin = np.sum(mask)
            # Number of positive samples in the current bin
            k = np.sum(t[mask])

            if n_bin > 0:
                # Set p_b to the mean of X in the current bin
                p_b = np.mean(X[mask])
            else:
                print(f"Warning: No samples in bin {i} for B={B}. Using fallback p_b=0.5.")
                # neutral fallback for empty bin - shound not happen in practice because of quantile binning
                p_b = 0.5

            p_b = np.clip(p_b, eps, 1 - eps)

            # Initialize alpha and beta for the bin using the conventions layed out in the paper
            alpha_b = prior_strength_per_bin * p_b
            beta_b = prior_strength_per_bin * (1 - p_b)

            if n_bin > 0:
                # Update alpha and beta based on the observed data in the bin by setting probability estimate to the posterior Bayesian Estimator
                bin_means[i] = (alpha_b + k) / (alpha_b + beta_b + n_bin)
                log_marginal_likelihood += compute_marginal_likelihood(k, n_bin, alpha_b, beta_b, prior_strength_per_bin)
            else:
                bin_means[i] = alpha_b / (alpha_b + beta_b)
                log_marginal_likelihood += compute_marginal_likelihood(0, 0, alpha_b, beta_b, prior_strength_per_bin)

        # Store Model Results
        preds = bin_means[bin_indices]
        model_preds.append(preds)
        model_log_evidences.append(log_marginal_likelihood)
        bin_edges_list.append(bin_edges)
        bin_means_list.append(bin_means)

    # Bayesian Model Averaging from here
    log_evidences = np.array(model_log_evidences)
    # This equates to the Softmax of P(D∣M) where D is the Data and M is the Model
    weights = np.exp(log_evidences - np.max(log_evidences))
    weights /= np.sum(weights)
    final_prediction = np.sum([w * p for w, p in zip(weights, model_preds)], axis=0)

    return final_prediction, bin_edges_list, bin_means_list, weights

# Prediction function for BBQ calibration on new data using a single BBQ Model
def predict_bbq(X, bin_edges_list, bin_means_list, model_weights):
    """
    Predict with fitted BBQ calibration model.

    Parameters:
    - X: New predicted probabilities.
    - bin_edges_list: Bin edges for each model.
    - bin_means_list: Posterior bin means for each model.
    - model_weights: Bayesian weight of each model.

    Returns:
    - Calibrated predictions (weighted average).
    """
    X = np.asarray(X)
    all_model_preds = []

    # iterate through models
    for bin_edges, bin_means in zip(bin_edges_list, bin_means_list):
        # find bin for each x \in X
        bin_indices = np.digitize(X, bin_edges, right=False) - 1
        # prevent out of bound indexing at bin edges
        bin_indices = np.clip(bin_indices, 0, len(bin_means) - 1)
        # get calibrated probability prediction for all x
        preds = bin_means[bin_indices]
        # store list of predictions for all x for the current model
        all_model_preds.append(preds)

    # compute weighted average of calibrated predictions over all models
    calibrated = np.sum([w * p for w, p in zip(model_weights, all_model_preds)], axis=0)
    return calibrated

# Predict labels with a single BBQ calibration model
def predict_bbq_labels(X, bin_edges_list, bin_means_list, model_weights, threshold=0.5):
    """
    Predict labels with fitted BBQ calibration model.

    Parameters:
    - X: New predicted probabilities.
    - bin_edges_list: Bin edges for each model.
    - bin_means_list: Posterior bin means for each model.
    - model_weights: Bayesian weight of each model.
    - threshold: Classification threshold.

    Returns:
    - Binary predicted labels (0 or 1).
    """
    # Use calibrated probabilities from BBQ
    calibrated = predict_bbq(X, bin_edges_list, bin_means_list, model_weights)
    # Apply threshold to get binary labels
    return (calibrated >= threshold).astype(int)

# fits a single BBQ bootstrap sample and returns predicted probabilities
def _single_bbq_bootstrap(X_sorted, t_sorted, n_samples, C, N_prime, seed, i):
    # Generate random indices for bootstrap sampling
    rng = np.random.default_rng(seed + i)
    # Sample with replacement: indices are random integers in the range [0, n_samples)
    indices = rng.integers(0, n_samples, size=n_samples)
    # Resample X and t using the generated indices
    X_resample, t_resample = X_sorted[indices], t_sorted[indices]

    try:
        # Fit the BBQ model to the resampled data
        # bin_edges_list and bin_means_list define the bins and their associated calibrated values
        # model_weights are used to combine ensemble members
        _, bin_edges_list, bin_means_list, model_weights = fit_bbq(
            X_resample, t_resample, C=C, N_prime=N_prime
        )
        # Predict probabilities on the original sorted X using the fitted BBQ model
        preds = predict_bbq(X_sorted, bin_edges_list, bin_means_list, model_weights)
        return preds
    except Exception:
        return None

# computes bootstrap-based confidence intervals for calibrated probabilities using BBQ method
def bootstrap_ci_bbq(X_sorted, t_sorted, n_bootstrap=500, ci=0.95, C=5, N_prime=2.0, seed=0, n_jobs=-1):
    '''
    Parameters:
        X_sorted: array-like of shape (n_samples,)
            Pre-sorted calibration set scores.
        t_sorted: array-like of shape (n_samples,)
            Labels corresponding to X_sorted.
        n_bootstrap: int
            Number of bootstrap samples.
        ci: float
            Confidence level (e.g., 0.95 for 95% CI).
        C: float
            Scaling factor for bin count search range.
        N_prime: float
            Prior strength distributed across all bins (e.g., 2.0 indicates a prior strength of 2 imaginary samples distributed across all bins).
        seed: int or None
            Random seed.
        n_jobs: int
            Number of parallel jobs (-1 = all cores available)

    Returns:
        mean: array-like of shape (n_samples,)
            Mean predicted probabilities.
        lower: array-like of shape (n_samples,)
            Lower bound of CI.
        upper: array-like of shape (n_samples,)
            Upper bound of CI.
    '''
    # Compute the number of samples in X
    n_samples = X_sorted.shape[0]

    # Parallel processing to speed up bootstrap sampling
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_bbq_bootstrap)(X_sorted, t_sorted, n_samples, C, N_prime, seed, i)
        for i in range(n_bootstrap)
    )

    # Filter out None results (failed bootstrap fits)
    if not results or all(r is None for r in results):
        raise RuntimeError("All bootstrap iterations failed. Cannot compute CI.")
    results = [r for r in results if r is not None]

    # probs_bootstrap contains the predicted probabilities from each bootstrap sample
    probs_bootstrap = np.array(results)

    # Compute the alpha level for the confidence interval
    alpha = (1 - ci) / 2
    # Compute the lower and upper bounds of the confidence interval using percentiles
    lower = np.percentile(probs_bootstrap, 100 * alpha, axis=0)
    upper = np.percentile(probs_bootstrap, 100 * (1 - alpha), axis=0)
    # Compute the mean predicted probabilities across all bootstrap samples
    mean = np.mean(probs_bootstrap, axis=0)

    # Compute the standard deviation of the predicted probabilities across all bootstrap samples
    std_per_point = np.std(probs_bootstrap, axis=0, ddof=1)
    mean_std = np.mean(std_per_point)

    return mean, lower, upper, mean_std