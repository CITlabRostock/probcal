# Implementation of NonParametric Models

import numpy as np
from joblib import Parallel, delayed

# Isotonic Regression
# only works if X and t are sorted
def fit_isotonic(X, t):
    # Fits isotonic regression using the Pool Adjacent Violators Algorithm (PAVA)
    # Assumes: X is 1D and sorted
    # Returns: X_sorted (n, 1) and corresponding fitted values p_fitted (n,)
    X = np.ravel(X)
    t = np.asarray(t, dtype=float)

    assert X.ndim == 1, "X must be 1D"
    assert X.shape[0] == t.shape[0], "X and t must have the same length"

    # Sort data in ascending order of X in case it is not sorted
    # This is necessary for isotonic regression to work correctly due to monotonicity assumption
    sorted_indices = np.argsort(X)
    X_sorted = X[sorted_indices]
    t_sorted = t[sorted_indices]

    # Call the PAVA function for isotonic regression
    p_fitted, weights, blocks = PAVA(t_sorted, np.ones_like(t_sorted))
    
    return X_sorted.reshape(-1, 1), p_fitted

# Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
def PAVA(t, w):
    n = len(t)
    # Copy target and weights to avoid modifying the original arrays
    r = t.copy()
    weights = w.copy()
    # Initialize blocks to keep track of pooled indices
    # Each block starts with its own index
    blocks = [[i] for i in range(n)]
    # Start with the second element, as the first one has no previous element to compare
    i = 1
    while i < n:
        # Find adjacent violators
        if r[i] < r[i - 1]:
            # Pool the violators
            r[i] = (weights[i] * r[i] + weights[i - 1] * r[i - 1]) / (weights[i] + weights[i - 1])
            weights[i] += weights[i - 1]
            blocks[i] = blocks[i - 1] + blocks[i]
            
            # Remove the previous entry by slicing
            r = np.delete(r, i - 1)
            weights = np.delete(weights, i - 1)
            blocks = blocks[:i - 1] + blocks[i:]

            # Decrease the total number of elements to account for the removed element
            n -= 1
            # Move back one step if possible because there is now an element less
            # Ensures O(n)
            if i > 1:
                i -= 1
        else:
            i += 1
    # After processing all elements, we have the final isotonic regression values
    probas = np.zeros_like(t)
    for i, block in enumerate(blocks):
        # Assign the pooled value to all indices in the block
        probas[block] = r[i]
    
    return probas, weights, blocks

# stepwise prediction: find largest X_train ≤ X_test and return the corresponding prediction value for all X_test in array
def predict_isotonic(X_train_sorted, y_fitted, X_test):
    X_train_sorted = X_train_sorted.flatten()
    y_fitted = y_fitted.flatten()
    X_test = np.ravel(X_test)

    preds = np.array([
        y_fitted[np.searchsorted(X_train_sorted, x, side='right') - 1]
        if x >= X_train_sorted[0] else y_fitted[0]
        for x in X_test
    ])

    return preds

# predicts isotonic labels based on fitted values and a threshold
# returns 1 if probas >= threshold, else 0
def predict_isotonic_labels(X_train_sorted, y_fitted, X_test, threshold=0.5):
    X_train_sorted = X_train_sorted.flatten()
    y_fitted = y_fitted.flatten()
    X_test = np.ravel(X_test)

    probas = np.array([
        y_fitted[np.searchsorted(X_train_sorted, x, side='right') - 1]
        if x >= X_train_sorted[0] else y_fitted[0]
        for x in X_test
    ])
    
    return (probas >= threshold).astype(int)

# fits a single isotonic regression bootstrap sample and returns predicted probabilities
def _single_isotonic_bootstrap(X_sorted, t_sorted, n_samples, seed, i):
    # Generate random indices for bootstrap sampling
    rng = np.random.default_rng(seed + i)
    # Sample with replacement: indices are random integers in the range [0, n_samples)
    indices = rng.integers(0, n_samples, size=n_samples)
    # Resample X and t using the generated indices
    X_resample, t_resample = X_sorted[indices], t_sorted[indices]
    
    try:
        # Fit the isotonic regression model to the resampled data
        # y_fitted are the calibrated probabilities corresponding to X_resample
        _, y_fitted = fit_isotonic(X_resample, t_resample)
        # Predict probabilities on the original sorted X using the fitted isotonic model
        preds = predict_isotonic(X_sorted, y_fitted, X_sorted)
        return preds
    except Exception:
        return None

# computes bootstrap-based confidence intervals for calibrated probabilities using isotonic regression
def bootstrap_ci_isotonic(X_sorted, t_sorted, n_bootstrap=500, ci=0.95, seed=0, n_jobs=-1):
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
        delayed(_single_isotonic_bootstrap)(X_sorted, t_sorted, n_samples, seed, i)
        for i in range(n_bootstrap)
    )

    # Filter out None results (failed bootstrap fits)
    results = [r for r in results if r is not None]
    if len(results) == 0:
        raise RuntimeError("All bootstrap fits failed.")

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