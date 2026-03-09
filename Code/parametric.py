# Implementation of Parametric Models

import numpy as np
from scipy.special import ndtr, expit
from scipy.stats import norm
from scipy.optimize import minimize
from joblib import Parallel, delayed

# returns correct link function for logit or probit model
# logit: expit (sigmoid function), probit: normal cdf (cumulative distribution function of the standard normal distribution)
def get_link_function_logit_probit(model_type):
    if model_type == "logit":
        return expit
    elif model_type == "probit":
        return ndtr
    else:
        raise ValueError("Unsupported model type. Choose 'logit' or 'probit'.")

# returns averaged cross entropy loss function using L2 regularization with strength lambda_
def cost_logit_probit(w, X, t, model_type="logit", lambda_=0.0):
    # get link function
    link = get_link_function_logit_probit(model_type)
    # compute probabilities
    probs = link(X @ w)
    # ensure numerical stability by clipping probabilities to avoid log(0)
    epsilon = 1e-10
    probs = np.clip(probs, epsilon, 1 - epsilon)
    # compute mean instead of sum for regularisation and numerical stability
    cost = -np.mean(t * np.log(probs) + (1 - t) * np.log(1 - probs))
    # L2 Reg Term
    reg_term = (lambda_ / 2) * np.sum(w[1:] ** 2)
    return cost + reg_term

# returns gradient of loss function
def gradient_logit_probit(w, X, t, model_type="logit", lambda_=0.0):
    z = X @ w
    # Compute the gradient based on the model type
    if model_type == "logit":
        # For logit, we use the sigmoid function
        probs = expit(z)
        # divide by len(t) due to cost function with mean instead of sum
        grad = X.T @ (probs - t) / len(t)
    elif model_type == "probit":
        # For probit, we use the normal CDF
        Phi = norm.cdf(z)
        # Compute PDF for later use in gradient
        phi = norm.pdf(z)
        # Ensure numerical stability by clipping probabilities to avoid division by zero
        epsilon = 1e-10
        Phi = np.clip(Phi, epsilon, 1 - epsilon)
        grad_terms = ((t * phi / Phi) - ((1 - t) * phi / (1 - Phi)))
        # divide by len(t) to guarantee consistency with cost function that uses mean instead of sum
        grad = -X.T @ grad_terms / len(t)
    else:
        raise ValueError("Unsupported model type.")
    
    # Add gradient of the regularization term (L2 regularization) to stay consistent with cost function
    grad[1:] += lambda_ * w[1:]  # Exclude bias term (w[0])
    
    return grad

# finds optimal weights with help of BFGS (Broyden–Fletcher–Goldfarb–Shanno)
# BFGS is a Quasi Newton Method that approximates the Hessian matrix
def fit_logit_probit(X, t, model_type="logit", lambda_=0.0):
    # X_bias is X with a bias term (column of ones) added
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]

    # set optimizer and w_0 according to model type (w_0 for warm start)
    p = np.clip(np.mean(t), 1e-5, 1 - 1e-5)  # mean t
    if model_type == "logit":
        initial_bias = np.log(p / (1 - p))
        method = 'BFGS'
    elif model_type == "probit":
        initial_bias = norm.ppf(p)
        method = 'trust-constr'
    else:
        raise ValueError("model_type must be 'logit' or 'probit'")

    # Initialize weights with bias term
    initial_weights = np.zeros(X_bias.shape[1])
    initial_weights[0] = initial_bias

    # Use minimize function from scipy.optimize to find optimal weights
    # fun: cost function, x0: initial weights, args: additional arguments for cost
    # jac: gradient function, method: optimization method
    result = minimize(
        fun=cost_logit_probit,
        x0=initial_weights,
        args=(X_bias, t, model_type, lambda_),
        jac=gradient_logit_probit,
        method=method
    )
    return result.x

# returns array of predicted probabilities based on optimal weights w and model type
def predict_logit_probit(X, w, model_type="logit"):
    X = np.asarray(X).reshape(-1)
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    link = get_link_function_logit_probit(model_type)
    return link(X_bias @ w)

# returns array of predicted Labels based on optimal weights w and model type
# threshold is used to convert probabilities to binary labels
# default threshold is 0.5, but can be adjusted using the threshold parameter which is computed with Youden's J statistic
def predict_logit_probit_labels(X, w, model_type="logit", threshold=0.5):
    probs = predict_logit_probit(X, w, model_type)
    return (probs >= threshold).astype(int)

# fits a single bootstrap sample and returns predicted probabilities and weights
def _single_bootstrap(X, t, n_samples, model_type, lambda_, seed, i):
    # Ensure n_samples matches the number of samples in X for safe indexing
    assert n_samples == X.shape[0], "n_samples must equal X.shape[0] for safe indexing"
    # Generate random indices for bootstrap sampling
    rng = np.random.default_rng(seed + i)
    # Sample with replacement
    # indices are random integers in the range [0, n_samples)
    indices = rng.integers(0, n_samples, size=n_samples)
    # Resample X and t using the generated indices
    X_resample, t_resample = X[indices], t[indices]
    try:
        # Fit the model to the resampled data
        # w is the optimal weights for the resampled data
        w = fit_logit_probit(X_resample, t_resample, model_type=model_type, lambda_=lambda_)
        # Predict probabilities on the original data using the fitted model from the bootstrap resample
        probs = predict_logit_probit(X, w, model_type=model_type)
        return probs, w
    except Exception:
        return None

# computes bootstrap-based confidence intervals for calibrated probabilities
# uses parallel processing to speed up the bootstrap sampling
def bootstrap_ci_parametric(X, t, model_type="logit", n_bootstrap=500, ci=0.95, lambda_=0.0, seed=0, n_jobs=-1):
    '''
    Parameters:
        X: array-like of shape (n_samples, n_features)
            Calibration set scores (from base model)
        t: array-like of shape (n_samples,)
            Calibration set labels
        model_type: "logit" or "probit"
            Calibration model
        n_bootstrap: int
            Number of bootstrap samples
        ci: float
            Confidence level (e.g., 0.95 for 95% CI).
        lambda_: float
            L2 regularization strength to reduce overfitting
        seed: int or None
            Random seed
        n_jobs: int
            Number of parallel jobs (-1 = all cores available)

    Returns:
        mean_probs: array-like of shape (n_samples,)
            Mean predicted probabilities
        lower_ci: array-like of shape (n_samples,)
            Lower bound of CI
        upper_ci: array-like of shape (n_samples,)
            Upper bound of CI
        closest_weights: array-like of shape (n_features,)
            Weights of the bootstrap sample closest to mean probabilities
    '''
    # compute the number of samples in X
    n_samples = X.shape[0]

    # Parallel processing to speed up bootstrap sampling
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_bootstrap)(X, t, n_samples, model_type, lambda_, seed, i)
        for i in range(n_bootstrap)
    )

    # Filter out None results (failed bootstrap fits)
    # This is necessary to ensure we only keep successful bootstrap iterations
    results = [r for r in results if r is not None]
    if len(results) == 0:
        raise RuntimeError("All bootstrap fits failed.")

    # Unzip the results into probabilities and weights
    # probs_bootstrap contains the predicted probabilities from each bootstrap sample
    # weights_bootstrap contains the Model weights for each bootstrap sample
    probs_bootstrap, weights_bootstrap = zip(*results)
    probs_bootstrap = np.array(probs_bootstrap)
    weights_bootstrap = np.array(weights_bootstrap)

    # Compute the alpha level for the confidence interval
    alpha = (1 - ci) / 2
    # Compute the lower and upper bounds of the confidence interval - currently using percentiles in plot function and therefore not returning them
    lower = np.percentile(probs_bootstrap, 100 * alpha, axis=0)
    upper = np.percentile(probs_bootstrap, 100 * (1 - alpha), axis=0)
    # Compute the mean predicted probabilities across all bootstrap samples
    mean_probs = np.mean(probs_bootstrap, axis=0)
    # Compute the standard deviation of the predicted probabilities across all bootstrap samples
    std_per_point = np.std(probs_bootstrap, axis=0, ddof=1)
    mean_std = np.mean(std_per_point)

    # Find the weights corresponding to the closest predicted probabilities to the mean probabilities to get a representative model
    distances_mean = np.linalg.norm(probs_bootstrap - mean_probs, axis=1)
    closest_idx_mean = np.argmin(distances_mean)
    closest_weights_mean = weights_bootstrap[closest_idx_mean]

    return mean_probs, closest_weights_mean, weights_bootstrap, mean_std