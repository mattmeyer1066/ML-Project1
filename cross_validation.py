"""Train, predict and cross-validate the 6 methods of implementations.py."""

import numpy as np

from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)
from preprocessing import fit_preprocessing, transform

LOGISTIC_METHODS = ["logistic_regression", "reg_logistic_regression"]


def sigmoid(t):
    """Numerically stable sigmoid."""
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def train_model(method, y, tx, lambda_=0.0, gamma=0.1, max_iters=500):
    """Train one of the 6 methods (given by its name) and return the weights."""
    w0 = np.zeros(tx.shape[1])
    if method == "mean_squared_error_gd":
        w, _ = mean_squared_error_gd(y, tx, w0, max_iters, gamma)
    elif method == "mean_squared_error_sgd":
        np.random.seed(42)  # SGD picks samples at random: fix the seed
        w, _ = mean_squared_error_sgd(y, tx, w0, max_iters, gamma)
    elif method == "least_squares":
        w, _ = least_squares(y, tx)
    elif method == "ridge_regression":
        w, _ = ridge_regression(y, tx, lambda_)
    elif method == "logistic_regression":
        w, _ = logistic_regression(y, tx, w0, max_iters, gamma)
    elif method == "reg_logistic_regression":
        w, _ = reg_logistic_regression(y, tx, lambda_, w0, max_iters, gamma)
    else:
        raise ValueError(f"Unknown method: {method}")
    return w


def predict_scores(method, tx, w):
    """Probability of class 1 for logistic methods, raw linear output otherwise."""
    if method in LOGISTIC_METHODS:
        return sigmoid(tx @ w)
    return tx @ w


def f1_score(y_true, y_pred):
    """F1 score for the positive class (labels in {0, 1})."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def best_threshold(y_true, scores):
    """Try thresholds from 0.05 to 0.5 and return the one with the best F1."""
    thresholds = np.linspace(0.05, 0.5, 46)
    f1s = [f1_score(y_true, (scores >= t).astype(int)) for t in thresholds]
    i = int(np.argmax(f1s))
    return thresholds[i], f1s[i]


def k_fold_indices(n, k=5, seed=42):
    """Shuffle the indices 0..n-1 and split them into k folds."""
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(n), k)


def cross_validate(
    y, x, names, method, k=5, config=None, threshold=None, **hyperparams
):
    """
    k-fold cross-validation of one method.

    For each fold, the preprocessing is fitted on the training folds only,
    then the model is trained and scored on the validation fold. One threshold
    is chosen on all validation scores, then the F1 is computed on each fold.

    Args:
        y: labels in {0, 1}
        x: raw features (not preprocessed)
        names: column names of x
        method: name of one of the 6 methods, e.g. "reg_logistic_regression"
        k: number of folds
        config: preprocessing options (see DEFAULT_CONFIG in preprocessing.py)
        **hyperparams: lambda_, gamma, max_iters
        threshold: if None, the F1-optimal threshold is chosen; otherwise this fixed value is used

    Returns:
        (mean_f1, std_f1, threshold)
    """
    folds = k_fold_indices(len(y), k)
    scores = np.zeros(len(y))
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        params = fit_preprocessing(x[tr_idx], names, config)
        tx_tr = transform(x[tr_idx], names, params)
        tx_val = transform(x[val_idx], names, params)

        w = train_model(method, y[tr_idx], tx_tr, **hyperparams)
        scores[val_idx] = predict_scores(method, tx_val, w)

    if threshold is None:
        threshold, _ = best_threshold(y, scores)
    f1s = [f1_score(y[f], (scores[f] >= threshold).astype(int)) for f in folds]
    return np.mean(f1s), np.std(f1s), threshold
