"""Figures for the report: PDFs in results/figures/, numbers in results/*.csv."""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cross_validation import (
    cross_validate,
    train_model,
    predict_scores,
    f1_score,
    k_fold_indices,
)
from preprocessing import (
    load_data_cached,
    load_feature_names,
    fit_preprocessing,
    transform,
    to_01,
)

SMALL = False  # True: 1/50 of the data for quick tests. False: final figures.

RESULTS = "results"
FIGURES = os.path.join(RESULTS, "figures")

# Best reg. logistic hyperparameters (see experiments.py)
BEST = dict(lambda_=1e-6, gamma=1, max_iters=1000)

# Same hyperparameters as the baselines in experiments.py
BASELINES = [
    ("MSE GD", "mean_squared_error_gd", dict(gamma=0.1, max_iters=300)),
    ("MSE SGD", "mean_squared_error_sgd", dict(gamma=0.001, max_iters=50000)),
    ("Least sq.", "least_squares", {}),
    ("Ridge", "ridge_regression", dict(lambda_=1e-4)),
    ("Logistic", "logistic_regression", dict(gamma=0.5, max_iters=500)),
    (
        "Reg. logistic",
        "reg_logistic_regression",
        dict(lambda_=1e-4, gamma=0.5, max_iters=500),
    ),
]

ABLATION = [
    ("Full", None),
    ("No indicators", {"indicators": False}),
    ("No one-hot", {"one_hot": False}),
    ("No clipping", {"clip_pct": 100}),
    ("Squares", {"poly": True}),
]

# Consistent colors: blue = main model (reg. logistic), grey = others
BLUE = "#1f77b4"
ORANGE = "#d62728"
GREY = "#8c8c8c"
GAMMA_COLORS = ["#9ecae1", "#3182bd", "#08306b"]

WIDTH = 3.4  # one column of a two-column LaTeX report, in inches
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # embed TrueType fonts
    }
)


def save_csv(filename, header, rows):
    """Write rows (lists of values) to results/<filename> as comma-separated text."""
    with open(os.path.join(RESULTS, filename), "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(
                ",".join(
                    f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v)
                    for v in row
                )
                + "\n"
            )


def save_fig(fig, filename):
    fig.savefig(os.path.join(FIGURES, filename))
    plt.close(fig)
    print(f"  wrote {os.path.join(FIGURES, filename)}")


def out_of_fold_scores(y, x, names, method, k=5, **hyperparams):
    """Validation scores of every sample, same folds as cross_validate."""
    folds = k_fold_indices(len(y), k)
    scores = np.zeros(len(y))
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        params = fit_preprocessing(x[tr_idx], names, None)
        tx_tr = transform(x[tr_idx], names, params)
        tx_val = transform(x[val_idx], names, params)
        w = train_model(method, y[tr_idx], tx_tr, **hyperparams)
        scores[val_idx] = predict_scores(method, tx_val, w)
    return scores


def plot_threshold(y, x, names):
    """Fig. 1: out-of-fold F1 as a function of the decision threshold."""
    scores = out_of_fold_scores(y, x, names, "reg_logistic_regression", **BEST)
    thresholds = np.round(np.arange(0.05, 0.6001, 0.01), 2)
    f1s = np.array([f1_score(y, (scores >= t).astype(int)) for t in thresholds])
    best = thresholds[np.argmax(f1s)]
    f1_05 = f1s[np.argmin(np.abs(thresholds - 0.5))]
    save_csv("threshold.csv", ["threshold", "f1"], zip(thresholds, f1s))

    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    ax.plot(thresholds, f1s, color=BLUE, lw=1.5)
    ax.axvline(best, color=BLUE, ls="--", lw=1, label=f"chosen $t$ = {best:.2f}")
    ax.axvline(0.5, color=ORANGE, ls=":", lw=1, label="$t$ = 0.5")
    ax.plot([best, 0.5], [f1s.max(), f1_05], "o", color="k", ms=3)
    ax.set_xlabel("Decision threshold $t$ on $P(y=1 \\mid x)$")
    ax.set_ylabel("Out-of-fold F1")
    ax.set_xlim(0.05, 0.6)
    ax.legend(frameon=False)
    save_fig(fig, "f1_vs_threshold.pdf")
    print(f"  best threshold {best:.2f} (F1 {f1s.max():.4f}), t=0.5 F1 {f1_05:.4f}")


def plot_lambda(y, x, names):
    """Fig. 2: CV F1 (mean ± std over folds) vs lambda for reg. logistic."""
    lambdas = np.logspace(-8, -1, 8)
    rows = []
    for lam in lambdas:
        params = {**BEST, "lambda_": lam}
        m, s, t = cross_validate(y, x, names, "reg_logistic_regression", **params)
        rows.append((lam, m, s, t))
        print(f"  lambda {lam:.0e}: F1 {m:.4f} ± {s:.4f}")
    save_csv("lambda.csv", ["lambda", "f1_mean", "f1_std", "threshold"], rows)
    _, means, stds, _ = map(np.array, zip(*rows))

    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    ax.errorbar(lambdas, means, yerr=stds, color=BLUE, marker="o", ms=3, lw=1.2, capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("Regularization strength $\\lambda$")
    ax.set_ylabel("5-fold CV F1")
    save_fig(fig, "f1_vs_lambda.pdf")


def plot_baselines(y, x, names):
    """Fig. 3: CV F1 of the 6 baseline methods."""
    rows = []
    for label, method, hp in BASELINES:
        try:
            m, s, t = cross_validate(y, x, names, method, **hp)
        except np.linalg.LinAlgError:
            m, s, t = np.nan, np.nan, np.nan
        rows.append((label, m, s, t))
        print(f"  {label:14s} F1 {m:.4f} ± {s:.4f} (t {t:.2f})")
    save_csv("baselines.csv", ["method", "f1_mean", "f1_std", "threshold"], rows)
    labels = [r[0] for r in rows]
    means = np.array([r[1] for r in rows])
    stds = np.array([r[2] for r in rows])
    colors = [BLUE if "logistic" in l.lower() else GREY for l in labels]

    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    pos = np.arange(len(labels))
    ax.bar(pos, means, yerr=stds, color=colors, capsize=2, width=0.65)
    for p, m, s in zip(pos, means, stds):
        if np.isfinite(m):
            ax.text(p, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("5-fold CV F1")
    ax.set_ylim(0, np.nanmax(means + stds) * 1.15)
    save_fig(fig, "baselines.pdf")


def logistic_loss(y, tx, w):
    """Mean logistic loss, computed stably as log(1 + e^z) - y z."""
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def plot_convergence(y, x, names, max_iters=500):
    """Fig. 4: training loss per iteration of logistic GD for several step sizes."""
    params = fit_preprocessing(x, names, None)
    tx = transform(x, names, params)
    gammas = [0.1, 0.5, 1.0]
    losses = np.zeros((max_iters + 1, len(gammas)))
    for j, gamma in enumerate(gammas):
        # Same update as logistic_regression in implementations.py
        w = np.zeros(tx.shape[1])
        losses[0, j] = logistic_loss(y, tx, w)
        for it in range(1, max_iters + 1):
            pred = 0.5 * (1.0 + np.tanh(0.5 * (tx @ w)))
            w = w - gamma * tx.T @ (pred - y) / len(y)
            losses[it, j] = logistic_loss(y, tx, w)
        print(f"  gamma {gamma}: final loss {losses[-1, j]:.4f}")
    iters = np.arange(max_iters + 1)
    save_csv(
        "convergence.csv",
        ["iteration"] + [f"loss_gamma_{g}" for g in gammas],
        np.column_stack([iters, losses]).tolist(),
    )

    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    for j, gamma in enumerate(gammas):
        ax.plot(iters, losses[:, j], color=GAMMA_COLORS[j], lw=1.3, label=f"$\\gamma$ = {gamma}")
    ax.set_xlabel("Gradient descent iteration")
    ax.set_ylabel("Training logistic loss")
    ax.legend(frameon=False)
    save_fig(fig, "convergence.pdf")


def plot_ablation(y, x, names):
    """Fig. 5: CV F1 of reg. logistic with one preprocessing step removed at a time."""
    rows = []
    for label, config in ABLATION:
        m, s, t = cross_validate(y, x, names, "reg_logistic_regression", config=config, **BEST)
        rows.append((label, m, s, t))
        print(f"  {label:14s} F1 {m:.4f} ± {s:.4f}")
    save_csv("ablation.csv", ["config", "f1_mean", "f1_std", "threshold"], rows)
    labels = [r[0] for r in rows]
    means = np.array([r[1] for r in rows])
    stds = np.array([r[2] for r in rows])

    # Dots with error bars: the axis is zoomed, so bars would exaggerate differences
    fig, ax = plt.subplots(figsize=(WIDTH, 1.8))
    pos = np.arange(len(labels))
    for p, m, sd, c in zip(pos, means, stds, [BLUE] + [GREY] * (len(labels) - 1)):
        ax.errorbar(m, p, xerr=sd, fmt="o", color=c, ms=4, capsize=2, lw=1.2)
        ax.text(m, p - 0.18, f"{m:.3f}", ha="center", va="bottom", fontsize=6)
    ax.axvline(means[0], color=BLUE, ls=":", lw=0.8)
    ax.set_yticks(pos)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.6)
    ax.set_xlabel("5-fold CV F1 (mean $\\pm$ std over folds)")
    save_fig(fig, "ablation.pdf")


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    x_train, _, y_train, _, _ = load_data_cached("dataset")
    names = load_feature_names("dataset")
    y = to_01(y_train)
    if SMALL:
        x_train, y = x_train[::50], y[::50]

    print("1. F1 vs threshold")
    plot_threshold(y, x_train, names)
    print("2. F1 vs lambda")
    plot_lambda(y, x_train, names)
    print("3. Baselines")
    plot_baselines(y, x_train, names)
    print("4. Convergence")
    plot_convergence(y, x_train, names)
    print("5. Preprocessing ablation")
    plot_ablation(y, x_train, names)
