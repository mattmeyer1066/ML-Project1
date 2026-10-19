"""Experiments for the report: baselines, lambda tuning and preprocessing ablation."""

import numpy as np

from cross_validation import cross_validate
from preprocessing import load_data_cached, load_feature_names, to_01

SMALL = True  # True: 1/50 of the data for quick tests. False: final numbers.

x_train, _, y_train, _, _ = load_data_cached("dataset")
names = load_feature_names("dataset")
y = to_01(y_train)
if SMALL:
    x_train, y = x_train[::50], y[::50]


def report(label, method, **kwargs):
    """Run one cross-validation and print the result."""
    try:
        mean_f1, std_f1, t = cross_validate(y, x_train, names, method, **kwargs)
    except np.linalg.LinAlgError:
        print(f"{label:30s} singular matrix")
        return
    print(f"{label:30s} F1 = {mean_f1:.4f} ± {std_f1:.4f}  (threshold {t:.2f})")

"""
print("=== 1. Baselines ===")
report("MSE GD", "mean_squared_error_gd", gamma=0.1, max_iters=300)
report("MSE SGD", "mean_squared_error_sgd", gamma=0.001, max_iters=50000)
report("Least squares", "least_squares")
report("Ridge", "ridge_regression", lambda_=1e-4)
report("Logistic", "logistic_regression", gamma=0.5, max_iters=500)
report("Reg. logistic", "reg_logistic_regression", lambda_=1e-4, gamma=0.5, max_iters=500)
"""
print("\n=== 2. Tuning lambda (reg. logistic) ===")
for lambda_ in [1e-6, 1e-4, 1e-2, 1e-1]:
    report(f"lambda = {lambda_:.0e}", "reg_logistic_regression",
           lambda_=lambda_, gamma=0.5, max_iters=500)

print("\n=== 3. Preprocessing ablation (reg. logistic) ===")
BEST = dict(lambda_=1e-6, gamma=0.5, max_iters=500)  # update with the best lambda from part 2
for label, config in [
    ("Full preprocessing", None),
    ("No missing indicators", {"indicators": False}),
    ("No one-hot encoding", {"one_hot": False}),
    ("No clipping", {"clip_pct": 100}),
    ("Squared features", {"poly": True}),
]:
    report(label, "reg_logistic_regression", config=config, **BEST)
report("Fixed threshold 0.5", "reg_logistic_regression", threshold=0.5, **BEST)

print("\n=== 4. Tuning gamma / max_iters (reg. logistic) ===")
for gamma in [0.1, 0.5, 1.0]:
    for max_iters in [200, 500, 1000]:
        report(f"gamma={gamma}, iters={max_iters}", "reg_logistic_regression",
               lambda_=1e-6, gamma=gamma, max_iters=max_iters)