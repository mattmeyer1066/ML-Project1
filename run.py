"""Reproduce our best submission: train on all the training data, predict the test set."""

from cross_validation import train_model, predict_scores
from helpers import create_csv_submission
from preprocessing import (
    load_data_cached,
    load_feature_names,
    fit_preprocessing,
    transform,
    to_01,
    to_pm1,
)

# Best values found with experiments.py (cross-validation)
METHOD = "reg_logistic_regression"
HYPERPARAMS = dict(lambda_=1e-6, gamma=1, max_iters=1000)
THRESHOLD = 0.2


def main():
    x_train, x_test, y_train, _, test_ids = load_data_cached("dataset")
    names = load_feature_names("dataset")
    y = to_01(y_train)

    params = fit_preprocessing(x_train, names)
    w = train_model(METHOD, y, transform(x_train, names, params), **HYPERPARAMS)

    scores = predict_scores(METHOD, transform(x_test, names, params), w)
    y_pred = to_pm1((scores >= THRESHOLD).astype(int))
    create_csv_submission(test_ids, y_pred, "submission.csv")
    print("Saved submission.csv")


if __name__ == "__main__":
    main()