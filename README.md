# ML Project 1 – Predicting coronary heart disease (MICHD)

CS-433 Machine Learning, EPFL. Team: Matteo Meyer, Ulysse du Sordet, Loann Colli.

We predict whether a respondent of the 2015 BRFSS survey has coronary heart disease
or has had a heart attack (MICHD) from 321 health and lifestyle features, using only
linear models implemented from scratch in NumPy. Since only 8.8% of the training
respondents are positive, models are evaluated with the F1 score of the positive class.

**Final model:** L2-regularized logistic regression (`lambda_=1e-6`, `gamma=1`,
`max_iters=1000`), full preprocessing, decision threshold 0.20.
5-fold cross-validation F1: **0.422 ± 0.002**. AIcrowd test F1: **0.434** (accuracy 0.875).

## Requirements

- Python 3
- NumPy (all models and preprocessing)
- matplotlib (only for `plots.py`)

No other library is used.

## Data

The dataset is not included in this repository (GitHub file-size limit).
Download it from the AIcrowd challenge page and place the files, without renaming them, in:

```
dataset/
    x_train.csv
    y_train.csv
    x_test.csv
```

The first run parses the CSV files and saves a cache `dataset/cache.npz`
(ignored by git); later runs load this cache, which is much faster.

## Reproducing our submission

```bash
python run.py
```

This trains the final model on the whole training set and writes `submission.csv`,
the exact file submitted to AIcrowd. Training is full-batch gradient descent with no
random component, so the output is deterministic (about 1 minute on a laptop once
the cache exists).

## Repository structure

| File | Content |
|---|---|
| `implementations.py` | The 6 required methods: `mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression`. |
| `preprocessing.py` | Data loading (with cache), cleaning of the BRFSS codes, column selection, imputation, missing indicators, one-hot encoding, clipping, standardization, optional squared features. |
| `cross_validation.py` | Training/prediction wrapper for the 6 methods, F1 score, threshold search and k-fold cross-validation. |
| `run.py` | Reproduces our final submission (`submission.csv`). |
| `experiments.py` | Cross-validation experiments of the report: baselines, tuning of `lambda_`, `gamma` and `max_iters`, preprocessing ablation. Results are printed. |
| `plots.py` | Generates the report figures in `results/figures/` and their numbers in `results/*.csv`. |
| `helpers.py` | Course helpers: `load_csv_data`, `create_csv_submission`. |
| `results/` | Figures and CSV results used in the report. |
| `test_project1_public.py`, `conftest.py` | Public grading tests of the course. |

## Pipeline

### Preprocessing (`preprocessing.py`)

All parameters are learned on the training data only (`fit_preprocessing`) and then
applied unchanged to validation and test data (`transform`).

1. **Code cleaning** (`clean_codes`): BRFSS special codes are mapped to their meaning,
   e.g. 7/9 ("don't know"/"refused") → missing, 2 ("no") → 0, 88 ("none") → 0,
   77/99 → missing. Only a whitelist of 91 features whose codes were checked in the
   codebook is kept (`SELECTED`).
2. **Column selection**: features with more than 80% missing values are dropped.
3. **Imputation**: median for numeric/ordinal features, mode for nominal ones.
4. **Missing indicators**: a 0/1 column for every feature with more than 1% missing
   values (duplicate indicators removed).
5. **One-hot encoding** of the 6 nominal features (reference category dropped).
6. **Clipping** of heavy-tailed continuous features to their 1st–99th percentiles,
   then **standardization** of all numeric features.
7. A **bias column** is added. Final design matrix: 158 columns.

Each step can be switched on/off through a config dictionary (see `DEFAULT_CONFIG`),
which is used for the ablation study.

### Cross-validation (`cross_validation.py`)

- 5 folds, shuffled with a fixed seed (42).
- In each fold, the preprocessing is refitted on the training folds only (no leakage).
- Labels are mapped from {-1, 1} to {0, 1}. Logistic models output probabilities,
  MSE-based models a real score.
- The decision threshold (searched between 0.05 and 0.5) is chosen to maximize F1
  on the pooled out-of-fold predictions; we report the mean and standard deviation
  of F1 over the 5 folds.

## Running the experiments and figures

```bash
python experiments.py   # prints the cross-validation results of the report
python plots.py         # writes results/figures/*.pdf and results/*.csv
```

Both scripts run many cross-validations on the full dataset and can take a while.
Set `SMALL = True` at the top of either file to run on 1/50 of the data for a quick check.

## Tests

```bash
pytest test_project1_public.py --github_link <path or URL of this repository>
```
