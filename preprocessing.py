"""Data loading and preprocessing for the BRFSS heart-disease dataset."""

import os
import numpy as np
from helpers import load_csv_data

CACHE = "dataset/cache.npz"


# 1 = yes, 2 = no, 7/9 = don't know / refused
YES_NO = [
    "HLTHPLN1",
    "MEDCOST",
    "BLOODCHO",
    "TOLDHI2",
    "CVDSTRK3",
    "ASTHMA3",
    "CHCSCNCR",
    "CHCOCNCR",
    "CHCCOPD1",
    "HAVARTH3",
    "ADDEPEV2",
    "CHCKIDNY",
    "VETERAN3",
    "INTERNET",
    "QLACTLM2",
    "USEEQUIP",
    "BLIND",
    "DECIDE",
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON",
    "SMOKE100",
    "EXERANY2",
    "FLUSHOT6",
    "PNEUVAC3",
    "HIVTST6",
    "DRNKANY5",
    "BPMEDS",
]

# CDC-computed variables: 9 = missing
MISSING_9 = [
    "_RFHLTH",
    "_HCVU651",
    "_RFHYPE5",
    "_CHOLCHK",
    "_RFCHOL",
    "_LTASTH1",
    "_CASTHM1",
    "_ASTHMS1",
    "_HISPANC",
    "_RACEGR3",
    "_RFBMI5",
    "_CHLDCNT",
    "_EDUCAG",
    "_INCOMG",
    "_SMOKER3",
    "_RFSMOK3",
    "_RFBING5",
    "_RFDRHV5",
    "_FRTLT1",
    "_VEGLT1",
    "_TOTINDA",
    "_PACAT1",
    "_PAINDX1",
    "_PA150R2",
    "_PA300R2",
    "_PA30021",
    "_PASTRNG",
    "_PAREC1",
    "_PASTAE1",
    "_RFSEAT2",
    "_RFSEAT3",
    "_LMTACT1",
    "_LMTWRK1",
    "_LMTSCL1",
    "_AIDTST3",
    "_FLSHOT6",
    "_PNEUMO2",
    "MARITAL",
    "EMPLOY1",
]

# Number of days out of 30: 88 = none, 77/99 = missing
DAYS = ["PHYSHLTH", "MENTHLTH", "POORHLTH"]

# 7/9 = missing, other values are valid categories or ordered levels
MISSING_79 = ["GENHLTH", "USENOW3", "RENTHOM1", "DIABETE3", "BPHIGH4", "CHECKUP1"]

# Single-column special codes: {column: {code: new_value}}
SPECIAL_CODES = {
    "CHECKUP1": {8: 5},  # "never" -> further than "5+ years"
    "CHILDREN": {88: 0, 99: np.nan},
    "_DRNKWEK": {99900: np.nan},
    "DROCDY3_": {900: np.nan},
    "SEX": {2: 0},  # 1 = male, 0 = female
}

# Clean CDC-computed continuous variables (no special codes, NaN = not asked)
CONTINUOUS = [
    "_AGE80",
    "_BMI5",
    "HTM4",
    "WTKG3",
    "FRUTDA1_",
    "VEGEDA1_",
    "_FRUTSUM",
    "_VEGESUM",
    "PA1MIN_",
    "PA1VIGM_",
    "STRFREQ_",
]
# Whitelist: only columns whose codes we have checked (dict.fromkeys removes duplicates)
SELECTED = list(
    dict.fromkeys(
        YES_NO + MISSING_9 + MISSING_79 + DAYS + list(SPECIAL_CODES) + CONTINUOUS
    )
)
# Unordered categorical columns: imputed with the mode, one-hot encoded later (step 5)
NOMINAL = ["MARITAL", "EMPLOY1", "RENTHOM1", "DIABETE3", "BPHIGH4", "_RACEGR3"]

# Heavy-tailed continuous columns whose extreme values are clipped before scaling
CLIP_COLS = CONTINUOUS + ["_DRNKWEK", "DROCDY3_", "CHILDREN"]

# Default preprocessing choices; each key is one axis of the ablation study
DEFAULT_CONFIG = {
    "nan_threshold": 0.8,
    "min_nan_rate": 0.01,
    "indicators": True,
    "one_hot": True,
    "clip_pct": 99,
}


def load_data_cached(data_path="dataset"):
    """Load the CSV files once, then reuse a .npz cache (much faster)."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["x_train"], d["x_test"], d["y_train"], d["train_ids"], d["test_ids"]
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
    np.savez(
        CACHE,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    return x_train, x_test, y_train, train_ids, test_ids


def load_feature_names(data_path="dataset"):
    """Read the header of x_train.csv to get the column names (without 'Id')."""
    with open(os.path.join(data_path, "x_train.csv")) as f:
        return f.readline().strip().split(",")[1:]


def clean_codes(x, names):
    """Replace BRFSS questionnaire codes (don't know, refused, none) with usable values."""
    x = x.copy()
    idx = {n: i for i, n in enumerate(names)}
    for n in YES_NO:
        col = x[:, idx[n]]  # view: modifying col modifies x
        col[np.isin(col, [7, 9])] = np.nan
        col[col == 2] = 0
    for n in MISSING_9:
        col = x[:, idx[n]]
        col[col == 9] = np.nan
    for n in MISSING_79:
        col = x[:, idx[n]]
        col[np.isin(col, [7, 9])] = np.nan
    for n in DAYS:
        col = x[:, idx[n]]
        col[col == 88] = 0
        col[np.isin(col, [77, 99])] = np.nan
    for n, mapping in SPECIAL_CODES.items():
        col = x[:, idx[n]]
        original = col.copy()  # match on original values so remappings don't chain
        for code, new_value in mapping.items():
            col[original == code] = new_value
    return x


def select_columns(x_clean, names, nan_threshold=0.8):
    """Return indices and names of whitelisted columns with NaN rate <= nan_threshold.

    Must be called on data already passed through clean_codes, since cleaning
    turns codes into NaN and therefore changes the NaN rates.
    """
    idx = {n: i for i, n in enumerate(names)}
    keep_idx, keep_names = [], []
    for n in SELECTED:
        j = idx[n]
        if np.isnan(x_clean[:, j]).mean() <= nan_threshold:
            keep_idx.append(j)
            keep_names.append(n)
    return np.array(keep_idx), keep_names


def fit_imputation(x_sel, sel_names, min_nan_rate=0.01):
    """Learn fill values on the training set.

    Numeric/ordered columns are filled with their median, nominal columns with
    their mode. Columns with NaN rate > min_nan_rate get a missing indicator.
    """
    fill = np.empty(x_sel.shape[1])
    for k, n in enumerate(sel_names):
        col = x_sel[:, k]
        col = col[~np.isnan(col)]
        if n in NOMINAL:
            values, counts = np.unique(col, return_counts=True)
            fill[k] = values[np.argmax(counts)]
        else:
            fill[k] = np.median(col)
    nan_rate = np.isnan(x_sel).mean(axis=0)
    indicator_cols = np.where(nan_rate > min_nan_rate)[0]
    # Drop duplicate indicators: columns that are always missing together
    nan_mask = np.isnan(x_sel[:, indicator_cols])
    _, first = np.unique(nan_mask, axis=1, return_index=True)
    indicator_cols = indicator_cols[np.sort(first)]
    return {"fill": fill, "indicator_cols": indicator_cols}


def apply_imputation(x_sel, imp):
    """Build missing indicators (before filling), then replace NaN with learned values.

    Returns the filled matrix and the indicator matrix separately, so the
    indicators can be switched on/off for the ablation study.
    """
    indicators = np.isnan(x_sel[:, imp["indicator_cols"]]).astype(float)
    x_filled = np.where(np.isnan(x_sel), imp["fill"], x_sel)
    return x_filled, indicators


def fit_one_hot(x_filled, sel_names):
    """Learn the categories of each nominal column on the training set.

    The first (smallest) category of each column is dropped and serves as the
    reference level, to avoid perfect collinearity with the bias column.
    """
    cols, categories = [], []
    for k, n in enumerate(sel_names):
        if n in NOMINAL:
            cats = np.unique(x_filled[:, k])
            cols.append(k)
            categories.append(cats[1:])  # drop the reference category
    return {"cols": cols, "categories": categories}


def apply_one_hot(x_filled, oh):
    """Encode each nominal column as 0/1 columns, one per non-reference category."""
    blocks = []
    for k, cats in zip(oh["cols"], oh["categories"]):
        blocks.append((x_filled[:, [k]] == cats).astype(float))
    if not blocks:
        return np.empty((x_filled.shape[0], 0))
    return np.hstack(blocks)


def fit_scaling(x_filled, sel_names, clip_pct=99, nominal=NOMINAL):
    """Learn clipping bounds and standardization statistics on the training set.

    Nominal columns are skipped (they are one-hot encoded instead). Columns in
    CLIP_COLS are clipped to the [100 - clip_pct, clip_pct] percentiles before
    computing mean and std, so that a few extreme answers do not dominate.
    """
    num_cols = [k for k, n in enumerate(sel_names) if n not in nominal]
    x_num = x_filled[:, num_cols]
    low = np.full(len(num_cols), -np.inf)
    high = np.full(len(num_cols), np.inf)
    for i, k in enumerate(num_cols):
        if sel_names[k] in CLIP_COLS:
            low[i], high[i] = np.percentile(x_num[:, i], [100 - clip_pct, clip_pct])
    x_num = np.clip(x_num, low, high)
    mean = x_num.mean(axis=0)
    std = x_num.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero for constant columns
    return {"num_cols": num_cols, "low": low, "high": high, "mean": mean, "std": std}


def apply_scaling(x_filled, sc):
    """Clip and standardize the numeric columns using the training statistics."""
    x_num = np.clip(x_filled[:, sc["num_cols"]], sc["low"], sc["high"])
    return (x_num - sc["mean"]) / sc["std"]


def fit_preprocessing(x_train, names, config=None):
    """Learn every preprocessing parameter on the training set only.

    config overrides entries of DEFAULT_CONFIG, e.g. {"indicators": False}.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    x_clean = clean_codes(x_train, names)
    keep_idx, keep_names = select_columns(x_clean, names, cfg["nan_threshold"])
    x_sel = x_clean[:, keep_idx]
    imp = fit_imputation(x_sel, keep_names, cfg["min_nan_rate"])
    x_filled, _ = apply_imputation(x_sel, imp)
    nominal = NOMINAL if cfg["one_hot"] else []
    oh = fit_one_hot(x_filled, keep_names) if cfg["one_hot"] else None
    sc = fit_scaling(x_filled, keep_names, cfg["clip_pct"], nominal)
    return {
        "cfg": cfg,
        "keep_idx": keep_idx,
        "keep_names": keep_names,
        "imp": imp,
        "oh": oh,
        "sc": sc,
    }


def transform(x, names, params):
    """Apply the learned preprocessing to any data (train, validation or test).

    Output columns: [bias | standardized numeric | missing indicators | one-hot].
    """
    x_clean = clean_codes(x, names)
    x_sel = x_clean[:, params["keep_idx"]]
    x_filled, indicators = apply_imputation(x_sel, params["imp"])
    blocks = [np.ones((x.shape[0], 1)), apply_scaling(x_filled, params["sc"])]
    if params["cfg"]["indicators"]:
        blocks.append(indicators)
    if params["cfg"]["one_hot"]:
        blocks.append(apply_one_hot(x_filled, params["oh"]))
    return np.hstack(blocks)


def to_01(y):
    """Convert labels from {-1, 1} (dataset format) to {0, 1} (logistic regression)."""
    return (y == 1).astype(float)


def to_pm1(y01):
    """Convert {0, 1} predictions back to {-1, 1} for create_csv_submission."""
    return np.where(y01 == 1, 1, -1)
