"""
Preprocessing utilities for the credit card default classification project.

Each function does ONE well-defined transformation, and none of them
mutate the input DataFrame in place — they return a new object. This
keeps the pipeline composable and lets us inspect intermediate steps
(e.g. compare df before/after cleaning) rather than losing that ability
to an in-place mutation.

Pipeline order (see `run_preprocessing_pipeline` at the bottom):
1. Clean undocumented category codes (EDUCATION, MARRIAGE)
2. Drop non-feature columns (ID)
3. Stratified train/val/test split
4. Log-transform skewed numeric features (shift values computed from
   TRAIN ONLY, then reused on val/test — same leakage-avoidance
   principle as the scaler, see log_transform_skewed / scale_features)
5. Scale features (fit on train only, applied to val/test)

Note the split happens BEFORE the log-transform and scaling steps —
this matters. Both the log-transform shift and the scaler's mean/std
must be learned from training data only, then applied unchanged to
val/test (and later, to any new input the Streamlit app receives).
Computing them from the full dataset before splitting would leak
information about val/test into those transformation parameters.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "default.payment.next.month"

# Columns identified in EDA as heavily right-skewed.
SKEWED_COLS = [
    "LIMIT_BAL",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def clean_undocumented_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fold undocumented EDUCATION/MARRIAGE codes into the existing
    "others" category, per the decision made during EDA.

    EDUCATION: documented codes are 1=grad school, 2=university,
    3=high school, 4=others. Codes 0, 5, 6 are undocumented and rare
    (~1.15% of rows combined) -> folded into 4.

    MARRIAGE: documented codes are 1=married, 2=single, 3=others.
    Code 0 is undocumented and rare (~0.18% of rows) -> folded into 3.

    Note: this is intentionally NOT applied to the PAY_* columns —
    their undocumented codes (0, -2) make up the majority of the data
    in those columns and were judged to carry real signal, not noise,
    so they're left untouched.
    """
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the ID column — a row identifier, not a predictive feature."""
    return df.drop(columns=["ID"])


def compute_log_shifts(df: pd.DataFrame, cols: list[str] = SKEWED_COLS) -> dict[str, float]:
    """
    Compute the shift needed for each column so every value is >= 0
    before applying log1p (log1p handles 0 fine, but not negatives).

    IMPORTANT: this must be called on TRAINING data only, then the
    resulting shifts dict is reused (via apply_log_transform) on
    val/test and on any new input at inference time. If you instead
    computed a fresh shift from whatever data you're currently
    transforming, a single-row input at inference (e.g. from the
    Streamlit app) would get a shift computed from that one row's own
    value — a different, meaningless transformation compared to what
    the model was trained on. This is exactly the same leakage/
    consistency principle as fitting the StandardScaler on train only.

    Only columns with a negative minimum need a shift; columns that
    are already >= 0 get a shift of 0 (i.e. untouched by this step).
    """
    shifts = {}
    for col in cols:
        col_min = df[col].min()
        shifts[col] = -col_min if col_min < 0 else 0.0
    return shifts


def apply_log_transform(
    df: pd.DataFrame, shifts: dict[str, float], cols: list[str] = SKEWED_COLS
) -> pd.DataFrame:
    """
    Apply log1p (log(1+x)) to skewed numeric columns, using
    PRE-COMPUTED shift values (from compute_log_shifts, on training
    data) rather than recomputing them from whatever df is passed in.

    This is what makes the transform reproducible: the same shifts
    dict is applied identically to training data, validation data,
    test data, and — later — single-row input from the Streamlit app.

    Edge case handled here: since shifts come from TRAINING data only,
    a value in val/test (or a new user input in the app) could in
    principle be more extreme than anything seen in training, meaning
    `value + shift` is still negative — log1p of a negative number is
    undefined and would silently produce NaN. We clip at 0 as a floor:
    anything more extreme than the most extreme training value gets
    treated as equally extreme (same log1p(0) floor), rather than
    breaking. This is a reasonable, defensible choice for a small
    number of extreme outliers; it does NOT affect any value that
    falls within the range the model was actually trained on.
    """
    df = df.copy()
    for col in cols:
        shifted = df[col] + shifts[col]
        df[col] = np.log1p(shifted.clip(lower=0))
    return df


@dataclass
class DataSplits:
    """Container for train/val/test splits, kept together so they can't
    be accidentally mismatched (e.g. wrong X paired with wrong y)."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DataSplits:
    """
    Stratified train/val/test split.

    Stratification (stratify=y) is essential here, not optional: it
    forces each split to preserve the same ~78/22 class ratio found in
    EDA. Without it, a random split could by chance produce a test set
    with a meaningfully different imbalance than train, which would
    make evaluation results misleading — you'd be testing on a
    different problem than you trained on.

    val_size and test_size are both fractions of the FULL dataset
    (e.g. 0.15 + 0.15 = 30% held out, 70% train), computed via two
    sequential splits since train_test_split only splits once at a time.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split off the test set.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Then split the remainder into train/val. val_size needs to be
    # rescaled relative to X_temp (which is already smaller than the
    # full dataset by test_size).
    val_fraction_of_temp = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, stratify=y_temp, random_state=random_state
    )

    return DataSplits(X_train, X_val, X_test, y_train, y_val, y_test)


def scale_features(splits: DataSplits) -> tuple[DataSplits, StandardScaler]:
    """
    Standardize features (zero mean, unit variance) — required for the
    neural network and for logistic regression, since both are
    sensitive to features being on different scales.

    Critical detail: the scaler is FIT on X_train only, then applied
    (transform, not fit_transform) to X_val and X_test. Fitting on the
    full dataset before splitting — or fitting separately on each
    split — would leak information from val/test into the scaling
    parameters (mean/std), which is a form of data leakage: the model
    would be evaluated under conditions that subtly "know" about the
    val/test distribution ahead of time, inflating apparent performance.
    """
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(splits.X_train),
        columns=splits.X_train.columns,
        index=splits.X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(splits.X_val),
        columns=splits.X_val.columns,
        index=splits.X_val.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(splits.X_test),
        columns=splits.X_test.columns,
        index=splits.X_test.index,
    )

    scaled_splits = DataSplits(
        X_train_scaled, X_val_scaled, X_test_scaled,
        splits.y_train, splits.y_val, splits.y_test,
    )
    return scaled_splits, scaler


@dataclass
class PreprocessingArtifacts:
    """Everything needed to transform NEW, raw input the exact same way
    training data was transformed — required for consistent inference
    later in the Streamlit app. Bundled together for the same reason
    DataSplits bundles X/y: these three things must always travel
    together, and saving/loading them separately risks a mismatch."""
    scaler: StandardScaler
    log_shifts: dict[str, float]
    feature_columns: list[str]


def run_preprocessing_pipeline(df: pd.DataFrame) -> tuple[DataSplits, PreprocessingArtifacts]:
    """
    Run the full preprocessing pipeline end to end, in order:
    clean -> drop ID -> split -> log-transform (shifts from train only)
    -> scale (fit on train only).

    Returns the scaled splits (ready for logistic regression / NN) and
    a PreprocessingArtifacts bundle (scaler + log shifts + feature
    column order) — everything needed to transform new, raw input
    identically at inference time in the Streamlit app.
    """
    df = clean_undocumented_codes(df)
    df = drop_id_column(df)

    splits = split_data(df)

    # Learn the log-transform shifts from TRAIN ONLY, then apply the
    # same shifts to all three splits — mirrors how the scaler below
    # is fit on train only and applied unchanged to val/test.
    log_shifts = compute_log_shifts(splits.X_train)
    splits = DataSplits(
        X_train=apply_log_transform(splits.X_train, log_shifts),
        X_val=apply_log_transform(splits.X_val, log_shifts),
        X_test=apply_log_transform(splits.X_test, log_shifts),
        y_train=splits.y_train,
        y_val=splits.y_val,
        y_test=splits.y_test,
    )

    scaled_splits, scaler = scale_features(splits)

    artifacts = PreprocessingArtifacts(
        scaler=scaler,
        log_shifts=log_shifts,
        feature_columns=list(scaled_splits.X_train.columns),
    )

    return scaled_splits, artifacts


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw_df = load_raw_data()
    splits, artifacts = run_preprocessing_pipeline(raw_df)

    print(f"Train: {splits.X_train.shape}, Val: {splits.X_val.shape}, Test: {splits.X_test.shape}")
    print("\nTrain target distribution:")
    print(splits.y_train.value_counts(normalize=True))
    print("\nVal target distribution:")
    print(splits.y_val.value_counts(normalize=True))
    print("\nTest target distribution:")
    print(splits.y_test.value_counts(normalize=True))
    