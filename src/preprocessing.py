"""
Each function does one transformation step on the dataset and returns transformed data
instead of mutating the input data. This allows for easy testing and debugging of each step in the pipeline.

Pipeline order (see `run_preprocessing_pipeline` at the bottom):
1. Clean undocumented category codes (EDUCATION, MARRIAGE)
2. Drop non-feature columns (ID)
3. Log-transform skewed numeric features 
4. Stratified train/val/test split
5. Scale features (fit on train only, applied to val/test)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "default.payment.next.month"

# columns identified in EDA as heavily right-skewed
SKEWED_COLS = [
    "LIMIT_BAL", 
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

def clean_undocumented_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fold undocumented EDUCATION/MARRIAGE codes into the existing "others"
    category, according to EDA findings.

    EDUCATION: documented codes are 1=grad school, 2=university,
    3=high school, 4=others. Codes 0, 5, 6 are undocumented and rare
    (~1.15% of rows combined) -> folded into 4.

    MARRIAGE: documented codes are 1=married, 2=single, 3=others.
    Code 0 is undocumented and rare (~0.18% of rows) -> folded into 3.

    Note: this is intentionally NOT applied to the PAY_* columns —
    their undocumented codes (0, -2) make up the majority of the data
    in those columns and from EDA we found out they carry real signal, not noise,
    so they're left untouched.
    """

    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    return df

def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the ID column - a row identifier, not a predictive feature."""
    return df.drop(columns=["ID"])

def log_transform_skewed(df: pd.DataFrame, cols: list[str] = SKEWED_COLS) -> pd.DataFrame:
    """
    Apply log1p(log(1+x)) to skewed numeric columns to compress their
    long right tails.

    log1p (rather than plain log) is used because several of these
    columns (BILL_AMT*) can be legitimately negative or zero, and
    log(0) or log(negative) is undefined. log1p handles zero safely,
    but a plain log1p still can't handle negative values — so for
    BILL_AMT* columns specifically, we shift by the column's minimum
    first so every value is >= 0 before transforming. This preserves
    relative ordering and just moves the whole column into a
    transformable range.
    """
    df = df.copy()
    for col in cols:
        if df[col].min() < 0:
            shift = -df[col].min()
            df[col] = np.log1p(df[col] + shift)
        else:
            df[col] = np.log1p(df[col])
    return df

@dataclass 
class DataSplits:
    """Container for train/val/test splits, kept together so they can't be
    accidentally mismatched (e.g wrong X paired with wrong y)
    """
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

def split_data(
        df: pd.DataFrame, 
        target_col: str =  TARGET_COL, 
        val_size: float = 0.15, 
        test_size: float = 0.15, 
        random_state: int = 42
    ) -> DataSplits:
    """
    Stratified train/val/test split

    Stratification (stratify=y) is essential here, not optional: it forces
    each split to preserve the same ~78/22 class ratio found in EDA. Without it,
    a random split could by chance produce a test set with a meaningful different 
    imbalance than train, which would make evaluation results misleading - you'd be 
    testing on different problem than you trained on.

    val_size and test_size are both fractions of full dataset
    (e.g 0.15 + 0.15 = 30% held out, 70% train), computed via two sequential splits 
    since train_test_split only splits once at a time.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Then split the remainer into train/val. val_size needs to be 
    # rescaled relative to X_temp (which is already smaller than the full dataset by test_size)
    val_fraction_of_temp = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, stratify=y_temp, random_state=random_state
    )

    return DataSplits(X_train, X_val, X_test, y_train, y_val, y_test)

def scale_features(splits: DataSplits) -> tuple[DataSplits, StandardScaler]:
    """
    Standardize features (zero mean, unit variance) - required fo the neural network
    & for logistic regression, since both are sensitive to features being in dfferent scales.

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
        splits.y_train, splits.y_val, splits.y_test
    )
    return scaled_splits, scaler

def run_preprocessing_pipeline(df: pd.DataFrame) -> tuple[DataSplits, StandardScaler]:
    """
    Run the full preprocessing pipeline end to end, in order:
    clean -> drop ID -> log-transform -> split -> scale

    Returns both the scaled splits (ready for logistic regression/NN) and the fitted
    scaler (needed later to tranform new/incoming data in the Streamlit app the samem 
    way training data was transformed).
    """
    df = clean_undocumented_codes(df)
    df = drop_id_column(df)
    df = log_transform_skewed(df)
    
    splits = split_data(df)
    scaled_splits, scaler = scale_features(splits)

    return scaled_splits, scaler

if __name__ == "__main__":
    from data_loader import load_raw_data

    raw_df = load_raw_data()
    splits, scaler = run_preprocessing_pipeline(raw_df)

    print(f"Train: {splits.X_train.shape}, Val: {splits.X_val.shape}, Test: {splits.X_test.shape}")
    print(f"\nTrain target distribution:")
    print(splits.y_train.value_counts(normalize=True))
    print(f"\nVal target distribution:")
    print(splits.y_val.value_counts(normalize=True))
    print(f"\nTest target distribution:")
    print(splits.y_test.value_counts(normalize=True))
