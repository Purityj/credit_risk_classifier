"""
Data loading utilities for the credit card default classification project.

Loading strategy (in order of preference):
1. kagglehub - downloads the dataset on the fly, no account needded for this public dataset.
2. Local file in data/raw/ - used as fallback if kagglehub is unavailable.
"""

from pathlib import Path 
import pandas as pd

KAGGLE_DATASET_SLUG = "uciml/default-of-credit-card-clients-dataset"
LOCAL_RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "UCI_Credit_Card.csv"

def _load_via_kagglehub() -> pd.DataFrame:
    """Attempt to download and load the dataset via kagglehub."""
    import kagglehub

    download_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    csv_files = list(download_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"kagglehub downloaded the dataset to {download_path}, "
            "but no CSV file was found inside it."
        )
    
    df = pd.read_csv(csv_files[0])
    
    # Persist a copy into data/raw/ so the repo's expected location is 
    # populated too, not just kagglehb's own cache
    LOCAL_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOCAL_RAW_PATH, index=False)

    return df

def _load_via_local_file() -> pd.DataFrame:
    """Load the dataset from data/raw/, if it's already been placed there."""
    if not LOCAL_RAW_PATH.exists():
        raise FileNotFoundError(
            f"No local copy found at {LOCAL_RAW_PATH}. "
            "Either install kagglehub (`pip install kagglehub`) so the "
            "dataset can be fetched automatically, manually download "
            "the CSV from kaggle and place it at that path."
        )
    return pd.read_csv(LOCAL_RAW_PATH)

def load_raw_data(prefer: str = "local") -> pd.DataFrame:
    """
    Load the raw credit card default dataset.

    Parameters
    ----------
    prefer : {"local", "kagglehub"}
        Which source to try first. Defaults to "local" - if the CSV is 
        already for data/raw/, there's no readon to hit the network. 
        Fallback to kagglehub automatically if the preferred source fails,
        so a fresh clone with no local data still works.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified dataset (30000 rows x 25 columns, including 
        the ID column and target column).
    """
    loaders = {
        "kagglehub": _load_via_kagglehub,
        "local": _load_via_local_file
    }

    if prefer not in loaders:
        raise ValueError(f"prefer must be one of {list(loaders)}, got {prefer!r}")

    order = [prefer] + [name for name in loaders if name != prefer]

    last_error = None
    for name in order:
        try:
            df = loaders[name]()
            print(f"Loaded data via {name} ({df.shape[0]} rows, {df.shape[1]} columns).")
            return df
        except Exception as e:
            print(f"[data_loader] {name} failed: {e}")
            last_error = e

    raise RuntimeError(
        "All data loading strategies failed"
    ) from last_error
if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
    