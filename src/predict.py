"""
Prediction utilities — takes raw, human-entered feature values and runs
them through the exact same transformation pipeline used during
training, then through the trained model.

Kept separate from the Streamlit app (app/app.py) so the actual
prediction logic can be tested independently of any UI code, and so
the same logic could be reused elsewhere (an API, a batch script, etc.)
without dragging Streamlit along as a dependency.
"""

from pathlib import Path

import joblib
import pandas as pd
from tensorflow import keras

import tensorflow as tf
tf.config.set_visible_devices([], "GPU") # Disable GPU usage for Streamlit app, to avoid the deadlock issue with TensorFlow-Metal + Streamlit on macOS

from preprocessing import apply_log_transform, clean_undocumented_codes

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

def load_artifacts():
    """
    Load the trained model and everything needed to preprocess new
    input identically to how training data was preprocessed.

    Returns a dict rather than a tuple so the Streamlit app can load
    this once (e.g. cached with st.cache_resource) and pass it around
    by name rather than positionally, which is less error-prone as
    the number of artifacts grows.
    """
    model = keras.models.load_model(MODELS_DIR / "final_nn_model.keras")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    log_shifts = joblib.load(MODELS_DIR / "log_shifts.pkl")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")

    # Force the model's prediction function to build NOW, in this
    # thread, right after loading — rather than lazily on the first
    # real prediction call. Keras builds this function on first use;
    # if that first use happens in a different thread than the one
    # that loaded the model (which can happen with Streamlit, since it
    # may rerun the script in a new thread while a cached model object
    # persists across reruns), it can hang or behave inconsistently.
    # A one-time dummy prediction here sidesteps that entirely, and as
    # a bonus, absorbs TensorFlow's "first call is slow" cost at
    # startup instead of on the user's first click.
    dummy_input = pd.DataFrame([[0.0] * len(feature_columns)], columns=feature_columns)
    model.predict(dummy_input, verbose=0)

    return {
        "model": model,
        "scaler": scaler,
        "log_shifts": log_shifts,
        "feature_columns": feature_columns,
    }


# def load_artifacts():
#     """
#     Load the trained model and everything needed to preprocess new
#     input identically to how training data was preprocessed.

#     Returns a dict rather than a tuple so the Streamlit app can load
#     this once (e.g. cached with st.cache_resource) and pass it around
#     by name rather than positionally, which is less error-prone as
#     the number of artifacts grows.
#     """
#     model = keras.models.load_model(MODELS_DIR / "final_nn_model.keras")
#     scaler = joblib.load(MODELS_DIR / "scaler.pkl")
#     log_shifts = joblib.load(MODELS_DIR / "log_shifts.pkl")
#     feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")

#     return {
#         "model": model,
#         "scaler": scaler,
#         "log_shifts": log_shifts,
#         "feature_columns": feature_columns,
#     }


def predict_default(raw_input: dict, artifacts: dict) -> dict:
    """
    Run one client's raw feature values through the full pipeline and
    return a default probability + label.

    Parameters
    ----------
    raw_input : dict
        Feature name -> raw value, e.g. {"LIMIT_BAL": 50000, "AGE": 34, ...}.
        Must contain every column in artifacts["feature_columns"].
    artifacts : dict
        The dict returned by load_artifacts().

    Returns
    -------
    dict with "probability" (float, 0-1) and "prediction" (0 or 1).

    This function deliberately mirrors run_preprocessing_pipeline's
    later steps (clean -> log-transform -> scale) exactly, MINUS the
    split step (there's nothing to split for a single new row), and
    using the SAVED log_shifts/scaler rather than fitting new ones —
    fitting on a single input row would be meaningless (a std of 0
    from one data point) and is exactly the bug this module exists to
    avoid repeating.
    """
    df = pd.DataFrame([raw_input])[artifacts["feature_columns"]]

    df = clean_undocumented_codes(df)
    df = apply_log_transform(df, artifacts["log_shifts"])

    X_scaled = pd.DataFrame(
        artifacts["scaler"].transform(df),
        columns=df.columns,
    )

    probability = float(artifacts["model"].predict(X_scaled, verbose=0).ravel()[0])
    prediction = int(probability >= 0.5)

    return {"probability": probability, "prediction": prediction}


# === Example usage (for testing) ==========================================
#from src.predict import load_artifacts, predict_default

artifacts = load_artifacts()

sample_input = {
    'LIMIT_BAL': 50000, 'SEX': 1, 'EDUCATION': 2, 'MARRIAGE': 1, 'AGE': 35,
    'PAY_0': 2, 'PAY_2': 2, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
    'BILL_AMT1': 20000, 'BILL_AMT2': 19000, 'BILL_AMT3': 18000,
    'BILL_AMT4': 17000, 'BILL_AMT5': 16000, 'BILL_AMT6': 15000,
    'PAY_AMT1': 1000, 'PAY_AMT2': 1000, 'PAY_AMT3': 1000,
    'PAY_AMT4': 1000, 'PAY_AMT5': 1000, 'PAY_AMT6': 1000,
}

result = predict_default(sample_input, artifacts)
print(result)