"""
Streamlit app for the Credit Card Default Risk Classifier.

Collects a client's raw feature values through a form, runs them
through the same preprocessing + trained neural network used
throughout the project (via src/predict.py), and displays the
predicted default probability.
"""

import sys
from pathlib import Path

# Make src/ importable from here, same pattern used in every notebook
# in this project (sys.path.append("..") + import from src.<module>).
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
from predict import load_artifacts, predict_default

st.set_page_config(page_title="Credit Default Risk Classifier", page_icon="💳", layout="centered")


@st.cache_resource
def get_artifacts():
    """
    Load the model + preprocessing artifacts once per app session, not
    on every interaction. Without this, every button click would
    reload the ~model file from disk and re-fit nothing (it's just
    loading), but still waste time doing so unnecessarily.
    """
    return load_artifacts()


artifacts = get_artifacts()

st.title("💳 Credit Card Default Risk Classifier")
st.write(
    "Estimates the probability that a credit card client will default on "
    "their payment next month, based on the UCI Default of Credit Card "
    "Clients dataset. Built with a TensorFlow neural network; see the "
    "GitHub repo for the full EDA, baseline model comparison, and "
    "hyperparameter tuning behind this app."
)

st.header("Client Information")

# --- Demographics ---
st.subheader("Demographics")
col1, col2 = st.columns(2)
with col1:
    limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=10000, max_value=1000000, value=50000, step=10000)
    age = st.number_input("Age", min_value=21, max_value=79, value=35)
with col2:
    sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    education = st.selectbox(
        "Education", options=[1, 2, 3, 4],
        format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 4: "Other"}[x],
    )
marriage = st.selectbox(
    "Marital Status", options=[1, 2, 3],
    format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x],
)

# --- Repayment status ---
st.subheader("Repayment Status (last 6 months)")
st.caption(
    "-2 = no consumption, -1 = paid duly, 0 = paid with revolving balance, "
    "1-8 = payment delay of that many months"
)
pay_status_options = list(range(-2, 9))  # -2 through 8, matches real data range
pay_cols = st.columns(3)
pay_0 = pay_cols[0].selectbox("Most recent month (PAY_0)", options=pay_status_options, index=2)
pay_2 = pay_cols[1].selectbox("2 months ago (PAY_2)", options=pay_status_options, index=2)
pay_3 = pay_cols[2].selectbox("3 months ago (PAY_3)", options=pay_status_options, index=2)
pay_cols2 = st.columns(3)
pay_4 = pay_cols2[0].selectbox("4 months ago (PAY_4)", options=pay_status_options, index=2)
pay_5 = pay_cols2[1].selectbox("5 months ago (PAY_5)", options=pay_status_options, index=2)
pay_6 = pay_cols2[2].selectbox("6 months ago (PAY_6)", options=pay_status_options, index=2)

# --- Bill and payment amounts ---
st.subheader("Bill Amounts (last 6 months)")
bill_cols = st.columns(3)
bill_amt1 = bill_cols[0].number_input("Bill 1 (most recent)", value=20000, step=1000)
bill_amt2 = bill_cols[1].number_input("Bill 2", value=19000, step=1000)
bill_amt3 = bill_cols[2].number_input("Bill 3", value=18000, step=1000)
bill_cols2 = st.columns(3)
bill_amt4 = bill_cols2[0].number_input("Bill 4", value=17000, step=1000)
bill_amt5 = bill_cols2[1].number_input("Bill 5", value=16000, step=1000)
bill_amt6 = bill_cols2[2].number_input("Bill 6", value=15000, step=1000)

st.subheader("Payment Amounts (last 6 months)")
pay_amt_cols = st.columns(3)
pay_amt1 = pay_amt_cols[0].number_input("Payment 1 (most recent)", min_value=0, value=1000, step=500)
pay_amt2 = pay_amt_cols[1].number_input("Payment 2", min_value=0, value=1000, step=500)
pay_amt3 = pay_amt_cols[2].number_input("Payment 3", min_value=0, value=1000, step=500)
pay_amt_cols2 = st.columns(3)
pay_amt4 = pay_amt_cols2[0].number_input("Payment 4", min_value=0, value=1000, step=500)
pay_amt5 = pay_amt_cols2[1].number_input("Payment 5", min_value=0, value=1000, step=500)
pay_amt6 = pay_amt_cols2[2].number_input("Payment 6", min_value=0, value=1000, step=500)

st.divider()

if st.button("Predict Default Risk", type="primary"):
    raw_input = {
        "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage, "AGE": age,
        "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3, "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
        "BILL_AMT1": bill_amt1, "BILL_AMT2": bill_amt2, "BILL_AMT3": bill_amt3,
        "BILL_AMT4": bill_amt4, "BILL_AMT5": bill_amt5, "BILL_AMT6": bill_amt6,
        "PAY_AMT1": pay_amt1, "PAY_AMT2": pay_amt2, "PAY_AMT3": pay_amt3,
        "PAY_AMT4": pay_amt4, "PAY_AMT5": pay_amt5, "PAY_AMT6": pay_amt6,
    }

    try:
        with st.spinner("Running prediction... (first prediction can take up to a minute while TensorFlow initializes)"):
            print("[app] Starting prediction...", flush=True)
            result = predict_default(raw_input, artifacts)
            print(f"[app] Prediction complete: {result}", flush=True)
    except Exception as e:
        st.exception(e)
        st.stop()

    probability = result["probability"]
    prediction = result["prediction"]

    st.header("Result")
    st.metric("Default Probability", f"{probability:.1%}")

    if prediction == 1:
        st.error("⚠️ High Risk: this client is predicted to default next month.")
    else:
        st.success("✅ Low Risk: this client is predicted NOT to default next month.")

    st.caption(
        "Note: this model was trained with a threshold of 50% and evaluated on "
        "precision, recall, F1, and ROC-AUC rather than accuracy alone, due to "
        "class imbalance in the training data (~78% non-default, ~22% default). "
        "See the project README for full model comparison and evaluation details."
    )