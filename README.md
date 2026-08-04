# Credit Risk Classifier

A machine learning project predicting whether a credit card client will default on their payment next month, using the [UCI Default of Credit Card Clients Dataset](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset) (30,000 clients, 23 features). Built to demonstrate an end-to-end classification workflow: EDA, preprocessing, baseline modeling, a TensorFlow neural network, hyperparameter tuning, and deployment.

**🔗 Live app:** [link once deployed]

## Problem

Given a client's demographic information, credit limit, and repayment history over the last 6 months, predict the probability they will default on their credit card payment next month. This is a binary classification problem with meaningful class imbalance (~78% non-default, ~22% default), so the project evaluates models on precision, recall, F1, and ROC-AUC rather than accuracy alone.

## Approach

1. **EDA** — examined target imbalance, feature distributions, and feature-target relationships. Found recent repayment status (`PAY_0`) to be the strongest predictor, and identified undocumented category codes in `EDUCATION`, `MARRIAGE`, and the `PAY_*` columns requiring deliberate cleaning decisions (see `notebooks/01_data_loading.ipynb` and `02_eda.ipynb` for full reasoning).
2. **Preprocessing** — cleaned undocumented codes, log-transformed skewed features, applied a stratified train/val/test split, and standardized features — all leakage-safe (transformation parameters fit on training data only). See `src/preprocessing.py`.
3. **Baseline models** — Logistic Regression and Random Forest, both using class weighting to address imbalance.
4. **Neural network** — a TensorFlow/Keras feedforward classifier, tuned via a manual hyperparameter search over architecture size and dropout rate.
5. **Deployment** — a Streamlit app for interactive, single-client risk prediction, using the saved final model.

## Results

| Model | ROC-AUC | F1 |
|---|---|---|
| Logistic Regression | 0.75 | 0.52 |
| Random Forest | 0.77 | 0.45 |
| Neural Network (untuned) | 0.77 | 0.52 |
| **Neural Network (tuned, final)** | **0.78** | **0.53** |

The tuned neural network was selected as the final model, though the margin over the baselines is modest — a deliberate, documented finding of this project: on tabular data of this size and structure, model complexity does not guarantee a large performance gain over simpler baselines. See `notebooks/04_neural_network.ipynb` and `05_hyperparameter_tuning.ipynb` for full discussion.

## Repo structure

## Running locally

```bash
git clone <repo-url>
cd credit-risk-classifier
pip install -r requirements.txt
streamlit run app/app.py
```

## Tech stack

Python, pandas, scikit-learn, TensorFlow/Keras, Streamlit. Dataset loaded via `kagglehub`, with a local-file fallback.

## Reference

Results in this project can be compared against the dataset's associated paper: Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications.