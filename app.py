import json
import math
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

@st.cache_data
def load_parameters():
    path = Path(__file__).with_name("model_parameters.json")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

P = load_parameters()


def make_features(sector, values):
    """Recreate the training-time numeric scaling and sector one-hot encoding."""
    vector = []
    for i, name in enumerate(P["numeric_features"]):
        value = values[name]
        if value is None:
            value = P["numeric_medians"][i]
        scale = P["numeric_scales"][i]
        if not scale:
            scale = 1.0
        vector.append((float(value) - P["numeric_means"][i]) / scale)

    # One-hot encoding uses the exact category order learned during training.
    # An unseen category is represented by all zeros, matching handle_unknown='ignore'.
    selected_sector = sector if sector in P["sector_categories"] else None
    for category in P["sector_categories"]:
        vector.append(1.0 if selected_sector == category else 0.0)
    return vector


def linear_predict(spec, features):
    return float(spec["intercept"] + sum(c * x for c, x in zip(spec["coefficients"], features)))


def logistic_probability(spec, features):
    z = linear_predict(spec, features)
    # Numerically stable sigmoid.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 700)))
    ez = math.exp(max(z, -700))
    return ez / (1.0 + ez)

st.title("📈 Stock Price Prediction System")
st.write("Estimate a stock price from selected financial fundamentals using three regression models.")
st.info("Educational demonstration only. The included dataset is synthetic, not real historical NSE/BSE market data. Predictions are not investment advice.")

with st.sidebar:
    st.header("Company Financial Inputs")
    sector_options = P["sector_categories"]
    sector = st.selectbox("Sector", sector_options)
    eps = st.number_input("Earnings Per Share (EPS)", value=25.0, step=0.5)
    book_value = st.number_input("Book Value", value=100.0, step=1.0)
    roe = st.number_input("Return on Equity (ROE %)", value=15.0, step=0.5)
    debt_equity = st.number_input("Debt-to-Equity Ratio", min_value=0.0, value=0.5, step=0.05)
    sales_growth = st.number_input("Sales Growth (%)", value=10.0, step=0.5)
    run_prediction = st.button("Predict Stock Price", type="primary", use_container_width=True)

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Model Predictions")
    if run_prediction:
        inputs = {"EPS": eps, "BookValue": book_value, "ROE": roe, "DebtEquity": debt_equity, "SalesGrowth": sales_growth}
        features = make_features(sector, inputs)
        results = {name: linear_predict(spec, features) for name, spec in P["regression_models"].items()}
        c1, c2, c3 = st.columns(3)
        for col, (name, value) in zip([c1, c2, c3], results.items()):
            with col:
                st.metric(name, f"₹{value:,.2f}")
        average = sum(results.values()) / len(results)
        st.success(f"Average of the three model estimates: ₹{average:,.2f}")
        st.caption("The average is a simple comparison of model outputs, not a separate trained model.")
        st.subheader("Input Summary")
        st.json({"Sector": sector, **inputs})
        prob_high = logistic_probability(P["classification_model"], features)
        label = "High Price" if prob_high >= 0.5 else "Low Price"
        st.subheader("Logistic Regression Classification")
        st.write(f"Predicted category: **{label}**")
        st.write(f"Probability of High Price: **{prob_high:.1%}**")
        st.caption(f"Classification rule: {P['classification_definition']}")
    else:
        st.write("Enter the financial details in the sidebar and click **Predict Stock Price**.")

with right:
    st.subheader("Model Evaluation")
    for name, spec in P["regression_models"].items():
        m = spec["metrics"]
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.write(f"R²: {m['R2']:.4f} ({m['R2']:.1%})")
            st.write(f"MAE: ₹{m['MAE']:.2f}  |  RMSE: ₹{m['RMSE']:.2f}")
    st.caption(f"Evaluation split: {P['training_rows']} training rows and {P['testing_rows']} testing rows.")

with st.expander("Classification metrics"):
    cm = P["classification_model"]["metrics"]
    st.write(f"Accuracy: {cm['Accuracy']:.1%}")
    st.write(f"Precision: {cm['Precision']:.1%}")
    st.write(f"Recall: {cm['Recall']:.1%}")
    st.write(f"F1 score: {cm['F1']:.1%}")
    st.caption("This classifier labels a record as above or at/below the training-set median price. It does not determine whether a stock is fundamentally undervalued or a good investment.")

with st.expander("About this project"):
    st.write("The models use EPS, Book Value, ROE, Debt-to-Equity, Sales Growth, and Sector. Company name is not used as a predictor.")
    st.write(f"Training dataset: {P['dataset_name']}")
    st.write(P["dataset_type"])
