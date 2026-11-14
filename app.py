
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Load saved models
bundle = joblib.load("model_bundle.joblib")
models = bundle["models"]

st.title("📈 Stock Price Prediction System")
st.write(
    "Predict an estimated stock price using company financial fundamentals."
)

st.subheader("Enter Company Financial Details")

col1, col2 = st.columns(2)

with col1:
    sector = st.selectbox(
        "Select Sector",
        ["Banking", "IT", "Pharma", "FMCG", "Automobile", "Energy",
         "Textile", "Other"]
    )

    eps = st.number_input("EPS", value=5.0)
    book_value = st.number_input("Book Value", value=4.0)
    roe = st.number_input("ROE", value=2.0)

with col2:
    debt_equity = st.number_input(
        "Debt to Equity Ratio", value=0.10, min_value=0.0
    )

    sales_growth = st.number_input("Sales Growth (%)", value=6.0)

input_data = pd.DataFrame([{
    "EPS": eps,
    "BookValue": book_value,
    "ROE": roe,
    "DebtEquity": debt_equity,
    "SalesGrowth": sales_growth,
    "Sector": sector
}])

st.divider()

if st.button("Predict Stock Price", type="primary"):

    st.subheader("Prediction Results")

    results = {}

    for name, model in models.items():
        prediction = model.predict(input_data)[0]
        results[name] = prediction

    col1, col2, col3 = st.columns(3)

    columns = [col1, col2, col3]

    for (name, prediction), column in zip(results.items(), columns):
        with column:
            st.metric(
                label=name,
                value=f"{prediction:,.2f}"
            )

    average_prediction = sum(results.values()) / len(results)

    st.success(
        f"Average Estimated Stock Price: {average_prediction:,.2f}"
    )

    st.caption(
        "This is an educational model using the supplied dataset. "
        "It is not investment advice or a guaranteed market price."
    )
