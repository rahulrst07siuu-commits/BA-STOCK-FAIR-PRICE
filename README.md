# Stock Price Prediction App (fixed portable version)

## Files
- `app.py`: Streamlit application.
- `model_parameters.json`: model coefficients and preprocessing values in plain JSON.
- `requirements.txt`: runtime dependency.
- `stock_price_prediction_sample_synthetic.csv`: the educational dataset used for training.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Extract this ZIP.
2. Upload all four files to the root of your GitHub repository.
3. In Streamlit Community Cloud, deploy the repository and select `app.py` as the main file.
4. If an older app is already deployed, commit and push the replaced files, then reboot the app.

## Why this version avoids the previous error
The earlier app loaded a scikit-learn object saved with Joblib. Pickled scikit-learn objects can fail when the deployed Python or package versions differ from the training environment. This version stores fitted coefficients, scaling values, and sector categories as plain JSON and performs prediction directly in Python. It does not call `joblib.load` and does not need scikit-learn at runtime.

## Important project limitation
The CSV is synthetic educational data, not real historical NSE/BSE data. The evaluation scores describe performance on a held-out portion of this synthetic dataset only. They should not be presented as real-market predictive accuracy.
