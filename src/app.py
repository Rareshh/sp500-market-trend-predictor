import pandas as pd
import gradio as gr
import joblib

# Load pre-trained scaler and models
scaler = joblib.load("../models/scaler.pkl")
model_rf = joblib.load("../models/RandomForest.pkl")
model_lr = joblib.load("../models/RegresieLogistica.pkl")
model_knn = joblib.load("../models/KNN.pkl")

# Mapping dictionary supporting both English display labels and model file names
modele_dict = {
    "Random Forest": model_rf,
    "Logistic Regression": model_lr,
    "KNN": model_knn,
    "RandomForest": model_rf,
    "RegresieLogistica": model_lr,
    "LogisticRegression": model_lr
}

# Feature names expected by the scaler
features_model = scaler.feature_names_in_

def predict_market(nume_model, open_p, close_p, vol, fear, sma5, sma20, rsi, ziua):
    # Initialize zero-filled DataFrame with model feature schema
    date_intrare = pd.DataFrame(0.0, index=[0], columns=features_model)
    
    # Populate numerical features from user inputs
    date_intrare['Open'] = float(open_p)
    date_intrare['Close'] = float(close_p)
    date_intrare['Volume'] = float(vol)
    date_intrare['Fear_Index'] = float(fear)
    date_intrare['SMA_5'] = float(sma5)
    date_intrare['SMA_20'] = float(sma20)
    date_intrare['RSI_14'] = float(rsi)
    
    # One-Hot Encoding for Day of Week
    coloana_zi = "Day_of_Week_" + ziua
    if coloana_zi in features_model:
        date_intrare[coloana_zi] = 1.0
        
    # Standardize input feature vector
    date_scalate = scaler.transform(date_intrare)
    
    # Run model inference
    model = modele_dict[nume_model]
    pred = model.predict(date_scalate)[0]
    prob = model.predict_proba(date_scalate)[0]
    
    # Format prediction summary and probability output
    text_prob = f"Bullish Probability (Up): {prob[1]*100:.2f}%\nBearish Probability (Down): {prob[0]*100:.2f}%"
    if pred == 1:
        rezultat = f"Prediction: Bullish (Price Increase)\n\n{text_prob}"
    else:
        rezultat = f"Prediction: Bearish (Price Decrease)\n\n{text_prob}"

    # Determine image asset paths
    model_key = "random_forest" if "Random" in nume_model else ("logistic_regression" if "Logis" in nume_model else "knn")
    cale_mc = f"../images/confusion_matrix_{model_key}.png"
    cale_roc = "../images/roc_curves_comparison.png"

    return rezultat, cale_mc, cale_roc

with gr.Blocks(title="S&P 500 Market Trend Predictor") as interfata:
    gr.Markdown("# 📈 S&P 500 Market Movement Predictor")
    gr.Markdown("Enter daily market parameters and technical indicators to predict the next-day index direction.")

    with gr.Row():
        with gr.Column():
            m_dropdown = gr.Dropdown(choices=["Random Forest", "Logistic Regression", "KNN"], label="Machine Learning Model", value="Random Forest")
            inp_open = gr.Number(label="Open Price ($)", value=5000)
            inp_close = gr.Number(label="Close Price ($)", value=5050)
            inp_vol = gr.Number(label="Trading Volume", value=3000000000)
            inp_fear = gr.Slider(0, 100, label="VIX (Fear / Volatility Index)", value=15)
            inp_sma5 = gr.Number(label="SMA 5 (5-Day Simple Moving Average)", value=5020)
            inp_sma20 = gr.Number(label="SMA 20 (20-Day Simple Moving Average)", value=4900)
            inp_rsi = gr.Slider(0, 100, label="RSI 14 (14-Day Relative Strength Index)", value=60)
            inp_zi = gr.Dropdown(choices=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], label="Day of Week", value="Monday")
            btn = gr.Button("Run Prediction", variant="primary")
    
        with gr.Column():
            out_txt = gr.Textbox(label="Prediction Result", lines=4)
            out_mc = gr.Image(label="Model Confusion Matrix")
            out_roc = gr.Image(label="Comparative ROC Curve")

    btn.click(
        fn=predict_market,
        inputs=[m_dropdown, inp_open, inp_close, inp_vol, inp_fear, inp_sma5, inp_sma20, inp_rsi, inp_zi],
        outputs=[out_txt, out_mc, out_roc]
    )

if __name__ == "__main__":
    interfata.launch(allowed_paths=["../images"])