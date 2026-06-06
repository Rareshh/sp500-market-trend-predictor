# S&P 500 Market Direction Predictor

A machine learning project that predicts next-day price movement directions for the S&P 500 index (`^GSPC`) based on technical indicators and market volatility, paired with an interactive Gradio web app.

Built for the PCLP3 course at UPB (Faculty of Automatic Control and Computers).

## Overview

Rather than predicting exact stock prices, the problem is framed as a binary classification task:
- `1` (Bullish): Next day's close price is higher than today's close.
- `0` (Bearish): Next day's close price is lower or equal.

The dataset covers 10 years of historical daily data fetched from Yahoo Finance via `yfinance`.

## Features Used

- **Price & Volume**: Daily Open, Close, and Volume.
- **Market Sentiment**: CBOE Volatility Index (`^VIX`) as a fear/volatility gauge.
- **Trend Indicators**: 5-day and 20-day Simple Moving Averages (`SMA_5`, `SMA_20`).
- **Momentum**: 14-day Relative Strength Index (`RSI_14`).
- **Calendar**: Day of the week (one-hot encoded).

## Data Processing & Modeling

1. **Preprocessing**: Missing values are imputed using column medians.
2. **Train/Test Split**: Chronological 80/20 split (`shuffle=False`) to avoid lookahead bias and data leakage.
3. **Scaling**: Numerical features are standardized using `StandardScaler` (fitted strictly on training data).
4. **Classifiers**:
   - Random Forest (`n_estimators=100`, `max_depth=7`)
   - Logistic Regression (baseline)
   - K-Nearest Neighbors (`k=50`)
5. **Evaluation**: Models are evaluated on accuracy, precision, recall, F1-score, confusion matrices, and comparative ROC curves.

## Interactive Web Interface

An interactive dashboard is implemented in `src/app.py` using Gradio:
- Choose between the trained models (Random Forest, Logistic Regression, KNN).
- Input market metrics (price, volume, VIX, SMA, RSI, day of week).
- See the predicted direction, probability breakdown, confusion matrix, and ROC curve.

## Project Structure

```text
Project_SP500/
├── data/                    # Generated train, test, and raw CSV files
├── images/                  # Plots (heatmaps, boxplots, ROC curves, confusion matrices)
├── models/                  # Saved models (.pkl) and scaler
├── src/
│   ├── fetch_data.py        # Downloads data from Yahoo Finance
│   ├── eda_analysis.py      # Exploratory data analysis and baseline model
│   ├── train_models.py      # Model training, comparison, and evaluation
│   └── app.py               # Gradio web app
├── requirements.txt
└── README.md
```

## How to Run

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Download data:**
```bash
cd src
python fetch_data.py
```

**3. Run analysis and model training:**
```bash
python eda_analysis.py
python train_models.py
```

**4. Launch the web app:**
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser.

## Author

- Rares Spinu
- GitHub: [Rareshh](https://github.com/Rareshh)
- Student at Faculty of Automatic Control and Computers, POLITEHNICA Bucharest
