import yfinance as yf
import pandas as pd

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

def get_data():
    # Retrieve historical market data for a 10-year period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    
    # Download S&P 500 (^GSPC) and VIX (^VIX) historical data from Yahoo Finance
    sp500 = yf.download("^GSPC", start=start_date, end=end_date, interval="1d", progress=False)
    vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d", progress=False)
    
    # Flatten MultiIndex columns if present
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    # Set up Fear Index (VIX)
    vix = vix[['Close']].rename(columns={'Close': 'Fear_Index'})
    
    # Combine datasets on date index
    data = sp500.join(vix, how='inner')
    
    # Compute technical trend indicators: Simple Moving Averages (SMA 5 and SMA 20)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Compute 14-day Relative Strength Index (RSI-14)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Add calendar features
    data = data.reset_index()
    data['Day_of_Week'] = data['Date'].dt.day_name()

    # Target variable: 1 if next-day close price increases, 0 if decreases
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # Clean missing values resulting from rolling window and shift
    data = data.dropna().iloc[:-1]

    essential_columns = ['Date', 'Open', 'Close', 'Volume', 'Fear_Index', 'SMA_5', 'SMA_20', 'RSI_14', 'Day_of_Week', 'Target']
    data = data[essential_columns]

    # Split dataset sequentially (chronological order without shuffle for time series)
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

    # Persist processed CSV datasets
    train_data.to_csv("../data/train_data.csv", index=False)
    test_data.to_csv("../data/test_data.csv", index=False)
    data.to_csv("../data/sp500_dataset.csv", index=False)

if __name__ == "__main__":
    get_data()