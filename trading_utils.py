
def add_features(df):
    df = df.copy()
    df['Return'] = df['Adj Close'].pct_change()
    df['MA5'] = df['Adj Close'].rolling(window=5).mean()
    df['MA10'] = df['Adj Close'].rolling(window=10).mean()
    df['MA20'] = df['Adj Close'].rolling(window=20).mean()
    df['STD5'] = df['Adj Close'].rolling(window=5).std()
    df['RSI'] = 100 - (100/(1 + df['Return'].rolling(window=14).mean() / df['Return'].rolling(window=14).std()))
    df = df.dropna()
    return df

def add_advanced_features(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD5'] = df['Close'].rolling(window=5).std()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['BB_High'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['BB_Low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df = df.dropna()
    return df

print("Training utilities loaded successfully.")
# This module provides utility functions for feature engineering in training datasets.