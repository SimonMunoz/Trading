import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# 1. Download Data: MSCI World (URTH ETF) and S&P 500 (SPY ETF)
start_date = "2015-01-01"
end_date = "2023-12-31"
msci = yf.download("URTH", start=start_date, end=end_date)
sp500 = yf.download("SPY", start=start_date, end=end_date)

# 2. Feature Engineering: Use technical indicators as features
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

msci_feat = add_features(msci)

# 3. Prepare Data for ML
features = ['MA5', 'MA10', 'MA20', 'STD5', 'RSI']
X = msci_feat[features].values
y = (msci_feat['Return'].shift(-1) > 0).astype(int).values[:-1]  # 1 if next day up, else 0
X = X[:-1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# 4. Build Neural Network
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train Model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

# 6. Evaluate Model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# 7. Backtest Strategy
msci_feat = msci_feat.iloc[:-1]  # align with y
msci_feat['Pred'] = model.predict(X_scaled).flatten() > 0.5
msci_feat['Strategy'] = msci_feat['Pred'].shift(1) * msci_feat['Return']
msci_feat['Strategy'].fillna(0, inplace=True)
msci_feat['Cumulative_Strategy'] = (1 + msci_feat['Strategy']).cumprod()
msci_feat['Cumulative_MSCI'] = (1 + msci_feat['Return']).cumprod()

# S&P 500 Benchmark
sp500 = sp500.loc[msci_feat.index]
sp500['Return'] = sp500['Adj Close'].pct_change().fillna(0)
sp500['Cumulative_SP500'] = (1 + sp500['Return']).cumprod()

# 8. Plot Results
plt.figure(figsize=(14,7))
plt.plot(msci_feat.index, msci_feat['Cumulative_Strategy'], label='NN Strategy (MSCI World)')
plt.plot(msci_feat.index, msci_feat['Cumulative_MSCI'], label='Buy & Hold (MSCI World)')
plt.plot(sp500.index, sp500['Cumulative_SP500'], label='Buy & Hold (S&P 500)')
plt.title('Backtest: Neural Network Day Trading vs Benchmarks')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid()
plt.show()

# 9. Plot Training History
plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 10. Plot Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = (model.predict(X_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix on Test Set')
plt.show()

# 11. Print Explanation
print("""
This script:
- Downloads daily data for MSCI World (URTH) and S&P 500 (SPY).
- Computes technical indicators as features.
- Trains a neural network to predict next-day up/down movement.
- Backtests a simple strategy: invest if model predicts up, else stay out.
- Benchmarks against buy-and-hold on MSCI World and S&P 500.
- Plots cumulative returns, training history, and confusion matrix.
""")