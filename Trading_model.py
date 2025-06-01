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
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
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


#-------------------

# Improved Model: Enhancements and Cross-Validation

# 1. More Features: Add more technical indicators for richer input
import ta  # Technical Analysis library

def add_advanced_features(df):
    df = df.copy()
    df['Return'] = df['Adj Close'].pct_change()
    df['MA5'] = df['Adj Close'].rolling(window=5).mean()
    df['MA10'] = df['Adj Close'].rolling(window=10).mean()
    df['MA20'] = df['Adj Close'].rolling(window=20).mean()
    df['STD5'] = df['Adj Close'].rolling(window=5).std()
    df['RSI'] = ta.momentum.RSIIndicator(df['Adj Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Adj Close']).macd()
    df['BB_High'] = ta.volatility.BollingerBands(df['Adj Close']).bollinger_hband()
    df['BB_Low'] = ta.volatility.BollingerBands(df['Adj Close']).bollinger_lband()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df = df.dropna()
    return df

msci_feat_adv = add_advanced_features(msci)

# 2. Prepare Data: Use more features, align y
features_adv = ['MA5', 'MA10', 'MA20', 'STD5', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'ATR']
X_adv = msci_feat_adv[features_adv].values
y_adv = (msci_feat_adv['Return'].shift(-1) > 0).astype(int).values[:-1]
X_adv = X_adv[:-1]

# 3. Time Series Split: Use only past data for training, future for testing

scaler_adv = StandardScaler()
X_adv_scaled = scaler_adv.fit_transform(X_adv)

tscv = TimeSeriesSplit(n_splits=4)
fold = 1
val_scores = []
for train_idx, test_idx in tscv.split(X_adv_scaled):
    print(f"\nFold {fold}:")
    X_train_cv, X_test_cv = X_adv_scaled[train_idx], X_adv_scaled[test_idx]
    y_train_cv, y_test_cv = y_adv[train_idx], y_adv[test_idx]

    # 4. Improved Model: Add Dropout, BatchNorm, more layers
    model_cv = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X_train_cv.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model_cv.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 5. Early Stopping
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history_cv = model_cv.fit(
        X_train_cv, y_train_cv,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[es],
        verbose=0
    )

    # 6. Evaluate
    loss_cv, acc_cv = model_cv.evaluate(X_test_cv, y_test_cv, verbose=0)
    print(f"Validation Accuracy: {acc_cv:.3f}")
    val_scores.append(acc_cv)
    fold += 1

print(f"\nAverage 4-Fold Validation Accuracy: {np.mean(val_scores):.3f}")

# 7. Final Model: Train on all but last fold, test on last fold (future data)
train_idx, test_idx = list(tscv.split(X_adv_scaled))[-1]
X_train_final, X_test_final = X_adv_scaled[train_idx], X_adv_scaled[test_idx]
y_train_final, y_test_final = y_adv[train_idx], y_adv[test_idx]

final_model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train_final.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es_final = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history_final = final_model.fit(
    X_train_final, y_train_final,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es_final],
    verbose=1
)

loss_final, acc_final = final_model.evaluate(X_test_final, y_test_final)
print(f"\nTest Accuracy on Future Data: {acc_final:.3f}")

# 8. Backtest Improved Strategy
msci_feat_adv = msci_feat_adv.iloc[:-1]
msci_feat_adv['Pred'] = final_model.predict(X_adv_scaled).flatten() > 0.5
msci_feat_adv['Strategy'] = msci_feat_adv['Pred'].shift(1) * msci_feat_adv['Return']
msci_feat_adv['Strategy'].fillna(0, inplace=True)
msci_feat_adv['Cumulative_Strategy'] = (1 + msci_feat_adv['Strategy']).cumprod()
msci_feat_adv['Cumulative_MSCI'] = (1 + msci_feat_adv['Return']).cumprod()

# 9. Plot Improved Results
plt.figure(figsize=(14,7))
plt.plot(msci_feat_adv.index, msci_feat_adv['Cumulative_Strategy'], label='Improved NN Strategy (MSCI World)')
plt.plot(msci_feat_adv.index, msci_feat_adv['Cumulative_MSCI'], label='Buy & Hold (MSCI World)')
plt.plot(sp500.index, sp500['Cumulative_SP500'], label='Buy & Hold (S&P 500)')
plt.title('Backtest: Improved Neural Network Day Trading vs Benchmarks')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid()
plt.show()

# 10. Explanation of Improvements
print("""
Improvements made:
- Added more technical indicators (MACD, Bollinger Bands, ATR) for richer features.
- Used TimeSeriesSplit for 4-fold cross-validation, ensuring only past data is used for training and future data for testing.
- Enhanced neural network with more layers, BatchNormalization, and Dropout to reduce overfitting.
- Used EarlyStopping to avoid overtraining.
- Final model is trained on all but the last fold and tested on the last (future) fold for realistic performance.
""")
