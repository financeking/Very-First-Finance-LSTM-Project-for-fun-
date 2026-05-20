# pip install yfinance pandas numpy matplotlib scikit-learn torch

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =========================
# 1. Download data
# =========================

TICKER = "AAPL"
START_DATE = "2018-01-01"

data = yf.download(TICKER, start=START_DATE, auto_adjust=True, progress=False)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

prices = data[["Close"]].dropna()


# =========================
# 2. Scale data
# =========================

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)


# =========================
# 3. Create sequences
# =========================

LOOKBACK = 60

X, y = [], []

for i in range(LOOKBACK, len(scaled_prices)):
    X.append(scaled_prices[i - LOOKBACK:i, 0])
    y.append(scaled_prices[i, 0])

X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0], X.shape[1], 1)


# =========================
# 4. Train-test split
# =========================

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# =========================
# 5. Build PyTorch LSTM model
# =========================

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Take output from the last time step
        last_output = lstm_out[:, -1, :]

        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.fc2(x)

        return x


model = LSTMModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)


# =========================
# 6. Train model
# =========================

EPOCHS = 30

train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    model.train()

    train_prediction = model(X_train)
    train_loss = criterion(train_prediction, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():
        test_prediction = model(X_test)
        test_loss = criterion(test_prediction, y_test)

    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}], "
        f"Train Loss: {train_loss.item():.6f}, "
        f"Test Loss: {test_loss.item():.6f}"
    )


# =========================
# 7. Predict
# =========================

model.eval()

with torch.no_grad():
    predictions = model(X_test).numpy()

actual_values = y_test.numpy()

predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(actual_values)


# =========================
# 8. Evaluation
# =========================

rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
mae = mean_absolute_error(actual_prices, predicted_prices)
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

print("\n===== PyTorch LSTM Model Evaluation =====")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print("=========================================")


# =========================
# 9. Visualization
# =========================

plt.figure(figsize=(14, 6))
plt.plot(actual_prices, label="Actual Price")
plt.plot(predicted_prices, label="Predicted Price")
plt.title(f"{TICKER} Stock Price Prediction using PyTorch LSTM")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Testing Loss")
plt.title("PyTorch LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 10. Forecast future prices
# =========================

FORECAST_DAYS = 30

last_sequence = scaled_prices[-LOOKBACK:]
current_sequence = torch.tensor(
    last_sequence.reshape(1, LOOKBACK, 1),
    dtype=torch.float32
)

future_predictions = []

model.eval()

for _ in range(FORECAST_DAYS):
    with torch.no_grad():
        next_prediction = model(current_sequence)

    next_value = next_prediction.item()
    future_predictions.append(next_value)

    next_tensor = torch.tensor([[[next_value]]], dtype=torch.float32)

    current_sequence = torch.cat(
        (current_sequence[:, 1:, :], next_tensor),
        dim=1
    )

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_prices = scaler.inverse_transform(future_predictions)

future_dates = pd.bdate_range(
    start=prices.index[-1] + pd.Timedelta(days=1),
    periods=FORECAST_DAYS
)

future_df = pd.DataFrame(
    future_prices,
    index=future_dates,
    columns=["Forecasted_Close"]
)

print("\n===== Future Forecast =====")
print(future_df)
print("===========================")


plt.figure(figsize=(14, 6))
plt.plot(prices.index, prices["Close"], label="Historical Price")
plt.plot(future_df.index, future_df["Forecasted_Close"], label="PyTorch LSTM Forecast")
plt.title(f"{TICKER} Future Forecast using PyTorch LSTM")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 11. Final conclusion
# =========================

latest_price = prices["Close"].iloc[-1]
forecast_end = future_df["Forecasted_Close"].iloc[-1]

if forecast_end > latest_price:
    trend = "positive / upward"
else:
    trend = "negative / downward"

print("\n===== Final Conclusion =====")

print(f"""
This project used a PyTorch LSTM deep learning model to analyze and forecast {TICKER} stock prices.

The model was trained using the previous {LOOKBACK} trading days to predict the next closing price.

Model performance:
- RMSE: {rmse:.2f}
- MAE: {mae:.2f}
- MAPE: {mape:.2f}%

The latest actual closing price was ${latest_price:.2f}.
The forecasted price after {FORECAST_DAYS} business days is approximately ${forecast_end:.2f}.

Based on the PyTorch LSTM forecast, the short-term trend appears to be {trend}.

However, this is only an educational model. Stock prices are influenced by many external factors such as news, earnings reports, interest rates, macroeconomic conditions, and investor sentiment.
Therefore, this result should not be considered financial advice.
""")