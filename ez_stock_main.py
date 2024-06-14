import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Fetch historical stock data from Yahoo Finance
ticker_symbol = 'AAPL'  # Example: Apple Inc.
data = yf.download(ticker_symbol, start='1970-01-01', end='2024-05-14')

# Extract relevant features
features = ['High', 'Low', 'Open', 'Close', 'Volume']
data = data[features]

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
print(scaled_data)

# Define sequence length (number of previous days to use for prediction)
n_steps = 500

# Initialize empty test set (X_test) and ground truth (y_test)
X_test, y_test = [], []

# Iterate over the scaled data to generate sequences for testing
for i in range(len(scaled_data) - n_steps):
    X_test.append(scaled_data[i:i + n_steps, :])  # Previous n_steps days' data
    y_test.append(scaled_data[i + n_steps, 3])    # Next day's close price (target)

X_test, y_test = np.array(X_test), np.array(y_test)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X_test))

X_train, y_train = X_test[:split_index], y_test[:split_index]
X_live, y_live = X_test[split_index:], y_test[split_index:]

# Define LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(X_train, y_train, epochs=100, batch_size=256)

# Iterate over live data for predicting each day
predicted_prices = []

for i in range(len(X_live)):
    # Reshape the input data for prediction
    x_live = X_live[i].reshape(1, n_steps, len(features))
    # Predict the next day's close price
    predicted_price_scaled = model.predict(x_live)
    # Inverse transform the predicted price to original scale
    predicted_price = scaler.inverse_transform([[0, 0, 0, predicted_price_scaled[0][0], 0]])[0, 3]
    # Append the predicted price to the list
    predicted_prices.append(predicted_price)

# Output the predicted prices and compare with actual prices
print("Predicted Close Prices vs. Actual Close Prices:")
for i in range(len(predicted_prices)):
    day_index = split_index + i  # Calculate the day index in the original data
    actual_price = data.iloc[day_index + n_steps]['Close']  # Get the actual close price
    print(f"Day {day_index}: Predicted = {predicted_prices[i]:.2f}, Actual = {actual_price:.2f}")
