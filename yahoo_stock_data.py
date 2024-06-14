import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

stocks = ["MSFT"]
start_date = "2022-09-01"
end_date = "2023-01-01"

# Download data for multiple stocks at once
data = yf.download(stocks, start=start_date, end=end_date)

# Add a 'Symbol' column to the data
data['Symbol'] = data.index.get_level_values(0)
print(data)
# Select the relevant features
features = ['High', 'Low', 'Open', 'Close', 'Volume']

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])
print(data)
# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Define the number of time steps
n_time_steps = 60

# Reshape the train_data DataFrame to have a 3D shape
train_data_3d = np.reshape(train_data[features].values, (train_data.shape[0], n_time_steps, train_data.shape[1] - 1))

# Define the LSTM model architecture
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(n_time_steps, train_data.shape[1] - 1)),
    tf.keras.layers.Dense(25),
    tf.keras.layers.Dense(1)
])

# Compile the LSTM model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
model.fit(train_data[features], train_data['Close'], epochs=50, batch_size=32)