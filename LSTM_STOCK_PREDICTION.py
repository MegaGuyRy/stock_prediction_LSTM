import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
import schedule
import time
import random
from datetime import datetime, timedelta

def job():

    def stock_data(ticker):
        #stock_data = yf.download(ticker, period='max')
        today = datetime.today()
        yesterday = today - timedelta(days=1)
        end_date = yesterday.strftime('%Y-%m-%d')
        stock_data = yf.download(ticker, start='2021-01-01', end=end_date)

        return stock_data


    def process_data(data):
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

    def create_dataset(data, time_step=60):
        X, Y = [], []
        for i in range(len(data) - time_step - 1): # Will execute until data_rows(size) - time_step because its looking ahead by value time step
            X.append(data[i:(i + time_step)]) # Slice data index i: (start) to i + time_step (stop)
            Y.append(data[i + time_step])  # Predicting all features
        return np.array(X), np.array(Y)

    # Split data into training and testing sets for predictions and actual values
    def test_train_sets(X, Y, rand=False):
        if rand == True:
            randomization = random.randint(1, 100)
            print (randomization)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=rand)
        # Reshape input to be [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

        return X_train, X_test, y_train, y_test

    # Select Desired stocks
    stocks = [
    'AAPL',  # Apple Inc.
    'MSFT',  # Microsoft Corporation
    'GOOGL', # Alphabet Inc. (Google)
    'AMZN',  # Amazon.com, Inc.
    'TSLA',  # Tesla, Inc.
    'FB',    # Meta Platforms, Inc. (Facebook)
    'BRK.B', # Berkshire Hathaway Inc.
    'JNJ',   # Johnson & Johnson
    'V',     # Visa Inc.
    'JPM',   # JPMorgan Chase & Co.
    'PG',    # Procter & Gamble Co.
    'NVDA',  # NVIDIA Corporation
    'DIS',   # The Walt Disney Company
    'MA',    # Mastercard Incorporated
    'HD',    # The Home Depot, Inc.
    'VZ',    # Verizon Communications Inc.
    'NFLX',  # Netflix, Inc.
    'PYPL',  # PayPal Holdings, Inc.
    'ADBE',  # Adobe Inc.
    'INTC'   # Intel Corporation
]
    # Select the relevant features
    features = ['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']
    
    for i in range(len(stocks)):
        data = stock_data(stocks[i]) # Update to later date.
        data = data[features]

        # Process the aquired data 
        scaled_data, scaler = process_data(data)
        #print(scaled_data)

        # Split data into training and testing datasets
        time_step = 60
        X, Y = create_dataset(scaled_data, time_step)
        #print(X.shape)
        #print(Y.shape)
        # Create training and texting sets for X and Y
        X_train, X_test, y_train, y_test = test_train_sets(X, Y, False)
        #print(X_train.shape)
        #print(X_test.shape)
        #print(y_train.shape)    
        #print(y_test.shape)
        # Create the LSTM model
        model = Sequential()
        # Adds an LSTM layer with 50 units (neurons)
        model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        # Add a second LSTM layer because we only want the last input we dont need the whole sequence
        model.add(LSTM(64, return_sequences=False))
        # Adds a dense layer of 25 fully connected neurons
        model.add(Dense(32))
        # Add the Output Layer
        model.add(Dense(Y.shape[1]))  # Adjust output layer to predict all features
        # Complie the model using the optimizer and loss func
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test))

        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform predictions
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        # Inverse transform actual values
        y_train = scaler.inverse_transform(y_train)
        y_test = scaler.inverse_transform(y_test)

        # Plot the results for 'Close' price as an example
        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'].values, label='Actual Price')
        plt.plot(np.arange(time_step, time_step + len(train_predict)), train_predict[:, 3], label='Train Predict - Close')
        plt.plot(np.arange(len(data) - len(test_predict), len(data)), test_predict[:, 3], label='Test Predict - Close')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        API_KEY = 'KEY' # Filler Keys
        SECRET_KEY = 'KEY'
        BASE_URL = 'https://paper-api.alpaca.markets'

        api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

        def place_order(symbol, qty, side, type, time_in_force):
            order = api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force=time_in_force,
            )
            return order

        # Example usage:
        #order = place_order('AAPL', 100, 'buy', 'market', 'gtc')
        #print(order)

        def get_current_position(symbol):
            try:
                position = api.get_position(symbol)
                return position.qty
            except tradeapi.rest.APIError as e:
                print(f"Error fetching position for {symbol}: {e}")
                return 0

        def execute_trade(predictions, threshold=0.02, stock=stocks[i]):
            last_close = data['Close'].iloc[-1] #.iloc[-1]: This accesses the last element in the 'Close' column,
            predicted_price = predictions[-1] #get the latest prediction
            predicted_price = predicted_price[3]
            change = (predicted_price - last_close) / last_close
            print(f"the Predicted Price: {predicted_price} the last close price: {last_close}")

            if change > threshold:
                place_order('AAPL', 1000, 'buy', 'market', 'gtc')
                print(f"Bought 1000: {stock}")
            elif change < -threshold:
                total = get_current_position(stock)
                total = int(total)
                if total > 0:
                    place_order('AAPL', total, 'sell', 'market', 'gtc')
                    print(f"Sold All {stock}")

        # Example usage:
        execute_trade(test_predict)


run = job()
# Schedule the job to run at 9 AM every day
"""
schedule.every().day.at("13:58").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
"""
