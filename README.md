# LSTM Stock Price Prediction

## Overview
This project uses a Long Short-Term Memory (LSTM) neural network to predict stock closing prices based on historical data pulled from Yahoo Finance. The goal was to build the framework from scratch (without using high-level auto-architectures) to deepen my understanding of how AI and RNN-based architectures function internally.

The model predicts future stock prices by learning sequential patterns in daily closing prices, leveraging the strength of LSTMs in modeling temporal dependencies.

## Features
- Download and preprocess historical stock data from Yahoo Finance
- Build a custom LSTM architecture in TensorFlow / Keras
- Predict and visualize future stock prices vs actual prices
- Compute mean absolute error (MAE) to evaluate model performance
  
## Dataset
- **Source:** Yahoo Finance
- **Features:** Date, Open, High, Low, Close, Volume
- Used primarily the `Close` price to build sequential training data.

## Model Architecture
- **Custom LSTM:**  
- LSTM layers with dropout for regularization  
- Dense output layer predicting the next closing price
- Optimizer: Adam  
- Loss: Mean Squared Error (MSE)
