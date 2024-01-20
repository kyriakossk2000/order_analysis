import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Function to convert strings to boolean
def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True)
parser.add_argument('--model_type', required=True, choices=['linear', 'forest', 'lstm'], help='Type of model to use')
parser.add_argument('--test_size', default=0.2, type=float)
parser.add_argument('--n_estimators', default=100, type=int)  # Only for Random Forest
parser.add_argument('--epochs', default=10, type=int)  # Only for LSTM
parser.add_argument('--output_dir', default='model_output')
parser.add_argument('--timesteps', default=1, type=int, help='Number of timesteps for LSTM input')
parser.add_argument('--future_days', default=7, type=int, help='Predict for the next X days')

args = parser.parse_args()

def preprocess_data(data_path):
    data = pd.read_csv(data_path)  # Load the data
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
    data.sort_values('TIMESTAMP', inplace=True)  # sort data by timestamp just in case there are some not correctly sorted ones
    data.fillna(data.mean(), inplace=True)
    return data

def feature_engineering(data):
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
    data.fillna(data.mean(), inplace=True)
    data['Hour'] = data['TIMESTAMP'].dt.hour
    data['Day_Of_Week'] = data['TIMESTAMP'].dt.day_name().astype('category').cat.codes
    # Ensure TIMESTAMP is set as the index
    data.set_index('TIMESTAMP', inplace=True)

    # First create the Order_Volume column
    data['Order_Volume'] = 1  # Assign 1 to each order

    # Aggregate to daily level
    daily_data = data.resample('D').agg({
        'USER_LAT': 'mean', 'USER_LONG': 'mean', 'ITEM_COUNT': 'sum',
        'TEMPERATURE': 'mean', 'WIND_SPEED': 'mean', 
        'CLOUD_COVERAGE': 'mean', 'PRECIPITATION': 'mean',
        'Day_Of_Week': 'first',
        'Order_Volume': 'sum'  # Sum the order volumes for each day
    })
    
    return daily_data.dropna().reset_index()

def split_data(data):
    features = ['Day_Of_Week', 'USER_LAT', 'USER_LONG', 'ITEM_COUNT', 'TEMPERATURE', 'WIND_SPEED', 'CLOUD_COVERAGE', 'PRECIPITATION']
    X = data[features]
    y = data['Order_Volume']
    return train_test_split(X, y, test_size=args.test_size, random_state=42)

def select_model(model_type, input_shape):
    if model_type == 'linear':
        return LinearRegression()
    elif model_type == 'forest':
        return RandomForestRegressor(n_estimators=args.n_estimators, random_state=42)
    elif model_type == 'lstm':
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    else:
        raise ValueError("Invalid model type selected")

def create_sequences(X, y, timesteps):
    """Prepares data for LSTM: Each sequence (X) predicts the next value (y)."""
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X.iloc[i:(i + timesteps)].values)
        ys.append(y.iloc[i + timesteps])
    return np.array(Xs), np.array(ys)

def train_model(model, X_train, y_train, timesteps):
    """Trains the model. LSTM requires sequence data."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if isinstance(model, Sequential):  # LSTM
        X_train_seq, y_train_seq = create_sequences(pd.DataFrame(X_train_scaled), y_train, timesteps)
        model.fit(X_train_seq, y_train_seq, epochs=args.epochs, verbose=1)
    else:
        model.fit(X_train_scaled, y_train)
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test, timesteps):
    if isinstance(model, Sequential):  # LSTM
        X_test_scaled = scaler.transform(X_test)
        X_test_seq, y_test_seq = create_sequences(pd.DataFrame(X_test_scaled), y_test, timesteps)
        y_pred = model.predict(X_test_seq)
        mae = mean_absolute_error(y_test_seq, y_pred)
        mse = mean_squared_error(y_test_seq, y_pred)
    else:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
    return mae, mse

def predict_future_days(model, scaler, last_data, future_days, model_type, timesteps):
    future_predictions = []
    current_data = last_data.copy()

    for _ in range(future_days):
        # Transform the data for the model
        current_scaled = scaler.transform([current_data])
        
        if model_type == 'lstm':
            # Reshape data for LSTM
            current_scaled_reshaped = np.reshape(current_scaled, (1, timesteps, current_scaled.shape[1]))
            next_day_volume = model.predict(current_scaled_reshaped)[0,0]
        else:
            # Use 2D data for Linear Regression and Random Forest
            next_day_volume = model.predict(current_scaled)[0]

        future_predictions.append(next_day_volume)
        # Update current_data for next prediction (simplified approach)
        current_data = np.roll(current_data, -1)
        current_data[-1] = next_day_volume

    return future_predictions


if __name__ == '__main__':
    results_path = os.path.join(args.output_dir, 'evaluation_results.txt') # file to save evaluation results
    f = open(results_path, 'w')

    os.makedirs(args.output_dir, exist_ok=True)
    data = preprocess_data(args.data_path)
    data = feature_engineering(data)

    X_train, X_test, y_train, y_test = split_data(data)
    input_shape = (args.timesteps, len(X_train.columns))

    model = select_model(args.model_type, input_shape)
    print(f"Training {args.model_type} model...")
    model, scaler = train_model(model, X_train, y_train, args.timesteps)


    print("Evaluating model...")
    mae, mse = evaluate_model(model, scaler, X_test, y_test, args.timesteps)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")

    f.write(f'Mean Absolute Error: {mae}\n')
    f.write(f'Mean Squared Error: {mse}\n')

    # Make future predictions
    last_data_point = X_test.iloc[-1].values  # Get the last row as an array
    future_days = args.future_days
    future_predictions = predict_future_days(model, scaler, last_data_point, future_days, args.model_type, args.timesteps)

    print("Future Order Volumes for the next {} days:".format(future_days))
    for i, volume in enumerate(future_predictions, 1):
        print(f"Day {i}: {volume}")

    f.close()
