import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True)
parser.add_argument('--model_type', required=True, choices=['linear', 'forest', 'lstm'], help='Type of model to use')
parser.add_argument('--n_estimators', default=100, type=int)  # for Random Forest
parser.add_argument('--epochs', default=10, type=int)  # for LSTM
parser.add_argument('--timesteps', default=1, type=int, help='Number of timesteps for LSTM input')
parser.add_argument('--future_days', default=7, type=int, help='Predict for the next X days')
parser.add_argument('--past_days', default=7, type=int, help='# of past days to plot')


args = parser.parse_args()

def load_and_preprocess_data(data_path):
    try:
        data = pd.read_csv(data_path)
        data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
        data.set_index('TIMESTAMP', inplace=True)
        data.sort_values('TIMESTAMP', inplace=True)
        data.fillna(data.mean(), inplace=True)
        return data
    
    except Exception as e:
        print(f"Error loading and preprocessing data: {e}")
        raise

def feature_engineering(data):
    try:
        daily_data = data.resample('D').agg({
        'ACTUAL_DELIVERY_MINUTES - ESTIMATED_DELIVERY_MINUTES': 'mean',   # Aggregate data to daily level
        'ITEM_COUNT': 'sum',
        'USER_LAT': 'mean',
        'USER_LONG': 'mean',
        'VENUE_LAT': 'mean',
        'VENUE_LONG': 'mean',
        'ESTIMATED_DELIVERY_MINUTES': 'mean',
        'ACTUAL_DELIVERY_MINUTES': 'mean',
        'CLOUD_COVERAGE': 'mean',
        'TEMPERATURE': 'mean',
        'WIND_SPEED': 'mean',
        'PRECIPITATION': 'mean'
        })

        # target variable
        daily_data['Order_Volume'] = data.resample('D').size() # count no of orders for each day

        # Select features and target
        features = daily_data.drop(columns=['Order_Volume'])
        target = daily_data['Order_Volume']

        return daily_data, features, target 
    
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        raise

# time-based split of the data
def time_based_split(data, days_for_test, features, target):
    try:
        split_date = data.index.max() - pd.Timedelta(days=days_for_test)
        X_train = features.loc[features.index <= split_date]
        X_test = features.loc[features.index > split_date]
        y_train = target.loc[target.index <= split_date]
        y_test = target.loc[target.index > split_date]
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error in time-based data split: {e}")
        raise

def train_model(X_train, y_train):
    try:
        model = LinearRegression() # linear regression model fit
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error in model training :{e}")
        raise

def eval_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"\nModel Evaluation:\nMean Absolute Error: {mae:.2f}\nMean Squared Error: {mse:.2f}")

    except Exception as e:
        print(f"Error in model evaluation: {e}")
        raise

def predict_future(X_test):
    try:
        future_dates = pd.date_range(start=X_test.index.max() + pd.Timedelta(days=1), periods=args.future_days, freq='D')
        future_X = X_test.copy()  # use the last # days data as a proxy for future data
        future_X.index = future_dates  # Assigning future dates to the data
        predicted_data =  model.predict(X_test)
        # Print actual and predicted order volumes for the test period
        print("Comparing Actual and Predicted Order Volumes for the last 7 days of the test period:")
        for actual, predicted, date in zip(y_test, predicted_data, X_test.index):
            print(f"Date: {date.date()} - Actual: {actual:.0f}, Predicted: {predicted:.0f}")
        return predicted_data
    except Exception as e:
        print(f"Error in predicting future order volume. Please check the prediction window!: {e}")
        raise


def plot_predictions(y_train, y_test, predicted, future_dates, past_days=14):
    # selecting most recent past_days data points from y_train
    recent_train_dates = y_train.index[-past_days:]
    recent_train_values = y_train.iloc[-past_days:]

    fig = go.Figure()

    # plotting recent train data
    fig.add_trace(go.Scatter(x=recent_train_dates, y=recent_train_values, mode='lines', name='Recent Train Data', line=dict(color='blue')))

    # continue line with from the last point of y_train to the first point of y_test
    combined_dates = recent_train_dates.tolist() + [y_test.index[0]]
    combined_values = recent_train_values.tolist() + [y_test.iloc[0]]
    fig.add_trace(go.Scatter(x=combined_dates, y=combined_values, mode='lines', name='Connecting Line', line=dict(color='blue')))   # plotting the connecting line
 
    # plotting actual future data as a continuous line
    actual_future_dates = y_test.index
    fig.add_trace(go.Scatter(x=actual_future_dates, y=y_test, mode='lines+markers', name='Actual Future Data', line=dict(color='green')))

    # plotting predicted data as separate markers (orange)
    fig.add_trace(go.Scatter(x=future_dates, y=np.rint(predicted), mode='markers', name='Predicted', marker=dict(color='firebrick')))

    # Legend design
    fig.update_layout(
        title='Recent Train, Actual, and Predicted Order Volumes',
        xaxis_title='Date',
        yaxis_title='Order Volume',
        hovermode='x unified'
    )

    fig.show()


if __name__ == '__main__':
    data = load_and_preprocess_data(args.data_path) # load and preprocess data

    daily_data, features, target = feature_engineering(data)

    X_train, X_test, y_train, y_test = time_based_split(daily_data, args.future_days, features, target) # time-based split

    model = train_model(X_train, y_train) # train 
    eval_model(model, X_test, y_test) # evaluate
    predicted_data = predict_future(X_test) # make predictions

    plot_predictions(y_train, y_test, predicted_data, X_test.index, args.past_days)


    