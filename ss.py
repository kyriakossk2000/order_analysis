import argparse
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import pandas as pd
import numpy as np 
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True)
parser.add_argument('--model_type', required=True, choices=['sarima', 'arima', 'all'], help='Type of model to use or all to choose all of them')
parser.add_argument('--future_days', default=7, type=int, help='Predict for the next X days')
parser.add_argument('--past_days', default=60, type=int, help='# of past days to plot')

args = parser.parse_args()

# Loading and preparing data method
def load_prepare_data(filename):
    data = pd.read_csv(filename)                                                
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP']) # TIMESTAMP' to datetime and set as index
    data.set_index('TIMESTAMP', inplace=True)

    data = data.resample('D').size() # resample to get daily order volume
    data.name = 'Order_Volume'
    return data

def get_best_parameters(data, model_type):
    best_params = {}
    if model_type in ['sarima', 'all']:
        # search for best sarima parameters
        auto_sarima_model = auto_arima(data, seasonal=True, m=7, suppress_warnings=True)
        best_params['sarima'] = (auto_sarima_model.order, auto_sarima_model.seasonal_order)

    if model_type in ['arima', 'all']:
        # search for best arima parameters
        auto_arima_model = auto_arima(data, seasonal=False, suppress_warnings=True)
        best_params['arima'] = (auto_arima_model.order, None)  # no seasonal_order for arima

    return best_params

# Define the model and training - options SARIMA and ARIMA models, however EDA showed that SARIMA is more appriopriate
def train_model(data, model_type):
    forecasts = {}
    
    best_params = get_best_parameters(data, model_type)
    print(best_params)
    if model_type in ['sarima', 'all']:
        order, seasonal_order = best_params['sarima']
        sarima_model = SARIMAX(data, 
                               order=order, 
                               seasonal_order=seasonal_order, 
                               enforce_stationarity=False, 
                               enforce_invertibility=False)
        
        sarima_results = sarima_model.fit() # fit sarima
        sarima_forecast = sarima_results.get_forecast(steps=args.future_days)
        forecasts['SARIMA'] = (sarima_forecast.predicted_mean, sarima_forecast.conf_int())

    if model_type in ['arima', 'all']:
        order, _ = best_params['arima']
        arima_model = ARIMA(data, order=order)
        arima_results = arima_model.fit()  # fit arima
        arima_forecast = arima_results.get_forecast(steps=args.future_days)
        forecasts['ARIMA'] = (arima_forecast.predicted_mean, arima_forecast.conf_int())

    return forecasts


# Plotting the historical data and forecast
def plot_forecasting(data, forecasts, past_days):
    fig = go.Figure()

    past_data = data[-past_days:]
    fig.add_trace(go.Scatter(x=past_data.index, y=past_data, mode='lines', name='Historical Daily Order Volume')) # historical data for the specified number of past days

    # colors for different models
    colors = {'SARIMA': 'red', 'ARIMA': 'green'}

    for model_name, (forecast_mean, forecast_conf_int) in forecasts.items():
        forecast_mean = np.rint(forecast_mean)
        forecast_conf_int = np.rint(forecast_conf_int)
        color = colors.get(model_name) 
        fig.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, mode='lines', name=f'{model_name} Forecast', line=dict(color=color))) # add forecasting

        if args.model_type != 'all':
            fig.add_trace(go.Scatter(x=forecast_conf_int.index, y=forecast_conf_int.iloc[:, 0], fill=None, mode='lines', line=dict(color='lightgrey'), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast_conf_int.index, y=forecast_conf_int.iloc[:, 1], fill='tonexty', mode='lines', line=dict(color='lightgrey'), showlegend=False))

    # layout
    fig.update_layout(title='Wolt Order Volume Forecast',
                      xaxis_title='Date',
                      yaxis_title='Order Volume',
                      legend=dict(x=0.03, y=0.97))
    fig.show()


# Calculate and display evaluation metrics. Metrics used: MAE, MSE, and R2 (Coefficient of determination)
def eval_model(data, forecast_mean, model_name):

    # Only calculate metrics for the available data (excluding forecast)
    true_values = data[-len(forecast_mean):]
    mae = mean_absolute_error(true_values, forecast_mean)
    mse = mean_squared_error(true_values, forecast_mean)
    r2 = r2_score(true_values, forecast_mean)
    print(f"{model_name} Model Evaluation:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Coefficient of determination: {r2:.2f}\n")

# Main method 
if __name__ == '__main__':
    data = load_prepare_data(args.data_path)
    forecasts = train_model(data, args.model_type) # train model
    print("\n")
    for model_name, (forecast_mean, forecast_conf_int) in forecasts.items():
        eval_model(data, forecast_mean, model_name)
        plot_forecasting(data, forecasts, args.past_days)