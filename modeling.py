import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True)
parser.add_argument('--model_type', required=True, choices=['linear', 'forest', 'mlp', 'all'], help='Type of model to use or all to choose all of them')
parser.add_argument('--n_estimators', default=100, type=int)  # for Random Forest
parser.add_argument('--epochs', default=30, type=int)  # for mlp
parser.add_argument('--units_layers', default='500, 250, 500', type=str, help='Type units, layers of MLP.')  # for mlp
parser.add_argument('--solver', default='adam', choices=['adam', 'lbfgs', 'sgd'], help='Type of solver for weight optimization in MLP.')  # for mlp
parser.add_argument('--future_days', default=7, type=int, help='Predict for the next X days')
parser.add_argument('--past_days', default=7, type=int, help='# of past days to plot')


args = parser.parse_args()

# Loading dataset and preporcessing by sorting & filling missing values
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

# Feature engineering method. Picking features, create target variable. 
# Based on EDA, the best features seem to be ITEM_COUNT, ESTIMATED_DELIVERY_MINUTES, ACTUAL_DELIVERY_MINUTES)
def feature_engineering(data):
    try:
        daily_data = data.resample('D').agg({ 
        'ITEM_COUNT': 'sum', # total number of items ordered that day 
        'ESTIMATED_DELIVERY_MINUTES': 'sum',  
        'ACTUAL_DELIVERY_MINUTES': 'sum',
        # 'WIND_SPEED': 'mean',
        # 'ACTUAL_DELIVERY_MINUTES - ESTIMATED_DELIVERY_MINUTES': 'mean',
        # 'USER_LAT': 'mean',
        # 'USER_LONG': 'mean', # for geospatial data -> mean for central loc
        # 'VENUE_LAT': 'mean',
        # 'VENUE_LONG': 'mean',
        # 'CLOUD_COVERAGE': 'mean', # for weather condition makes sense to have mean of the day
        # 'TEMPERATURE': 'mean',
        # 'PRECIPITATION': 'mean'
         })

        # target variable
        daily_data['Order_Volume'] = data.resample('D').size() # count no of orders for each day

        # select features and target
        features = daily_data.drop(columns=['Order_Volume']) # could add here more to drop
        target = daily_data['Order_Volume']

        return daily_data, features, target 
    
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        raise

# Time-based split of the data according to user input. If 7 days as prediction goal, split data by days and use the last 7 days for test.
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

# Model training function. Includes options for 3 models (Random Forest Regression, Linear Regression, and Multilayer Perceptron MLP)
def train_models(X_train, y_train):
    models = {}
    scalers = {} 

    if args.model_type in ['all', 'linear']:
        models['linear'] = LinearRegression().fit(X_train, y_train) # fit linear regression -> does not require scaling 
    if args.model_type in ['all', 'forest']:
        models['forest'] = RandomForestRegressor(n_estimators=args.n_estimators).fit(X_train, y_train) # fit random forest -> does not require scaling 
    if args.model_type in ['all', 'mlp']:
        scaler_X = MinMaxScaler() # scaling features and target
        scaler_y = MinMaxScaler() 
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        mlp_hidden_layers = eval(args.units_layers)  # read into tuple
        models['mlp'] = MLPRegressor(hidden_layer_sizes=mlp_hidden_layers, learning_rate='adaptive', max_iter=args.epochs, verbose=False, solver=args.solver).fit(X_train_scaled, y_train_scaled)

        # storing scalers for MLP
        scalers['mlp'] = {'X': scaler_X, 'y': scaler_y}

    return models, scalers

# Method used to make future predictions 
def predict_future(models, X_test, scalers):
    predictions = {}
    future_dates = pd.date_range(start=X_test.index.max() + pd.Timedelta(days=1), periods=args.future_days, freq='D')

    for model_name, model in models.items():
        future_X = X_test.copy() # use the last # days data as a proxy for future data
        future_X.index = future_dates # future dates to the data
        if model_name == 'mlp':
            # Use appropriate scaler for MLP
            X_test_scaled = scalers['mlp']['X'].transform(X_test)
            pred_scaled = model.predict(X_test_scaled)
            pred = scalers['mlp']['y'].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            predictions[model_name] = pred
        else:
            predictions[model_name] = model.predict(future_X)
        # message for actual and predicted order volumes for the test period
        print(f"Comparing Actual and Predicted Order Volumes for the last 7 days of the test period for {model_name} model:")
        for actual, predicted, date in zip(y_test, predictions[model_name], X_test.index):
            print(f"Date: {date.date()} - Actual: {actual:.0f}, Predicted: {predicted:.0f}")
        print('\n')
    return predictions

# Methods used to plot future predictions for better visualizing the results and the difference in models' performance
def plot_predictions(y_train, y_test, predictions, future_dates, past_days=14):
    recent_train_dates = y_train.index[-past_days:]  # selecting recent past days from y_train
    recent_train_values = y_train.iloc[-past_days:]

    fig = go.Figure()

    # plot past train data
    fig.add_trace(go.Scatter(
        x=recent_train_dates, 
        y=recent_train_values, 
        mode='lines', 
        name='Recent Train Data', 
        line=dict(color='blue')
    ))

    # connecting the last point of y_train to the first point of y_test with a line so we don't see gap
    connecting_dates = [recent_train_dates[-1], y_test.index[0]]
    connecting_values = [recent_train_values.iloc[-1], y_test.iloc[0]]
    fig.add_trace(go.Scatter(
        x=connecting_dates, 
        y=connecting_values, 
        mode='lines', 
        name='Connecting Line',
        line=dict(color='blue'),
        showlegend=False,
        hoverinfo='skip' # don't show this point as it's only for asthetic
    ))

    # plotting actual future data with different line
    actual_future_dates = y_test.index
    fig.add_trace(go.Scatter(
        x=actual_future_dates, 
        y=y_test, 
        mode='lines+markers', 
        name='Actual Future Data', 
        line=dict(color='green')
    ))

    # plotting predicted data from different models as markers
    colors = ['firebrick', 'royalblue', 'gold', 'purple']  # different colors for different models
    for (model_name, predicted), color in zip(predictions.items(), colors):
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=np.rint(predicted), 
            mode='markers', 
            name=f'{model_name.capitalize()} predicted',
            marker=dict(color=color)
        ))

    # updating layout of the figure
    fig.update_layout(
        title='Recent Train, Actual, and Predicted Order Volumes',
        xaxis_title='Date',
        yaxis_title='Order Volume',
        hovermode='x unified'
    )
    fig.show()

# Method used to evaluate model(s). Uses MSE and MAE, R2
def evaluate_models(models, X_test, y_test, scalers):
    model_performance = {}
    for model_name, model in models.items():
        if model_name == 'mlp':
            X_test_scaled = scalers['mlp']['X'].transform(X_test)
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scalers['mlp']['y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        else:
            y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_performance[model_name] = {'MAE': mae, 'MSE': mse, 'R2' : r2}
        print(f"{model_name.capitalize()} Model Evaluation:")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Coefficient of determination: {r2:.2f}\n")
    return model_performance

# Method to print the best model based on MSE
def compare_models(model_performance):
    # --> best model based on lowest MSE
    best_model = min(model_performance, key=lambda k: model_performance[k]['MSE'])
    print(f"Best Model: {best_model.capitalize()}")
    print(f"Performance: MAE = {model_performance[best_model]['MAE']:.2f}, MSE = {model_performance[best_model]['MSE']:.2f}, R2 = {model_performance[best_model]['R2']:.2f}")
    print('\n')

# Main method 
if __name__ == '__main__':
    data = load_and_preprocess_data(args.data_path) # load and preprocess data

    daily_data, features, target = feature_engineering(data) # features and target

    X_train, X_test, y_train, y_test = time_based_split(daily_data, args.future_days, features, target) # time-based split
        
    models, scalers = train_models(X_train, y_train) # train 

    print("\n")
    model_performance = evaluate_models(models, X_test, y_test, scalers) # evaluate models
    compare_models(model_performance) # compare models

    predictions = predict_future(models, X_test, scalers) # make future predictions
    plot_predictions(y_train, y_test, predictions, X_test.index, args.past_days) # plot predictions