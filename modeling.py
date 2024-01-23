import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import folium
from sklearn.ensemble import GradientBoostingClassifier
from folium.plugins import HeatMap
from geopy.distance import great_circle
import random
import argparse


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True)
parser.add_argument('--model_type', required=True, choices=['forest', 'naive', 'mlp', 'svm', 'gb'], help='Type of model to use')
parser.add_argument('--n_estimators', default=100, type=int)  # for random forest and gradient boosting
parser.add_argument('--kernel', default='rbf', type=str)  # for svm
parser.add_argument('--degree', default=3, type=int)  # for svm
parser.add_argument('--epochs', default=30, type=int)  # for mlp
parser.add_argument('--user_loc', default=False, type=bool)  # use user location as feature
parser.add_argument('--units_layers', default='500, 250, 500', type=str, help='Type units, layers of MLP.')  # for mlp
parser.add_argument('--solver', default='adam', choices=['adam', 'lbfgs', 'sgd'], help='Type of solver for weight optimization in MLP.')  # for mlp

args = parser.parse_args()

def load_data(file_path):
    return pd.read_csv(file_path)

def calculate_average_distance(group):
    venue_location = (group.name[0], group.name[1])  # (VENUE_LAT_ROUNDED, VENUE_LONG_ROUNDED)
    distances = group.apply(lambda row: great_circle(venue_location, (row['USER_LAT'], row['USER_LONG'])).kilometers, axis=1)
    return distances.mean()

#  Feature Engineering
def feature_engineering(data):
    # adjusting coordinates for location accuracy. Close venues will count as 1. Maybe GPS error. 
    data['VENUE_LAT_ROUNDED'] = data['VENUE_LAT'].round(3)
    data['VENUE_LONG_ROUNDED'] = data['VENUE_LONG'].round(3)

    # aggregate orders by venue long and lat
    venue_data = data.groupby(['VENUE_LAT_ROUNDED', 'VENUE_LONG_ROUNDED']).agg(
        total_orders=pd.NamedAgg(column='TIMESTAMP', aggfunc='count')
    ).reset_index()

    # Calculate the average distance of users from each venue
    venue_data['avg_user_distance'] = data.groupby(['VENUE_LAT_ROUNDED', 'VENUE_LONG_ROUNDED']).apply(calculate_average_distance).reset_index(level=[0,1], drop=True)

    #  class popularity tiers
    venue_data['popularity_tier'] = pd.qcut(venue_data['total_orders'], 3, labels=['Low', 'Medium', 'High'])
    if args.user_loc:
        features = ['VENUE_LAT_ROUNDED', 'VENUE_LONG_ROUNDED', 'avg_user_distance'] # feature selection
    else:
        features = ['VENUE_LAT_ROUNDED', 'VENUE_LONG_ROUNDED'] 
    return venue_data, features

# Train models
def train_models(venue_data, features):
    X = venue_data[features]
    y = venue_data['popularity_tier']

    X_train_venue, X_test_venue, y_train_venue, y_test_venue = train_test_split(X, y, test_size=0.2, random_state=42) # train-test split

    train_indices = X_train_venue.index
    test_indices = X_test_venue.index  # retain indices

    # standardizing features
    scaler = StandardScaler()
    X_train_venue = scaler.fit_transform(X_train_venue)
    X_test_venue = scaler.transform(X_test_venue)

    if args.model_type == 'forest':
        model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42) # training random forest classifier
    elif args.model_type == 'mlp':
        mlp_hidden_layers = eval(args.units_layers)  # read into tuple
        model = MLPClassifier(hidden_layer_sizes=mlp_hidden_layers, learning_rate='adaptive', max_iter=args.epochs, verbose=False, solver=args.solver)  # train MLP
    elif args.model_type == 'svm':
        model = SVC(kernel=args.kernel, degree=args.degree) # train SVM
    elif args.model_type == 'gb':
        model = GradientBoostingClassifier(n_estimators=args.n_estimators)
    else:
        model = GaussianNB()  # train Naive Bayes 
    model.fit(X_train_venue, y_train_venue)

    return model, X_test_venue, y_test_venue, scaler, train_indices, test_indices

# Model Evaluation
def eval_model(model, X_test_venue, y_test_venue):
    print("\n")
    print(f"Model Evaluation of {args.model_type.capitalize()}:")
    y_pred = model.predict(X_test_venue)
    classification_metrics_venue = classification_report(y_test_venue, y_pred) # generate report of results
    print(classification_metrics_venue)
    return y_pred

# data augmentation and prediction
def create_synthetic_venues(model, scaler, venue_data):
    # Helsinki city center coordinates --> (60.17, 24.94)
    lat_range = (60.155, 60.195)  # try to find best values for broad synthetic data 
    long_range = (24.925, 24.955)   
    new_venue_data = {     # generating 10 random new venue coordinates within Helsinki area
        'VENUE_LAT_ROUNDED': [random.uniform(*lat_range) for _ in range(10)],
        'VENUE_LONG_ROUNDED': [random.uniform(*long_range) for _ in range(10)]
    }
    if args.user_loc:
        # using mean userdistance from the training data
        avg_user_distance = venue_data['avg_user_distance'].mean() 

        new_venue_data['avg_user_distance'] = [avg_user_distance] * 10  # repeat for all new venues

    new_venues_df = pd.DataFrame(new_venue_data)
    new_venues_df_scaled = scaler.transform(new_venues_df)

    new_venue_predictions = model.predict(new_venues_df_scaled)    # predict popularity tiers of new venues
    new_venues_df['predicted_popularity_tier'] = new_venue_predictions

    return new_venues_df


# Plotting
def plot_map(venue_data, train_indices, test_indices, y_pred, new_venues_df):
    # combine train and test sets for the actual popularity heatmap
    combined_set_df = pd.concat([venue_data.iloc[train_indices], venue_data.iloc[test_indices]])

    # heatmap intensity values for each tier
    intensity_mapping = {'Low': 0.5, 'Medium': 1, 'High': 1.5}

    # add markers
    actual_heatmap_data = [
        (row['VENUE_LAT_ROUNDED'], row['VENUE_LONG_ROUNDED'], intensity_mapping[row['popularity_tier']])
        for idx, row in combined_set_df.iterrows()
    ]
    actual_map = folium.Map(location=[60.17, 24.94], zoom_start=12)
    HeatMap(actual_heatmap_data, radius=25).add_to(actual_map)

    # blue markers for actual popularity tiers in test set
    for idx, row in venue_data.iloc[test_indices].iterrows():
        folium.CircleMarker(
            location=[row['VENUE_LAT_ROUNDED'], row['VENUE_LONG_ROUNDED']],
            radius=5,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7,
            popup=f'Actual Popularity: {row["popularity_tier"]}'
        ).add_to(actual_map)

    actual_map.save('actual_popularity.html')

    # predicted popularity heatmap with markers for correct and incorrect predictions
    test_set_df = venue_data.iloc[test_indices].copy()
    test_set_df['predicted_popularity_tier'] = y_pred
    predicted_heatmap_data = [
        (row['VENUE_LAT_ROUNDED'], row['VENUE_LONG_ROUNDED'], intensity_mapping[row['predicted_popularity_tier']])
        for idx, row in test_set_df.iterrows()
    ]
    predicted_map = folium.Map(location=[60.17, 24.94], zoom_start=12)
    HeatMap(predicted_heatmap_data, radius=25).add_to(predicted_map)

    # adding markers
    for idx, row in test_set_df.iterrows():
        marker_color = 'green' if row['popularity_tier'] == row['predicted_popularity_tier'] else 'red'
        folium.CircleMarker(
            location=[row['VENUE_LAT_ROUNDED'], row['VENUE_LONG_ROUNDED']],
            radius=5,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.7,
            popup=f'Actual: {row["popularity_tier"]}, Predicted: {row["predicted_popularity_tier"]}'
        ).add_to(predicted_map)

    # adding blue markers for new synthetic venue predictions
    for idx, row in new_venues_df.iterrows():
        folium.CircleMarker(
            location=[row['VENUE_LAT_ROUNDED'], row['VENUE_LONG_ROUNDED']],
            radius=5,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7,
            popup=f'Predicted Popularity (New Venue): {row["predicted_popularity_tier"]}'
        ).add_to(predicted_map)

    predicted_map.save('predicted_popularity.html')

if __name__ == '__main__':
    
    data = load_data(args.data_path)
    
    venue_data, features = feature_engineering(data)
    
    model, X_test_venue, y_test_venue, scaler, train_indices, test_indices = train_models(venue_data, features)

    y_pred = eval_model(model, X_test_venue, y_test_venue)

    new_venues_df = create_synthetic_venues(model, scaler, venue_data)
    
    plot_map(venue_data, train_indices, test_indices, y_pred, new_venues_df)