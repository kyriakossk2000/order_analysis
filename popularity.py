import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
from geopy.distance import great_circle
import random
import argparse
from scipy.stats import ttest_rel
import numpy as np

def str2bool(s):
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True)
parser.add_argument('--model_type', required=True, choices=['forest', 'naive', 'mlp', 'svm', 'gb', 'ensemble', 'all'], help='Type of model to use')
parser.add_argument('--n_estimators', default=100, type=int)  # for random forest and gradient boosting
parser.add_argument('--kernel', default='rbf', type=str)  # for svm
parser.add_argument('--degree', default=3, type=int)  # for svm
parser.add_argument('--epochs', default=100, type=int)  # for mlp
parser.add_argument('--user_loc', default=False, type=str2bool)  # use user location as feature
parser.add_argument('--cluster_venues', default=False, type=str2bool)  # cluster venues feature
parser.add_argument('--n_clusters', default=10, type=int)  # use user location as feature
parser.add_argument('--units_layers', default='500, 250, 500, 500', type=str, help='Type units, layers of MLP.')  # for mlp
parser.add_argument('--learning_rate', default=0.01, type=float)  # for mlp
parser.add_argument('--solver', default='adam', choices=['adam', 'lbfgs', 'sgd'], help='Type of solver for weight optimization in MLP.')  # for mlp

args = parser.parse_args()

def load_data(file_path):
    return pd.read_csv(file_path)

# Calculate average distance of users from each venue using great circle:
# Geodesic Distance: 
# It is the length of the shortest path between 2 points on any surface. In our case, the surface is the earth --> latitude-longitude data
# Reference: https://www.geeksforgeeks.org/python-calculate-distance-between-two-places-using-geopy/
def calculate_average_distance(group):
    venue_location = (group.name[0], group.name[1])  # (VENUE_LAT, VENUE_LONG)
    distances = group.apply(lambda row: great_circle(venue_location, (row['USER_LAT'], row['USER_LONG'])).kilometers, axis=1)
    return distances.mean()

#  Feature Engineering
def feature_engineering(data):

    # aggregate orders by venue lat and long
    venue_data = data.groupby(['VENUE_LAT', 'VENUE_LONG']).agg(
        total_orders=pd.NamedAgg(column='TIMESTAMP', aggfunc='count')
    ).reset_index()
    kmeans = None
    if args.cluster_venues:
        # clustering venues based on geographic coordinates
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
        venue_data['cluster_label'] = kmeans.fit_predict(venue_data[['VENUE_LAT', 'VENUE_LONG']])

    # calculate average distance of users from each venue
    if args.user_loc:
        avg_distances = data.groupby(['VENUE_LAT', 'VENUE_LONG']).apply(calculate_average_distance)
        venue_data = venue_data.merge(avg_distances.rename('avg_user_distance'), on=['VENUE_LAT', 'VENUE_LONG'])

    #  class popularity tiers
    venue_data['popularity_tier'] = pd.qcut(venue_data['total_orders'], 3, labels=['Low', 'Medium', 'High'])

    # selecting features
    features = ['VENUE_LAT', 'VENUE_LONG']
    if args.user_loc:
        features.append('avg_user_distance')
    if args.cluster_venues:
        features.append('cluster_label')
    return venue_data, features, kmeans

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
    models = {} # save models
    if args.model_type in ['all', 'ensemble']:
        model1 = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
        model2 = MLPClassifier(hidden_layer_sizes=eval(args.units_layers), learning_rate='adaptive', max_iter=args.epochs, verbose=False, solver=args.solver)
        model3 = SVC(kernel=args.kernel, degree=args.degree, probability=True)  # probability=True for SVC in ensemble
        model4 = GaussianNB()
        model5 = GradientBoostingClassifier(n_estimators=args.n_estimators)

        # create ensemble model
        models['ensemble'] = VotingClassifier(estimators=[
            ('rf', model1), ('mlp', model2), ('svm', model3), ('nb', model4), ('gb', model5)],
            voting='soft').fit(X_train_venue, y_train_venue)  # soft voting for predicted probabilities
        
    if args.model_type in ['all', 'forest']:
        models['forest'] = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42).fit(X_train_venue, y_train_venue) # training random forest classifier
    if args.model_type in ['all', 'mlp']:
        mlp_hidden_layers = eval(args.units_layers)  # read into tuple
        models['mlp'] = MLPClassifier(hidden_layer_sizes=mlp_hidden_layers, learning_rate='adaptive', max_iter=args.epochs, verbose=False, solver=args.solver, learning_rate_init=args.learning_rate).fit(X_train_venue, y_train_venue)  # train MLP
    if args.model_type in ['all', 'svm']:
        models['svm'] = SVC(kernel=args.kernel, degree=args.degree).fit(X_train_venue, y_train_venue) # train SVM
    if args.model_type in ['all', 'gb']:
        models['gb'] = GradientBoostingClassifier(n_estimators=args.n_estimators).fit(X_train_venue, y_train_venue)
    if args.model_type in ['all', 'naive']:
        models['naive'] = GaussianNB().fit(X_train_venue, y_train_venue)  # train Naive Bayes 
    
    return models, X_test_venue, y_test_venue, scaler, train_indices, test_indices

# Significance test for models
def significance_test(models, X_test, y_test, n_splits=5):
    scores = {}
    significant_diff_found = False

    # cross validation scores for each model
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_test, y_test, cv=n_splits)
        scores[name] = cv_scores
    
    for model1 in models:   # comparing models
        for model2 in models:
            if model1 < model2:  # no repetitions
                stat, p = ttest_rel(scores[model1], scores[model2])
                print(f"Comparing {model1} with {model2}: p-value = {p}")
                if p < 0.05:
                    print(f"Significant difference between {model1} and {model2}")
                    significant_diff_found = True

    if not significant_diff_found:
        print("No significant difference found between any of the models.")

    return scores


# Model Evaluation
def eval_model(models, X_test_venue, y_test_venue):
    predictions = {}
    for name, model in models.items():
        if args.model_type in ['all', name]:
            print(f"\nModel Evaluation of {name.capitalize()}:")
            y_pred = model.predict(X_test_venue)
            predictions[name] = y_pred
            classification_metrics_venue = classification_report(y_test_venue, y_pred) # generate report of results
            print(classification_metrics_venue)
    return predictions


# data augmentation and prediction
def create_synthetic_venues(models, scaler, venue_data, model_type, kmeans_model):
    # Helsinki city center coordinates --> (60.17, 24.94)
    lat_range = (60.155, 60.195)  # try to find best values for broad synthetic data 
    long_range = (24.925, 24.955)   
    new_venue_data = {     # generating 10 random new venue coordinates within Helsinki area
        'VENUE_LAT': [random.uniform(*lat_range) for _ in range(10)],
        'VENUE_LONG': [random.uniform(*long_range) for _ in range(10)]
    }
    if args.user_loc:
        # using mean user distance from the training data
        avg_user_distance = venue_data['avg_user_distance'].mean() 
        new_venue_data['avg_user_distance'] = [avg_user_distance] * 10  # repeat for all new venues

    new_venues_df = pd.DataFrame(new_venue_data)

    # assign cluster labels to synthetic venues
    if args.cluster_venues:
        new_venues_df['cluster_label'] = kmeans_model.predict(new_venues_df[['VENUE_LAT', 'VENUE_LONG']])
    
    new_venues_df_scaled = scaler.transform(new_venues_df)

    selected_model = models.get(model_type, None)
    if selected_model is None:
        raise ValueError("Model type not found in trained models")

    new_venue_predictions = selected_model.predict(new_venues_df_scaled)    # predict popularity tiers of new venues
    new_venues_df['predicted_popularity_tier'] = new_venue_predictions

    return new_venues_df


# Plotting
def plot_map(venue_data, train_indices, test_indices, predictions, new_venues_df, model_name, kmeans_model):
    # combine train and test sets for the actual popularity heatmap
    combined_set_df = pd.concat([venue_data.iloc[train_indices], venue_data.iloc[test_indices]])

    # heatmap intensity values for each tier
    intensity_mapping = {'Low': 0.5, 'Medium': 1, 'High': 1.5}

    # add markers
    actual_heatmap_data = [
        (row['VENUE_LAT'], row['VENUE_LONG'], intensity_mapping[row['popularity_tier']])
        for idx, row in combined_set_df.iterrows()
    ]
    actual_map = folium.Map(location=[60.17, 24.94], zoom_start=12)
    predicted_map = folium.Map(location=[60.17, 24.94], zoom_start=12)

    HeatMap(actual_heatmap_data, radius=25).add_to(actual_map)

    # visualize kmeans cluster centers
    if args.cluster_venues:
        cluster_centers = kmeans_model.cluster_centers_
        for center in cluster_centers:
            folium.Circle(
                location=center,  # cluster center location (lat, long)
                radius=500,  # 
                color='#1f77b4',
                fill=True,
                fill_color='#1f77b4',
                fill_opacity=0.2  # opacity low
            ).add_to(actual_map)
            folium.Circle(
                location=center,  
                radius=500,  
                color='#1f77b4',  # Circle color
                fill=True,
                fill_color='#1f77b4',
                fill_opacity=0.2 
            ).add_to(predicted_map)

    # blue markers for actual popularity tiers in test set
    for idx, row in venue_data.iloc[test_indices].iterrows():
        folium.CircleMarker(
            location=[row['VENUE_LAT'], row['VENUE_LONG']],
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
    test_set_df['predicted_popularity_tier'] = predictions[model_name]
    predicted_heatmap_data = [
        (row['VENUE_LAT'], row['VENUE_LONG'], intensity_mapping[row['predicted_popularity_tier']])
        for idx, row in test_set_df.iterrows()
    ]

    HeatMap(predicted_heatmap_data, radius=25).add_to(predicted_map)

    # adding markers
    for idx, row in test_set_df.iterrows():
        marker_color = 'green' if row['popularity_tier'] == row['predicted_popularity_tier'] else 'red'
        folium.CircleMarker(
            location=[row['VENUE_LAT'], row['VENUE_LONG']],
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
            location=[row['VENUE_LAT'], row['VENUE_LONG']],
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
    
    venue_data, features, kmeans_model = feature_engineering(data)
    
    models, X_test_venue, y_test_venue, scaler, train_indices, test_indices = train_models(venue_data, features)

    predictions = eval_model(models, X_test_venue, y_test_venue)

    chosen_model_type = args.model_type if args.model_type != 'all' else 'mlp'  # default to mlp
    new_venues_df = create_synthetic_venues(models, scaler, venue_data, chosen_model_type, kmeans_model)
    
    plot_map(venue_data, train_indices, test_indices, predictions, new_venues_df, chosen_model_type, kmeans_model)

    test_results = significance_test(models, X_test_venue, y_test_venue) if args.model_type == 'all' else None