# Predicting tasks

## Project Overview
This project focuses on predicting venue popularity, order forecasting, and doing a time series order analysis. For predicting venue popularity, we utilize geospatial venue data points (longitude and latitude) and predict if a venue with that coordinates, will be Highly popular, Medium popular, or Low popular. This will help in resource allocation and planning. Dataset from Wolt.

## Folder Structure
```bash
Order_Analysis/
│
├── Data/
│ ├── data_analysis.ipynb
│ ├── orders_autumn_2020.csv
│ ├── user_heatmap.html
│ ├── user_heatmap.png
│ ├── venue_heatmap.html
│ ├── venue_heatmap.png
│ └── venue_orders_heatmap.png
│
├── order_timeseries.py
├── order_forecasting.py
├── popularity.py
├── Presentation.pdf
├── actual_popularity.html
├── predicted_popularity.html
└── requirements.txt
```
### Prerequisites
Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
### Running the Model
The `popularity.py`, `order_timeseries.py`, `order_forecasting.py` scripts are the main files that performs the modelling, training, and evaluation. They accepts several arguments to customize the training process.

#### Example Commands for Venue popularity prediction
Run the following command in the command line to execute venue popularity modelling script with all models:
```bash
python popularity.py --data_path Data/orders_autumn_2020.csv --model_type all --user_loc True --cluster_venues True --n_clusters 10
```
To specify a model type and configure other parameters:
```bash
python popularity.py --data_path Data/orders_autumn_2020.csv --model_type mlp --user_loc False --learning_rate 0.01 --units_layers 500,250,500,500
```
```bash
python popularity.py --data_path Data/orders_autumn_2020.csv --model_type mlp --user_loc False --cluster_venues True --n_clusters 7 
```
Many other hyper-parameters can be set. Here is another example of an ensemble model:
```bash
python popularity.py --data_path Data/orders_autumn_2020.csv --model_type ensemble --epochs 50 --degree 3 --n_estimators 50
```
Models available (`model_type` command): Multilayer Perceptron (`mlp`), Gradient Boosting Classifier (`gb`), Random Forest Classifier (`forest`), Naive Bayes (`naive`), Support Vector Classifier (`svm`), Ensemble model based on all (`ensemble`), All models run (`all`). When running `all`, a significance test will be executed to compare the models and check if there is a statistically significant difference.

## Data Analysis
The `data_analysis.ipynb` notebook within the `Data` folder includes a comprehensive exploratory data analysis (EDA) of the dataset. Key insights are drawn through various visualizations.

## Task Description for Venue Popularity prediction
The task is to predict venue popularity based on geographic coordinates. The model also plots a popularity heatmap demonstrating areas that might be busy.
- This is a classification task. Based on order volume of each coordinate, the task is to classify the venue into `High`, `Medium`, or `Low` popularity category.
- Adjust the `user_loc` argument to use user geographic coordinates as features to the model.
- Adjust the `cluster_venues` and `n_clusters` arguments to create and include geographic clustering as a feature. Note: EDA shows that there is almost zero benefit from this (sometimes might be misleading).
- Synthetic venue coordinates have been created to demonstrate model's generalization. It is expected that when these augmented points fall into 'popular' areas, will be classified into High popular category.  
- Two `HTML` files are created that demonstrate the actual and predicted popularity of the venues. The `actual_popularity` shows all the venues in the dataset. The venues that belong to the test set are also displayed with blue markes. The `predicted_popularity`, presents only the venues from the test set. Red markers indicate misclassification, green ones present the correct classifications. Blue markers present the synthetic venues. Finally, if `cluster_venues` argument is set to True, the created clusters will be visualized in both maps.

## Task Description for Order Time series analysis
The task is to analyze order trends and predict into the future.