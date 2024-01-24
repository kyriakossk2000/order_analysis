# Predicting Venue Popularity for Wolt

## Project Overview
This project focuses on predicting venue popularity. It aims to utilize geospatial venue data points (longitude and latitude) and predict if a venue with that coordinates, will be Highly popular, Medium popular, or Low popular. This will help Wolt in resource allocation and planning.

## Folder Structure
```bash
Predicting-Venue-Popularity/
│
├── Data/
│ ├── data_analysis.ipynb
│ ├── orders_autumn_2020.csv
│ ├── user_heatmap.html
│ └── user_heatmap.png
│
├── modeling.py
└── requirements.txt
```
### Prerequisites
Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
### Running the Model
The `modeling.py` script is the main file that performs the modelling, training, and evaluation. It accepts several arguments to customize the training process.

#### Example Commands
Run the following command in the command line to execute the modelling script with all models:
```bash
python modeling.py --data_path Data\orders_autumn_2020.csv --model_type all --user_loc True
```
To specify a model type and configure other parameters:
```bash
python modeling.py --data_path Data/orders_autumn_2020.csv --model_type mlp --user_loc True --learning_rate 0.01 --units_layers 500,250,500,500
```
```bash
python modeling.py --data_path Data/orders_autumn_2020.csv --model_type all --user_loc True
```
Many other hyper-parameters can be set. Here is another example of an ensemble model:
```bash
python modeling.py --data_path Data/orders_autumn_2020.csv --model_type ensemble --epochs 50 --degree 3 --n_estimators 50
```
Models available (model_type command): Multilayer Perceptron (mlp), Gradient Boosting Classifier (gb), Random Forest Classifier (forest), Naive Bayes (naive), Support Vector Classifier (svm), Ensemble model based on all (ensemble), All models run (all). When running 'all', a significance test will be executed to compare the models and check if there is a statistically significant difference.

## Data Analysis
The `data_analysis.ipynb` notebook within the `Data` folder includes a comprehensive exploratory data analysis (EDA) of the dataset. Key insights are drawn through various visualizations.

## Task Description
The task is to predict venue popularity based on geographic coordinates. The model also plots a popularity heatmap demonstrating areas that might be busy.
- This is a classification task. Based on order volume of each coordinate, the task is to classify the venue into High, Medium, or Low popularity category.
- Very close coordinates have been merged to one (round to three decimals).
- Adjust the `user_loc` argument to use user geographic coordinates as features to the model.
- Synthetic venue coordinates have been created to demonstrate model's generalization. It is expected that when these augmented points fall into 'popular' areas, will be classified into High category.  
- Two HTML files are created that demonstrate the actual and predicted popularity of the venues. The **actual_popularity** shows all the venues in the dataset. The venues that belong to the test set are also displayed with blue markes. The **predicted_popularity**, presents only the venues from the test set. Red markers indicate misclassification, green ones present the correct classifications. 
