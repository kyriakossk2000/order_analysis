# Predicting Order Volume for Wolt

## Project Overview
This project focuses on forecasting the number of orders Wolt might receive in the coming days. It aims to utilize historical order data to predict future order volumes, helping in resource allocation and planning.

## Folder Structure
```bash
Predicting-Order-Volume/
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
Run the following command in the command line to execute the modelling script with the default parameters:
```bash
python modeling.py --data_path Data/orders_autumn_2020.csv --model_type sarima
```
To specify a model type and configure other parameters:
```bash
python modeling.py --data_path Data/orders_autumn_2020.csv --model_type all --future_days 14 --past_days 50
```

## Data Analysis
The `data_analysis.ipynb` notebook within the `Data` folder includes a comprehensive exploratory data analysis (EDA) of the dataset. Key insights are drawn through various visualizations.

## Task Description
The task is to predict the daily order volume for the next `X` days (`future_days` argument) based on historical data. The model also plots the historical order volumes for the last `Y` days (`past_days` argument) alongside the predictions.
- Adjust the `future_days` argument to set the forecast horizon.
- Adjust the `past_days` argument to determine the number of historical days to plot.
- This is a time-series forecasting analysis. 
