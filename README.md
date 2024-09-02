# First Machine Learning Model

This repository contains my first steps into the world of Machine Learning. It demonstrates the implementation of basic machine learning algorithms, data exploration, model training, and evaluation.

## Overview

This project serves as an introduction to the fundamentals of machine learning. It involves building a simple model using a standard dataset, processing the data, training the model, and evaluating its performance. The goal is to provide a clear, hands-on example for beginners in machine learning.

## Features

- **Data Exploration:** Analyze and visualize the dataset to understand its structure and key characteristics.
- **Model Training:** Implement and train a machine learning model using a popular algorithm.
- **Model Evaluation:** Assess the model's performance using metrics like accuracy, precision, recall, and F1-score.
- **Documentation:** Clear and concise documentation to guide beginners through the process of building a machine learning model.


### Code
```
import pandas as pd
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *
print("Setup Complete")

# Step 1: Specify Prediction Target

# print the list of columns in the dataset to find the name of the prediction target
print(home_data.columns)

y = home_data.SalePrice

# Step 2: Create X
# Create the list of features below
feature_names = ["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]

# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Review data
# print description or statistics from X
print(X.describe())

# print the top few lines
print(X.head())

# Step 3: Specify and Fit Model
from sklearn.tree import DecisionTreeRegressor
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state = 42)

# Fit the model
iowa_model.fit(X,y)

# Step 4: Make Predictions
predictions = iowa_model.predict(X)
print(predictions)

```


