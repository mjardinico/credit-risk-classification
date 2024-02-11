##  Module 20 Challenge
* Project Name: Credit Risk Classification
* Submitted by:  Michael Jardinico
* Date Submitted: Feb 10, 2024

### Project Overview
`This project uses various techniques to train and evaluate a model based on a loan risk. Utilizing historical data from a peer-to-peer lending company's lending activities, the goal is to construct a model capable of determining borrowers' creditworthiness. This README outlines the steps to set up and execute the project.`

### Working Files
1. `credit_risk_classification.ipynb`
2. `/Resources/lending_data.csv`
    
### Instructions
1. __GitHub Repository Setup:__
    - Create a Repository: Start by creating a GitHub repository named `credit-risk-classification`.
    - Clone to Local: Clone the repository to your local computer using Git.

2. __Data Preparation__
`Import the necessary libraries and modules for data analysis:`
    - import numpy as np
    - import pandas as pd
    - from pathlib import Path
    - from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
   
3. __Data Processing__  

    __Split the Data into Training and Testing Sets:__ 
    _Step 1:_ Read the `lending_data.csv` data from the Resources folder into a Pandas DataFrame.

    _Step 2:_ Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.
    ![X DataFramce](https://github.com/mjardinico/credit-risk-classification/blob/main/Resources/X_variable_dataframe.png)

    _Step 3:_ Check the balance of the labels variable (y) by using the `value_counts` function.

    _Step 4:_ Split the data into training and testing datasets by using `train_test_split`.

    __Create a Logistic Regression Model with the Original Data__
    _Step 1:_ Fit a logistic regression model by using the training data (X_train and y_train).

    _Step 2:_ Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.
    ![Prediction vs Actual](https://github.com/mjardinico/credit-risk-classification/blob/main/Resources/prediction_actual1.png)

    - __Create Scaled DataFrame:__ Construct a DataFrame with the scaled data, ensuring the "coin-id" is set as the index.
    ![Crypto Currency DataFrame](https://github.com/mjardinico/CryptoClustering/blob/main/Resources/crypto_dataframe1.png) 
    
    - __Elbow Curve:__ Generate an elbow curve to identify the optimal value of k. 
    ![Elbow Curve](https://github.com/mjardinico/CryptoClustering/blob/main/Resources/elbow_curve.png)

    - __Principal Component Analysis (PCA):__ Optimize clusters using PCA.
    ![PCS DataFrame](https://github.com/mjardinico/CryptoClustering/blob/main/Resources/PCA_DataFrame.png)

    - __Cluster Scatter Plot:__ Create a scatter plot using `hvplot` using `KMeans`
    ![PCA Scatter Plot](https://github.com/mjardinico/CryptoClustering/blob/main/Resources/cluster_scatterplot1.png)


`NOTE: Rendering of interactive plots on GitHub is not supported. Use Jupyter Notebook to view interactive plots correctly.`