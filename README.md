##  Module 20 Challenge
* Project Name: Credit Risk Classification
* Submitted by:  Michael Jardinico
* Date Submitted: Feb 10, 2024

### Working Files
1. `credit_risk_classification.ipynb`
2. `/Resources/lending_data.csv`

### Overview of the Analysis
This project involves assessing credit risk, a crucial aspect for financial institutions when determining the likelihood of a borrower defaulting on a loan. The dataset contains information relevant to this assessment, such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, total debt, and a binary loan status indicating whether the loan was paid or defaulted. The goal of your analysis is to predict the loan status (0 or 1, where 1 might indicate default) based on the other variables.

### Variables to Predict
The variable to be predicted is the `loan_status`. This is a binary variable where a value of 0 may indicate the loan was repaid, and 1 a default. Understanding the distribution of the variable `value_counts` is crucial to determine the balance or imbalance between repaid loans and defaults, impacting the preparation of the data and model it. (Refer to the following `value_counts` results based on the dataset)
![loan_status result]()

    
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

    _Step 3:_ Evaluate the model’s performance by doing the following:
    - Calculate the accuracy score of the model.
    - Generate a confusion matrix.
    - Print the classification report.

    ![Classification Report 1](https://github.com/mjardinico/credit-risk-classification/blob/main/Resources/balanced_accuracy_score1.png)

    _Step 4:_ Answer the following question.
    __Question:__ How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

    __Predict a Logistic Regression Model with Resampled Training Data__    

    _Step 1:_ Use the `RandomOverSampler` module from the `imbalanced-learn` library to resample the data. Be sure to confirm that the labels have an equal number of data points

    _Step 2:_ Use the LogisticRegression classifier and the resampled data to fit the model and make predictions.

    _Step 3:_ Evaluate the model’s performance by doing the following:
    
    - Calculate the accuracy score of the model.
    - Generate a confusion matrix.
    - Print the classification report.

    ![Classification Report 2](https://github.com/mjardinico/credit-risk-classification/blob/main/Resources/classification_report2.png)





    ## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.