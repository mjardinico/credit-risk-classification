##  Module 20 Challenge
* Project Name: Credit Risk Classification
* Submitted by:  Michael Jardinico
* Date Submitted: Feb 10, 2024

## Working Files
1. `credit_risk_classification.ipynb`
2. `/Resources/lending_data.csv`

## Overview of the Analysis
This project involves assessing credit risk, a crucial aspect for financial institutions when determining the likelihood of a borrower defaulting on a loan. The dataset contains information relevant to this assessment, such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, total debt, and a binary loan status indicating whether the loan was paid or defaulted. The goal of your analysis is to predict the loan status (0 or 1, where 1 might indicate default) based on the other variables.

## Variables to Predict
The variable to be predicted is the `loan_status`. This is a binary variable where a value of 0 may indicate the loan was repaid, and 1 a default. Understanding the distribution of the variable `value_counts` is crucial to determine the balance or imbalance between repaid loans and defaults, impacting the preparation of the data and model it. (Refer to the following `value_counts` results based on the dataset)

![loan_status result](https://github.com/mjardinico/credit-risk-classification/blob/main/Resources/loan_status.png)


## Stages of the Machine Learning Process for this analysis.

### 1. Data Splitting: 

_Step 1:_ Read the `lending_data.csv` data from the Resources folder into a Pandas DataFrame.

_Step 2:_ Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.

![X DataFramce](https://github.com/mjardinico/credit-risk-classification/blob/main/Resources/X_variable_dataframe.png)

_Step 3:_ Check the balance of the labels variable (y) by using the `value_counts` function.

_Step 4:_ Split the data into training and testing datasets by using `train_test_split`.

    
### 2. Model Creation, Testing and Evaluation:

_Step 1:_ Fit a logistic regression model by using the training data (X_train and y_train).

_Step 2:_ Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.

![Prediction vs Actual](https://github.com/mjardinico/credit-risk-classification/blob/main/Resources/prediction_actual1.png)


_Step 3:_ Evaluate the model’s performance by doing the following:
- Calculate the accuracy score of the model.
- Generate a confusion matrix.
- Print the classification report.


_Step 4:_ Answer the following question.
- Question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

### 3. Methods Used
- Data Splitting
- Logistic Regression including predictions
- Calculate Accuracy Score and create Confusion Matrix


### 4. Results

* Machine Learning Model 1 (Using Original Data):
  * Balanced Accuracy Score: 0.9443
    * This score is relatively high, indicating the model performs well in identifying recalls (Actual 1 and Actual 0)
  * Precision score (1): 0.87
    * Suggests that the model predicts 87% on a default loan
  * Precision score (0): 1.00
    * Suggests that the model predicts 100% on non-default loan
  * Recall for Actual 1 (default loans): 0.89
    * This indicates that the model is able to identify 89% of the actual defaults
  * Recall for Actual 0 (non-default loans): 1.00
    * This indicates that the model is able to identify 100% of the actual non-default loans

  ![Classification Report Using Original Data](https://github.com/mjardinico/credit-risk-classification/blob/main/Resources/balanced_accuracy_score1.png)


* Machine Learning Model 2 (Using Resampled Data):
  * Balanced Accuracy Score: 0.9952    
  * Precision score (1): 0.87
  * Precision score (0): 1.00
  * Recall for Actual 1: 1.00
  * Recall for Actual 0: 1.00

  ![Classification Report Using Resampled Data](https://github.com/mjardinico/credit-risk-classification/blob/main/Resources/balanced_accuracy_score2.png)


## Summary

In summary, the results of the two machine learning models, using the original data and the resampled data indicates that the latter (Machine Learning Model 2 using resampled data) performs much better and could significantly reduce financial risk and is recommended. This model not only improves the ability to identify defaults (with a recall of 1.00 for defaults) but also maintain a high precision, ensuring that identifying a loan as default is minimized.