# American Express User Exit Prediction

This project aims to predict whether an American Express user will exit (close their account) using a deep learning model.

## Problem Statement

Predict whether an American Express user will exit using deep learning.

## Data Points

The dataset includes the following features:

-   Credit Score
-   Geography
-   Gender
-   Age
-   Customer Since
-   Current Account
-   Number of Products
-   UPI Enabled
-   Estimated Yearly Salary
-   Closed (Target Variable)

## Dataset

The dataset is stored in the same folder under the file name `Dataset_master.csv`. (Originally Excel, converted to CSV).

## Data Preprocessing

### Step 1: Importing Libraries

Before processing the data, required libraries such as pandas, numpy, and tensorflow should be imported.

### Step 2: Importing Dataset

Convert `.xlsx` files to `.csv` using the following code:

```python
import pandas as pd

df = pd.read_excel('Dataset_master.xlsx')
df.to_csv('Dataset_master.csv', index=False)
