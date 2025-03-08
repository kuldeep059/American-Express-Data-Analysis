# American Express User Exit Prediction

Problem Statement

Predict whether an American Express user will exit using deep learning.

Data Points

The dataset includes the following features:

Credit Score

Geography

Gender

Age

Customer Since

Current Account

Number of Products

UPI Enabled

Estimated Yearly Salary

Closed (Target Variable)

Dataset

The dataset is stored in the same folder under the file name Dataset_master.

Data Preprocessing

Step 1: Importing Libraries

Before processing the data, required libraries such as pandas, numpy, and tensorflow should be imported.

Step 2: Importing Dataset

Convert .xlsx files to .csv using the following code:

import pandas as pd
df = pd.read_excel('file_name_Excel_to_Convert.xlsx')
df.to_csv('File_name.csv', index=False)

Download the converted file if working in Google Colab:

from google.colab import files
files.download('File_name.csv')

Step 3: Handling Missing Data

Identify missing values and impute or remove them accordingly.

Step 4: Encoding Categorical Data

Convert categorical variables into numerical representations using encoding techniques like One-Hot Encoding or Label Encoding.

Step 5: Splitting Data into Training and Test Sets

Divide the dataset into training and test sets using train_test_split from sklearn.

Step 6: Feature Scaling

Apply feature scaling to normalize numerical values using StandardScaler.

Artificial Neural Network (ANN) Implementation

Step 1: ANN Initialization

Initialize the ANN model using tensorflow.keras.models.Sequential().

Step 2: Adding Input Layer and First Hidden Layer

Define input features and add the first hidden layer using Dense from tensorflow.keras.layers.

Step 3: Adding Second Hidden Layer

Add another hidden layer with activation functions like ReLU.

Step 4: Adding Output Layer

Define the output layer using the appropriate activation function (e.g., sigmoid for binary classification).

Step 5: Compiling ANN

Compile the ANN using an optimizer (e.g., Adam) and loss function (e.g., binary cross-entropy).

Step 6: Training on Dataset

Train the model using fit() method with training data.

Step 7: Predictions

Use the trained model to make predictions on test data.

Step 8: Confusion Matrix

Evaluate model performance using a confusion matrix to measure accuracy.

Requirements

Python

TensorFlow

Pandas

NumPy

Scikit-learn

Matplotlib (optional for visualization)

Usage

Clone the repository

git clone https://github.com/yourusername/your-repo-name.git

Install required dependencies

pip install -r requirements.txt

Run the preprocessing and training scripts

python train.py

License

This project is open-source and available under the MIT License.

