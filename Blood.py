# 1. Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# 2. Load and Preprocess the Dataset

# Define the column names
col_names = [
    'Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape',
    'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 
    'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class'
]

# Load the dataset
file_path = r"C:\Users\VICTUS\Desktop\class (data science)\28th - KNN\28th - KNN\projects\KNN\brest cancer.txt"
df = pd.read_csv(file_path, header=None, names=col_names)

# Display the first few rows of the dataset
df.head()

# 3. Handle Missing Data

# Check for missing values
df.isnull().sum()

# Replace '?' with NaN in the 'Bare_Nuclei' column
df['Bare_Nuclei'].replace('?', pd.NA, inplace=True)

# Convert 'Bare_Nuclei' to numeric
df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'])

# Replace missing values (NaN) with the median of the column
median_value = df['Bare_Nuclei'].median()
df['Bare_Nuclei'].fillna(median_value, inplace=True)

# Check if the column has been cleaned properly
print(df['Bare_Nuclei'].unique())

# 4. Split the Data into Features and Target

# Define features (X) and target (y)
X = df.drop(columns=['Id', 'Class'])  # Features
y = df['Class']  # Target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 5. Initialize Models

# Initialize the models: SVM, KNN, and Logistic Regression
svm_model = SVC()
knn_model = KNeighborsClassifier()
logreg_model = LogisticRegression(max_iter=1000)

# 6. Train the Models

# Train SVM
svm_model.fit(X_train, y_train)

# Train KNN
knn_model.fit(X_train, y_train)

# Train Logistic Regression
logreg_model.fit(X_train, y_train)

# 7. Make Predictions

# Make predictions using SVM
svm_pred = svm_model.predict(X_test)

# Make predictions using KNN
knn_pred = knn_model.predict(X_test)

# Make predictions using Logistic Regression
logreg_pred = logreg_model.predict(X_test)

# 8. Define Evaluation Function

# Function to evaluate models
def evaluate_model(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    
    # Since the labels are 2 and 4, we specify pos_label as 4 (malignant) or 2 (benign)
    precision = precision_score(y_test, y_pred, pos_label=4)
    recall = recall_score(y_test, y_pred, pos_label=4)
    
    print(f'{model_name} Model Evaluation:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision (for class 4 - malignant): {precision:.4f}')
    print(f'Recall (for class 4 - malignant): {recall:.4f}\n')
    print(classification_report(y_test, y_pred, labels=[2, 4]))

# 9. Evaluate Models

# Evaluate SVM
evaluate_model(y_test, svm_pred, "SVM")

# Evaluate KNN
evaluate_model(y_test, knn_pred, "KNN")

# Evaluate Logistic Regression
evaluate_model(y_test, logreg_pred, "Logistic Regression")

# 10. Save the Trained Models Together

# Create a dictionary to hold the models
models = {
    'svm_model': svm_model,
    'knn_model': knn_model,
    'logreg_model': logreg_model
}

# Save the trained models to disk
import pickle
filename = 'combined_models.pkl'
with open(filename, 'wb') as file:
    pickle.dump(models, file)
print(f"Models have been pickled and saved as combined_models.pkl")

# Check current working directory
import os
print("Current working directory:", os.getcwd())