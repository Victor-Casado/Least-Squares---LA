import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv("insurance.csv")

# Select features and target variable
A = data[['age', 'bmi', 'children', 'smoker']].copy()

# Convert categorical 'smoker' variable into numeric (1 = smoker, 0 = non-smoker)
A['smoker'] = A['smoker'].map({'yes': 1, 'no': 0})

# Add a column of ones to A for the intercept (beta_0)
A.insert(0, 'intercept', 1)

# Convert A to a NumPy array with explicit float type
A = A.to_numpy(dtype=np.float64)  # Ensures all values are float

# Extract the dependent variable and ensure it's also float64
b = data['charges'].to_numpy(dtype=np.float64)

# Solve for x using the normal equation
x = np.linalg.inv(A.T @ A) @ A.T @ b  # Solve for coefficients

# Display the coefficients
intercept = x[0]
coefficients = x[1:]

print("Intercept:", intercept)
print("Coefficients:", coefficients)
