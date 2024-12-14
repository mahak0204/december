import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the mtcars dataset
mtcars = sns.load_dataset("mpg").dropna()

# For this example, let's use 'horsepower' (x) to predict 'mpg' (y)
# Seaborn's "mpg" dataset has a similar structure to mtcars but different column names.
# We will rename the columns for clarity.
mtcars.rename(columns={"horsepower": "hp", "mpg": "mpg"}, inplace=True)

# Extract independent (X) and dependent (Y) variables
X = mtcars["hp"].values  # Horsepower
Y = mtcars["mpg"].values  # Miles per gallon

# Perform the least squares fit method
n = len(X)  # Number of data points

# Calculate the means of X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Calculate the terms for the slope (m) and intercept (b)
numerator = np.sum((X - mean_x) * (Y - mean_y))
denominator = np.sum((X - mean_x) ** 2)
m = numerator / denominator
b = mean_y - (m * mean_x)

# Print the coefficients
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

# Predict Y values using the model
def predict(x):
    return m * x + b

Y_pred = predict(X)

# Plot the data and regression line
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles per Gallon (mpg)')
plt.title('Linear Regression using Least Squares Method')
plt.legend()
plt.show()

# Calculate R-squared value
total_variance = np.sum((Y - mean_y) ** 2)
explained_variance = np.sum((Y_pred - mean_y) ** 2)
r_squared = explained_variance / total_variance
print(f"R-squared: {r_squared}")
