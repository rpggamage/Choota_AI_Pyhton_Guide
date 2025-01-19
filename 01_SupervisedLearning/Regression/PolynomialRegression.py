import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Dataset: Experience (years) and Salary (USD)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([20000, 25000, 40000, 45000, 50000,
             60000, 80000, 110000, 150000, 200000])

# Transform the feature to include polynomial terms (degree 2)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Train a Linear Regression model on the polynomial features
model = LinearRegression()
model.fit(X_poly, y)

# Predict salaries
# Generate more points for a smooth curve
X_range = np.linspace(1, 10, 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_pred = model.predict(X_range_poly)

# Visualize the results
plt.scatter(X, y, color='blue', label='Data')  # Original data points
plt.plot(X_range, y_pred, color='red',
         label='Polynomial Regression')  # Fitted curve
plt.xlabel('Experience (Years)')
plt.ylabel('Salary (USD)')
plt.title('Polynomial Regression: Experience vs. Salary')
plt.legend()
plt.show()

# Print coefficients and example prediction
print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)

# Predict salary for someone with 6.5 years of experience
experience = np.array([[6.5]])
experience_poly = poly.transform(experience)
predicted_salary = model.predict(experience_poly)
print(f"Predicted salary for 6.5 years of experience: ${
      predicted_salary[0]:,.2f}")
