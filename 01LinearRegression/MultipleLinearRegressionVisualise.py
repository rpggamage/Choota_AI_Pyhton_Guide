import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Data: [House Size (sq. ft), Number of Bedrooms, Age of House (years)]
X = np.array([
    [1500, 3, 10],
    [2000, 4, 15],
    [2500, 4, 5],
    [1800, 3, 8],
    [3000, 5, 20]
])

# House Prices (in USD)
y = np.array([300000, 400000, 450000, 350000, 500000])

# Model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the actual data points
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Actual Prices')

# Generate a meshgrid for the plane (reduce the number of points)
# Reduce points for faster rendering
x_range = np.linspace(min(X[:, 0]), max(X[:, 0]), 5)
# Reduce points for faster rendering
y_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 5)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# Evaluate the regression model on the meshgrid to generate Z values (predicted prices)
Z_grid = model.intercept_ + model.coef_[0] * X_grid + model.coef_[1] * Y_grid

# Plot the regression plane
ax.plot_surface(X_grid, Y_grid, Z_grid, color='red', alpha=0.5)

# Labels and title
ax.set_xlabel('House Size (sq. ft)')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price (USD)')
ax.set_title('3D Plot: Actual Prices vs. Regression Plane')

# Show plot
plt.legend()
plt.show()
