import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data
x = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)  # House sizes
y = np.array([150000, 200000, 250000, 300000, 350000])      # Prices

# Model
model = LinearRegression()
model.fit(x, y)

# Prediction
size_to_predict = np.array([[2200]])  # Example house size
predicted_price = model.predict(size_to_predict)

print(f"Predicted price for a 2200 sq. ft house: ${predicted_price[0]:,.2f}")

# Visualization
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, model.predict(x), color='red', label='Fitted Line')
plt.scatter(size_to_predict, predicted_price,
            color='green', label='Prediction')
plt.xlabel('House Size (sq. ft)')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
