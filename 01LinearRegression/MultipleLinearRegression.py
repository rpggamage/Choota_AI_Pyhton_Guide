import numpy as np
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

# Prediction: Predict the price of a 2200 sq. ft house with 4 bedrooms, 10 years old
house_to_predict = np.array([[2200, 4, 10]])
predicted_price = model.predict(house_to_predict)

print(f"Predicted price: ${predicted_price[0]:,.2f}")

# Example Coefficients (Optional)
print("Model Coefficients (Weights):", model.coef_)
print("Model Intercept:", model.intercept_)
