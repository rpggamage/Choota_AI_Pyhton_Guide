import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer

# Load the iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Taking only the first two features for 2D plot
y = iris.target

# Create a logistic regression model with multiclass option
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X, y)

# Create a mesh grid for decision boundaries
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 100))

# Predict on the mesh grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting the decision boundaries and data points
plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')

# Plot data points
scatter = plt.scatter(X[:, 0], X[:, 1], c=y,
                      edgecolors='k', marker='o', cmap='coolwarm')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Multiclass Classification (Logistic Regression)')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.colorbar()
plt.show()
