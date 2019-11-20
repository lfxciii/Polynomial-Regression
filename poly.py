# salary prediction for new employees based on current employee levels at company


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
x_train = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y_train = [[4.5], [5.3], [6.0], [6.7], [11.6], [15.3], [20.5], [30.2], [50.5], [100.8]]               

# Testing set
x_test = [[2], [4], [5], [8]] 
y_test = [[5.5], [8.9], [9.2], [35.3]]

# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(0, 10, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
X_train_quadratic = quadratic_featurizer.fit_transform(x_train)
X_test_quadratic = quadratic_featurizer.transform(x_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Salary regressed on employee level')
plt.xlabel('Employee level (1 - 10)')
plt.ylabel('Employee Salary x1000')
plt.axis([0, 11, 0, 100])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()
print (x_train)
print (X_train_quadratic)
print (x_test)
print (X_test_quadratic)