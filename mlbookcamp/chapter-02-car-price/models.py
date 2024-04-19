from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_linear_regression(X_train, y_train):
    """
    Trains a linear regression model using scikit-learn.

    Parameters:
    X_train (array-like): The training input samples.
    y_train (array-like): The target values.

    Returns:
    model: The trained linear regression model.
    """

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a linear regression model using scikit-learn.

    Parameters:
    model: The trained linear regression model.
    X_test (array-like): The test input samples.
    y_test (array-like): The true target values.

    Returns:
    error: The root mean squared error of the model's predictions.
    """

    y_pred = model.predict(X_test)
    error = mean_squared_error(y_test, y_pred)/len(y_test)
    return error

