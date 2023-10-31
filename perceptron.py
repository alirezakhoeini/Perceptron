import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("normalized_train.csv")
validation_data = pd.read_csv("normalized_validation.csv")

X_train = train_data[['x1', 'x2']].values
y_train = train_data['labels'].values
X_val = validation_data[['x1', 'x2']].values
y_val = validation_data['labels'].values

# Define a function to train a perceptron model with polynomial features and return accuracy
def train_perceptron_with_poly_features(degree):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
    perceptron = Perceptron(max_iter=1000, tol=1e-3)
    perceptron.fit(X_train_poly, y_train)
    
    train_accuracy = accuracy_score(y_train, perceptron.predict(X_train_poly))
    val_accuracy = accuracy_score(y_val, perceptron.predict(X_val_poly))
    
    return train_accuracy, val_accuracy

# Function to calculate accuracy
degrees = [2, 3, 5, 10]

# Train perceptron models with different degrees and report accuracies
for degree in degrees:
    train_accuracy, val_accuracy = train_perceptron_with_poly_features(degree)
    print(f"Degree {degree} - Training Accuracy: {train_accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}")

# Plot the decision boundary for the model with the best validation accuracy (you can choose the best degree)
best_degree = degrees[np.argmax([train_perceptron_with_poly_features(degree)[1] for degree in degrees])]

poly = PolynomialFeatures(degree=best_degree)
X_train_poly = poly.fit_transform(X_train)

perceptron = Perceptron(max_iter=1000, tol=1e-3)
perceptron.fit(X_train_poly, y_train)

xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100),
                     np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100))

Z = perceptron.predict(poly.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral)
plt.title(f'Perceptron Decision Boundary with Degree {best_degree}')
plt.show()