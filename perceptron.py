import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np

# Load the training data
train_data = pd.read_csv("train.csv")

# Randomly select 50 data points as validation data
validation_data = train_data.sample(n=50, random_state=42)  # You can change the random_state for reproducibility

# Remove the selected validation data from the training data
train_data = train_data.drop(validation_data.index)

# Save the modified training data and validation data to separate CSV files
train_data.to_csv("modified_train.csv", index=False)
validation_data.to_csv("validation.csv", index=False)


# Load the modified training data and validation data
train_data = pd.read_csv("modified_train.csv")
validation_data = pd.read_csv("validation.csv")

# Define a function for Min-Max scaling
def min_max_scaling(data):
    return (data - data.min()) / (data.max() - data.min())

def z_score_scaling(data):
    return (data - data.mean()) / data.std()

# Apply Min-Max scaling to the x1 and x2 columns
train_data['x1'] = min_max_scaling(train_data['x1'])
train_data['x2'] = min_max_scaling(train_data['x2'])
validation_data['x1'] = min_max_scaling(validation_data['x1'])
validation_data['x2'] = min_max_scaling(validation_data['x2'])

# Save the normalized data to new CSV files
train_data.to_csv("normalized_train.csv", index=False)
validation_data.to_csv("normalized_validation.csv", index=False)

class_0 = train_data[train_data['labels'] == 0]
class_1 = train_data[train_data['labels'] == 1]

# Create two separate plots for the two classes
plt.figure(figsize=(10, 5))

# Plot data points for class 0 in blue
plt.scatter(class_0['x1'], class_0['x2'], c='blue', label='Class 0', marker='o')

# Plot data points for class 1 in red
plt.scatter(class_1['x1'], class_1['x2'], c='red', label='Class 1', marker='x')

# Add labels and legend
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of Normalized Data')
plt.legend()

# # Show the plots
# plt.show()

# Extract features (x1 and x2) and labels from the data
X_train = train_data[['x1', 'x2']].values
y_train = train_data['labels'].values
X_val = validation_data[['x1', 'x2']].values
y_val = validation_data['labels'].values

# Perceptron algorithm implementation
class Perceptron:
    def __init__(self, num_features, learning_rate=0.1, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = np.zeros(num_features + 1)  # Include the bias term

    def train(self, X, y):
        for _ in range(self.max_iterations):
            misclassified = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                misclassified += int(update != 0)
            if misclassified == 0:
                break

    def predict(self, xi):
        activation = np.dot(xi, self.weights[1:]) + self.weights[0]
        return 1 if activation >= 0 else 0

# Initialize and train the perceptron
perceptron = Perceptron(num_features=2)
perceptron.train(X_train, y_train)

# Function to calculate accuracy
def accuracy(X, y, model):
    predictions = [model.predict(xi) for xi in X]
    return np.mean(predictions == y)

# Calculate accuracy for training and validation data
train_accuracy = accuracy(X_train, y_train, perceptron)
validation_accuracy = accuracy(X_val, y_val, perceptron)

# Plot the data points and decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=plt.cm.Spectral, marker='o', edgecolor='k')
plt.title(f'Perceptron Decision Boundary\nTraining Accuracy: {train_accuracy:.2f}, Validation Accuracy: {validation_accuracy:.2f}')

# Define the decision boundary line
x_decision_boundary = np.linspace(X_val[:, 0].min(), X_val[:, 0].max())
y_decision_boundary = -(perceptron.weights[1] * x_decision_boundary + perceptron.weights[0]) / perceptron.weights[2]

# Plot the decision boundary
plt.plot(x_decision_boundary, y_decision_boundary, 'k-')

# Show the plot
plt.show()

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Validation Accuracy: {validation_accuracy:.2f}")