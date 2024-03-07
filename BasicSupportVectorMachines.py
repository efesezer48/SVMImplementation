# Implementation of Basic Support Vector Machines (SVMs)

# Import the libraries
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

#Create a syntetic dataset for illustration purpose
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=42) # This function is particularly useful for generating a random n-class classification problem. 

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, = train_test_split(X,y, test_size= 0.2 , random_state =42)

#Create an SVM classifier
clf = svm.SVC(kernel='linear') #linear kernel for simplicity

#Train the SVM classifier
clf.fit(X_train, y_train)  #It adjusts the model's parameters to find the best linear boundary that separates the different classes based on the input features (X_train) and their corresponding labels (y_train).
# This process prepares the SVM classifier to make accurate predictions on new, unseen data.

#Make predictions on the test set
y_pred = clf.predict(X_test)

#Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy:{accuracy}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using Grid Search
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
print(f"Best hyperparameters: {grid_search.best_params_}")

# Train the SVM classifier with the best hyperparameters
clf = svm.SVC(kernel='rbf', C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])
clf.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_scaled)

# Evaluate the accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize the decision boundary
def plot_decision_boundary(clf, X, y):
    # Create a mesh grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict the class labels for each point in the mesh grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and training points
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()

# Visualize the decision boundary
plot_decision_boundary(clf, X_test_scaled, y_test)