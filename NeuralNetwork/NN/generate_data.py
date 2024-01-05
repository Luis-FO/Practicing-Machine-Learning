import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances

# Function to generate 2D points labeled as groups A and B
def generate_data(test_sz = 0.30):
    # Generate data with two groups
    X, y = make_blobs(n_samples=10000, centers=2, random_state=42, cluster_std=1.5)
    #X = normalize_vector(X)
    min = np.min(X)
    max = np.max(X) 
    X = ((2*(X - min))/(max-min))-1
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sz, random_state=42)
    return X_train, X_test, y_train, y_test


def one_hot_encode(labels, num_classes):
    """
    Perform one-hot encoding on the given list of labels.

    Parameters:
    - labels: List of labels to be encoded.
    - num_classes: Number of classes for one-hot encoding.

    Returns:
    - One-hot encoded representation of the labels.
    """
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i, label] = 1
    return one_hot_labels

# Function to visualize the generated points
def plot_data(X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Function to calculate the average intra-group distance
def average_intra_group_distance(X, y):
    distances = euclidean_distances(X)
    intra_group_distances = []

    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if y[i] == y[j]:
                intra_group_distances.append(distances[i, j])

    return np.mean(intra_group_distances)

if __name__ == "__main__":

    # Generate the data
    X_train, X_test, y_train, y_test = generate_data()
    print(one_hot_encode(y_train, 2))
    # Visualize the training data
    plot_data(X_train, y_train, 'Training Data')

    # Calculate the average intra-group distance in the training data
    avg_distance_train = average_intra_group_distance(X_train, y_train)
    print(f"Average intra-group distance in training data: {avg_distance_train:.2f}")

    # Visualize the testing data
    plot_data(X_test, y_test, 'Testing Data')

    # Calculate the average intra-group distance in the testing data
    avg_distance_test = average_intra_group_distance(X_test, y_test)
    print(f"Average intra-group distance in testing data: {avg_distance_test:.2f}")
