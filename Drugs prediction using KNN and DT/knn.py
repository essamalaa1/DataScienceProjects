# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
data = pd.read_csv(r"C:\Users\Essam\Desktop\Assignment 2\diabetes.csv")

# %%
# Function to normalize the dataset
def normalize_dataset(dataset):
    mins = dataset.iloc[:, :-1].min()
    maxs = dataset.iloc[:, :-1].max()
    dataset.iloc[:, :-1] = (dataset.iloc[:, :-1] - mins) / (maxs - mins)
    return dataset

# %%
# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# %%
# Function to perform KNN classification
def knn_classify(training_data, test_instance, k):
    distances = []
    for i, row in enumerate(training_data):
        distance = euclidean_distance(row[:-1], test_instance[:-1])
        distances.append((i, distance))

    distances.sort(key=lambda x: x[1])
    k_nearest_neighbors = distances[:k]


    votes = {}
    for neighbor_index, distance in k_nearest_neighbors:
        neighbor_class = training_data[neighbor_index, -1]
        if neighbor_class in votes:
            votes[neighbor_class] += 1 / distance 
        else:
            votes[neighbor_class] = 1 / distance
    return max(votes, key=votes.get)

# %%
# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# %%
# Normalize the training and testing datasets
train_data = normalize_dataset(train_data)
test_data = normalize_dataset(test_data)
train_data=np.array(train_data)
test_data=np.array(test_data)

# %%
k_values = [2, 3, 4, 5]
accuracies = []

for k in k_values:
    correct_classifications = 0
    for i in range(len(test_data)):
        test_instance = test_data[i, :]
        predicted_class = knn_classify(train_data, test_instance, k)
        actual_class = test_instance[-1]
        if predicted_class == actual_class:
            correct_classifications += 1
    total_instances = len(test_data)
    accuracy = (correct_classifications / total_instances) * 100
    accuracies.append(accuracy)
    print(f'k value: {k}')
    print(f'Number of correctly classified instances: {correct_classifications}')
    print(f'Total number of instances: {total_instances}')
    print(f'Accuracy: {accuracy:.2f}%')
    print()
average_accuracy = sum(accuracies) / len(accuracies)
print(f'Average Accuracy Across All k Values: {average_accuracy:.2f}%')


