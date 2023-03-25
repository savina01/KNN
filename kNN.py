import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, X):
        predictions = []
        for sample in X:
            distances = []
            for i, train_sample in enumerate(self.X):
                distance = np.sqrt(np.sum((sample - train_sample)**2))
                distances.append((distance, self.y[i]))
            distances.sort()
            top_k = distances[:self.k]
            classes = {}
            for distance, label in top_k:
                if label in classes:
                    classes[label] += 1
                else:
                    classes[label] = 1
            predictions.append(max(classes, key=classes.get))
        return np.array(predictions)

# create a KNN object with k=3
knn = KNN(k=3)

# train the KNN model on some data X and labels y
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 1, 0])
knn.fit(X, y)

# use the KNN model to predict labels for some test data
test_data = np.array([[2, 3], [4, 5], [6, 7], [8, 9]])
predicted_labels = knn.predict(test_data)
print(predicted_labels)
