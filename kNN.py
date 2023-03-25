import numpy as np

class KNN:
    # Инициализира KNN обекта със стойност за k,
    # която указва броя на съседите, които да се използват за всяка прогноза.
    def __init__(self, k):
        self.k = k

    # Взема данните за обучение X и етикетира y и ги съхранява в обекта за по-късна употреба.    
    def fit(self, X, y):
        self.X = X
        self.y = y

    # Взема някои тестови данни X и прави прогнози за етикетите въз основа на k-най-близките
    # съседи в данните за обучение.
    # За всяка тестова проба в X ние изчисляваме разстоянията до всички обучителни проби,
    # използвайки евклидово разстояние.
    # След това сортираме разстоянията във възходящ ред и избираме първите k най-близки съседи.
    # Преброяваме броя на съседите, които принадлежат към всеки клас и 
    # избираме етикета на класа на мнозинството като предвиден етикет за тестовата проба.
    # Добавяме предвидения етикет към списъка с прогнози за всички тестови проби.
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
            # Накрая връщаме масива от предвидени етикети за всички тестови проби.
        return np.array(predictions)

knn = KNN(k=3)

# Обучаваме модела върху данните и етикета.
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 1, 0])
knn.fit(X, y)

# Използвайки модела за прогнозиране на етикети за някои данни.
test_data = np.array([[2, 3], [4, 5], [6, 7], [8, 9]])
predicted_labels = knn.predict(test_data)
print(predicted_labels)
