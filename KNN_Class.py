import pandas as pd
import numpy as np

class K_NN:
    X_train = None
    y_train = None
    def __init__(self, K):
        self.K = K

    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, row1, row2):
        return np.linalg.norm(row1-row2)

    def predict(self, pred):
        # empty list to hold each prediction
        predictions = []
        for p in pred:
            # empty list to hold distances for a specific test row
            distances = []
            # for loop to iterate through every row in the training set
            for i, v in enumerate(self.X_train):
                # calculate the euclidean distance between a training row
                # and test row
                distance = self.euclidean_distance(v, p)
                # append the distance to the distances list
                distances.append([distance, i])
            # sort the distances from least to greatest
            sorted_distances = sorted(distances)
            # take only smallest k distances
            k_distances = sorted_distances[:self.K]
            # Get the predicted classification
            predict = [self.y_train[i[1]] for i in k_distances]
            # Get the most frequent predicted element from predict
            result = max(set(predict), key = predict.count)
            # Append the result to the predictions list
            predictions.append(result)
        # return the prediction
        return predictions

