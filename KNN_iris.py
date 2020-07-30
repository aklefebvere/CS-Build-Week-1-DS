import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from KNN_Class import K_NN


# import the iris dataset
df = pd.read_csv("iris.csv", header=None)

# identify our features and target
x = np.array(df.drop(columns=[4]))
y = np.array(df[4])

# create a train and test dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# instatiate our class object with k=3
model = K_NN(K=3)
# fit our data onto the model
model.fit(X_train, y_train)
# create our prediction
y_pred = model.predict(X_test)

# print out the result of our prediction
print(f"By hand accuracy: {accuracy_score(y_test, y_pred)}")


# Using the sklearn KNN class
from sklearn.neighbors import KNeighborsClassifier
# instantiate the sklearn KNN class
neigh = KNeighborsClassifier(n_neighbors=3)
# fit the data onto the model
neigh.fit(X_train, y_train)
# create the prediction
sklearn_pred = neigh.predict(X_test)
# print out the accuracy of the predictions compared to the actual labels
print(f"Sklearn model accuracy: {accuracy_score(y_test, sklearn_pred)}")