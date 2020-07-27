import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from KNN_Class import K_NN



df = pd.read_csv("iris.csv", header=None)

x = np.array(df.drop(columns=[4]))
y = np.array(df[4])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


model = K_NN(K=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print(f"By hand accuracy: {accuracy_score(y_test, y_pred)}")


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
sklearn_pred = neigh.predict(X_test)
print(f"Sklearn model accuracy: {accuracy_score(y_test, sklearn_pred)}")