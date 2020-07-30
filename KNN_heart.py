from KNN_Class import K_NN
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import csv
df = pd.read_csv("heart.csv")

# identify features and target
x = np.array(df.drop(columns=['target']))
y = np.array(df['target'])

# split data set into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# uncomment if you want a validation dataset
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

# uncomment if you want to normalize the data
# X_train = preprocessing.normalize(X_train)
# X_val = preprocessing.normalize(X_val)
# X_test = preprocessing.normalize(X_test)

# instantiate the standardize class
scaler = StandardScaler()

# fit transform on the training data
X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# transform on the test data
X_test = scaler.transform(X_test)


# for loop to determine what k value had the highest acc
# high_acc = 0
# high_k = 0
# for k in range(1,100):
    
#     model = K_NN(K=k)

#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)

#     if acc > high_acc:
#         high_acc = acc
#         high_k = k


# instantiate the class with k=7
model = K_NN(K=7)

# fit the data on the object
model.fit(X_train, y_train)

# predict all of x_test vectors
y_pred = model.predict(X_test)
# test the accuracy of the model
acc = accuracy_score(y_test, y_pred)
# print the accuracy score
print(f"By hand accuracy: {acc}")
# print(high_k)

# sklearn implementation
from sklearn.neighbors import KNeighborsClassifier
# instantiate sklearn KNN class
neigh = KNeighborsClassifier(n_neighbors=7)
# fit the data
neigh.fit(X_train, y_train)
# predict all of X_test data
sklearn_pred = neigh.predict(X_test)
# print out the results of the predictions
print(f"Sklearn model accuracy: {accuracy_score(y_test, sklearn_pred)}")

