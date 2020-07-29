from KNN_Class import K_NN
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("heart.csv")


x = np.array(df.drop(columns=['target']))
y = np.array(df['target'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

# X_train = preprocessing.normalize(X_train)
# X_val = preprocessing.normalize(X_val)
# X_test = preprocessing.normalize(X_test)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# print(X_train)


high_acc = 0
high_k = 0
# for k in range(1,100):
    
#     model = K_NN(K=k)

#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)

#     if acc > high_acc:
#         high_acc = acc
#         high_k = k



model = K_NN(K=7)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)


print(f"By hand accuracy: {acc}")
# print(high_k)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X_train, y_train)
sklearn_pred = neigh.predict(X_test)
print(f"Sklearn model accuracy: {accuracy_score(y_test, sklearn_pred)}")

