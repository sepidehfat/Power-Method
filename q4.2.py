import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def largest_eigenvector(Matrix):
    v = np.ones(shape=[4, 1])
    v = v / np.linalg.norm(v)
    largest_lambda = 0
    for i in range(10):
        v = np.dot(Matrix, v)
        largest_lambda = np.linalg.norm(v)
        v = v / largest_lambda
    return largest_lambda, v
#%%
data = pd.read_csv('iris.data', header=None)
data = data.sample(frac=1).reset_index(drop=True)
data[4] = data[4].replace({"Iris-virginica": 0, "Iris-versicolor": 1, "Iris-setosa": 2})
X = data.loc[:, 0:3].to_numpy()
y = data.loc[:, 4].to_numpy()
#%%
A = np.cov(X.T)
largest_lambda, v1 = largest_eigenvector(A)
#%%
B = A - np.eye(4)*largest_lambda
second_largest_lambda, v2 = largest_eigenvector(B)
#%%
v = np.concatenate((v1, v2), axis=1)
Z = np.zeros(shape=(X.shape[0],  2))
Z = np.dot(X, v)
#%% plot data
plt.figure()
plt.scatter(Z[:, 0], Z[:, 1], c=y)
plt.show()