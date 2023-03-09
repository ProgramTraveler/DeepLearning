
from mxnet import nd

x = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

k = nd.array([[0, 1], [2, 3]])

y = nd.zeros(shape=(2, 2))

h, w = k.shape

for i in range(y.shape[0]):
    for j in range(y.shape[1]):
        y[i, j] = (x[i : i + h, j : j + w] * k).sum()
        
print(y)

y_tem = nd.zeros(shape=(2, 2))
k_tem = nd.array([[3, 2], [1, 0]])

for i in range(y.shape[0]):
    for j in range(y.shape[1]):
        y_tem[i, j] = (x[i : i + h, j : j + w] * k_tem).sum()
        
print(y_tem)
print(y[:] == y_tem[:])