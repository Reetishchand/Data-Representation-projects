import sys
import numpy as np
from numpy.linalg import eig

# if len(sys.argv) != 4  :
#     print('input format must be : python3 mds.py <input>.txt <output>.txt alpha')
#     sys.exit()

def calculateDistance(M, x):
    n,m = M.shape
    T=[]
    for i in range(n):
        T.append([0]*n)
    T= np.matrix(T)
    for i in range(n):
        for j in range(n):
            T[i,j] = np.sum(np.square(M[i] - M[j])) #** (x * 0.5)
    return T

def calculateGram(M):
    n,m = M.shape
    d=[]
    for i in range(n):
        d.append([0]*n)
    d= np.matrix(d).astype(float)
    temp=[]
    for i in range(n):
        temp.append(np.sum(M[i]))
    x= np.sum(temp)
    for i in range(n):
        for j in range(n):
            d[i,j] = (-M[i, j] / 2 + ((temp[i] + temp[j]) / (2 * n)) - (x / (2 * (n ** 2))))
    return d



# inputData, outputData, alpha = sys.argv[1], sys.argv[2], float(sys.argv[3])
# input = np.loadtxt(inputData, delimiter=',')
M = np.matrix("0 4 3;4 0 5;3 5 0")
alpha = 1.4894865645321
X = calculateDistance(M, alpha)
X = calculateGram(X)
print("X = ",X)
eigValues, eigVectors = eig(X)
sortedValues = np.argsort(-eigValues)
eigValues = eigValues[sortedValues]
eigVectors = eigVectors[:, sortedValues]
print("--------------------------------------")
print(eigValues)
print("--------------------------------------")

eigValues[eigValues < 0] = 0
dim = 2
z=np.zeros(shape=(dim, X.shape[0]))
for i in range(dim):
    val,vec = eigValues[i], eigVectors[i]
    z[i]= np.sqrt(val) * vec
print(z.T)
# np.savetxt(outputData, z.T, delimiter=',', fmt='%.4f')
