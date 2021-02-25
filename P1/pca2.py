import sys
import numpy as np
from numpy.linalg import eig


if len(sys.argv) != 2 :
    print('input format must be : python3 pca2.py <input>.data <output>.data')
    sys.exit()


def centeredPCA(matrix, r):
    mean = np.mean(matrix, axis =1). reshape ((-1, 1))
    matrix = matrix - mean
    eval,evec = np.linalg.eigh(matrix @ matrix.T)
    indexSorted = np.argsort(-eval)
    eval = eval[indexSorted]
    evec = evec[:,  indexSorted]
    t = evec[:,:r]
    op = t.T @ matrix
    return op



input, output = sys.argv[1], sys.argv[2]
inputData = np.loadtxt(input, delimiter=',')
npInput = np.matrix(inputData)
n = 2
outputData =centeredPCA(npInput.T, n)
np.savetxt(output, outputData.T, delimiter=',', fmt='%.2f')