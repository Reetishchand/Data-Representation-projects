import sys
import numpy as np
from numpy.linalg import eig

# if  len(sys.argv) != 3 :
#     print('input format must be : python3 pca1.py <input>.data <output>.data')
#     sys.exit()

def uncenteredPCA(matrix, r):
    eval,evec = np.linalg.eigh(matrix @ matrix.T)
    indexSorted = np.argsort(-eval)
    eval = eval[indexSorted]
    evec = evec[:,  indexSorted]
    t = evec[:,:r]
    op = t.T @ matrix
    return op

#input, output = sys.argv[1], sys.argv[2]
#inputData = np.loadtxt(input, delimiter=',')
npInput = np.matrix("1 2 3 4 5; 5 4 3 2 1;1 2 3 2 1")
n = 3
outputData =uncenteredPCA(npInput.T, n)
print(outputData)
#np.savetxt(output, outputData.T, delimiter=',', fmt='%.2f')



