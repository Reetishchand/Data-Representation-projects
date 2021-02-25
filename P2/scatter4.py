import sys
import numpy as np
from numpy.linalg import eig

def getMatrixTranspose(matrix):
    return matrix.T

def getFinalMatrix(matrix1,matrix2):
    return np.dot(matrix1[:2, ], getMatrixTranspose(matrix2))

def getEigenProperties(matrix):
    eigenValues, eigenVectors = np.linalg.eigh(matrix)
    rankedIndexes = np.argsort(-eigenValues)
    eigenValues = eigenValues[rankedIndexes]
    eigenVectors = eigenVectors[:, rankedIndexes]
    return eigenValues, eigenVectors

def caluclateNetAverage(a,b):
    return np.sum(a, axis=0) / b

def initalize(colLen):
    return np.zeros((colLen, colLen))


if len(sys.argv)!=4:
    print("invalid arguments")
    print("format : <program.py> <input> <labels> <output>")
    sys.exit()

varList, scatterList= [], []
storeData = np.loadtxt(sys.argv[1], delimiter=',')
tagData = np.loadtxt(sys.argv[2], delimiter=',')
clusters, count = np.unique(tagData, return_counts=True)
rowLen, colLen=storeData.shape
temp1,temp2=np.matrix(tagData),np.matrix(storeData)

for each in clusters:
    lis=[]
    for index, value in enumerate(temp2):
        if tagData[index]==each:
            lis.append(value)
    mean = np.mean(lis,axis=0)
    varList.append(mean)

lis=[]
for every in range(len(clusters)):
    dummy=initalize(colLen)
    for index, value in enumerate(temp2):
        if tagData[index]==clusters[every]:
            dummy+=np.multiply(value - varList[every], getMatrixTranspose(value - varList[every]))
        scatterList.append(dummy)

W=sum(scatterList)


for index,value in enumerate(varList):
    lis.append(count[index] * value)

netAverage = caluclateNetAverage(lis,len(tagData))
Q=initalize(colLen)

for every in range(len(clusters)):
    Q+= count[every] * np.multiply((varList[every] - netAverage), getMatrixTranspose(varList[every] - netAverage))

M= W + Q
eigenValues, eigenVectors = getEigenProperties(W)
final = getFinalMatrix(eigenVectors,temp2)
np.savetxt(sys.argv[3], getMatrixTranspose(final), delimiter=',', fmt="%.10f")
