import sys
import numpy as np

def initialize(colLen):
    return np.zeros((1,colLen))

if len(sys.argv) != 4  :
    print("invalid arguments")
    print("format : <program.py> <input> <labels> <output>")
    sys.exit()

storeData,tagData = np.loadtxt(sys.argv[1], delimiter=','),np.loadtxt(sys.argv[2], delimiter=',')
clusters, count = np.unique(tagData, return_counts=True)
gap,temp1 = np.mean(tagData), np.matrix(storeData)
rowLen, colLen = storeData.shape
averageList,scatterList = np.mean(temp1, axis=0), initialize(colLen)
var= 0

for index, value in enumerate(temp1):
    temp= np.square(value - averageList)
    scatterList+=temp
    t=tagData[index] - gap
    t=t*t
    var =var+t

Z= initialize(colLen)
for index in range(rowLen):
    a1 = temp1[index] - averageList
    b1 = tagData[index] - gap
    tm = a1 * b1
    Z+=tm

calc = np.sqrt(scatterList[0]) * np.sqrt(var)
F= np.abs(Z[0] / calc)
rankedIndexs = np.argsort(-F)
final= temp1[:, rankedIndexs]
np.savetxt(sys.argv[3], final[:, :2], delimiter=',', fmt='%.10f')
