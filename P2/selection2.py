import sys
import numpy as np

def initialize(colLen):
    return np.zeros((1,colLen))

if len(sys.argv) != 4:
    print("invalid arguments")
    print("format : <program.py> <input> <labels> <output>")
    sys.exit()

storeData , tagData = np.loadtxt(sys.argv[1], delimiter=','), np.loadtxt(sys.argv[2], delimiter=',')
clusters, count = np.unique(tagData, return_counts=True)
rowLen, colLen = storeData.shape
temp1,temp2 = np.matrix(storeData),np.zeros((len(clusters), 1, colLen))
averageList,gap = np.mean(temp1, axis=0),np.mean(tagData)
scatterList = np.zeros((len(clusters), 1, colLen))

for each in range(len(clusters)):
    for index, value in enumerate(temp1):
        if tagData[index] == clusters[each]:
            temp2[each] = value + temp2[each]
    temp2[each] = temp2[each]/count[each]


F = initialize(colLen)
for each in range(len(clusters)):
    te = np.square(temp2[each] - averageList)
    F += count[each] * te

for each in range(len(clusters)):
    for index, value in enumerate(temp1):
        if clusters[each] == tagData[index]:
            scatterList[each] += np.square(value - temp2[each])

final = np.sum(scatterList, axis=0)[0]
X = F[0] / final
rankedIndexes = np.argsort(-X)
final = temp1[:, rankedIndexes]
np.savetxt(sys.argv[3], final[:, :2], delimiter=',', fmt='%.10f')