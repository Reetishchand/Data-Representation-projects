import sys
import numpy as np
import math
from numpy.random import seed


def calculateQE(matrix, closest):
    e = 0
    for _ in closest:
        row = matrix[_]
        center = row.mean(axis=0)
        for row in row:
            x = np.linalg.norm(row - center)
            x = math.pow(x, 2)
            e = e + x
    return e

def performSpectralClustering(data, weightList, deg, lapNorm):
    cap,val = len(data),float('inf')
    A1,A2 = None,None
    eVals, eVecs = np.linalg.eig(lapNorm)
    sortedIndexs = np.argsort(eVals)[::-1]
    eVals = eVals[sortedIndexs]
    eVecs = eVecs[:, sortedIndexs]
    eigen_val_second_small = eVals[-2]
    filteredEigenVecs = eVecs[:, -2]
    sortedIndexs = np.argsort(filteredEigenVecs)[::]
    for each in range(len(sortedIndexs) - 1):
        t1,t2 = set(sortedIndexs[:each + 1]), set(sortedIndexs[each + 1:])
        z1,z2,z3 = 0,0,0
        for _ in range(cap):
            if _ in t1:
                z1 = z1 + deg[_]
            elif _ in t2:
                z2 = z2 + deg[_]
            for u in range(cap):
                if _ in t1 and u in t2:
                    z3 = z3 + weightList[_, u]
        zamp = z3 / (float(min(z1, z2)))
        if zamp < val:
            A1,A2 = list(t1),list(t2)
            val = zamp
    A1 = np.asarray(data)[A1].tolist()
    A2 = np.asarray(data)[A2].tolist()
    return A1, A2


def getLapLacianParameters(matrix, cols, alpha):
    cap = len(cols)
    distance = np.zeros((cap, cap))
    for each in range(cap):
        for every in range(cap):
            distance[each, every] = np.linalg.norm(matrix[each] - matrix[every])
    deg,weightArr = np.zeros(cap),np.zeros((cap, cap))

    for each in range(cap):
        for every in range(cap):
            temp = (distance[each, every] / alpha)
            thetha = math.pow(temp, 2)
            thetha =  0.5 *thetha
            weightArr[each, every] = math.exp(-thetha)
            deg[each] = deg[each] + weightArr[each, every]
    lapNorm = np.zeros((cap, cap))
    for each in range(cap):
        for every in range(cap):
            if each != every and deg[each] != 0 and deg[every] != 0:
                z = math.sqrt(deg[each] * deg[every])
                z =  (-weightArr[each, every]) / z
                lapNorm[each, every] =z
            elif deg[each] != 0 and each == every:
                lapNorm[each, every] = 1
            else:
                lapNorm[each, every] = 0
    return weightArr, deg, lapNorm







if len(sys.argv) != 5:
    print('invalid arguments')
    print('<program> <data> <k> <sigma> <output> ')
    sys.exit()
seed(1)

kValue,alpha = int(sys.argv[2]),float(sys.argv[3])
M = np.genfromtxt(sys.argv[1], delimiter=',', autostrip=True)
partArr = [list(_ for _ in range(M.shape[0]))]
lin = partArr.pop(0)
weightArr, deg, lapNorm = getLapLacianParameters(M, lin, alpha)
updatedCenters = performSpectralClustering(lin, weightArr, deg, lapNorm)
partArr.extend(updatedCenters)

while len(partArr) < kValue:
    lamdaMin,partMin,wtMin,degMin,lapNormMin = float('inf'),None,None,None,None
    for index in range(len(partArr)):
        weightArr, deg, lapNorm = getLapLacianParameters(M, partArr[index], alpha)
        eVals, eVecs = np.linalg.eig(lapNorm)
        sortedIndexs = np.argsort(eVals)[::-1]
        eVals = eVals[sortedIndexs]
        eValMin = eVals[-2]
        if eValMin < lamdaMin:
            wtMin,degMin,partMin,lapNormMin =weightArr, deg,index, lapNorm
    bound = partArr.pop(partMin)
    updatedBound = performSpectralClustering(bound, wtMin, degMin, lapNormMin)
    partArr.extend(updatedBound)

result = np.zeros((M.shape)[0])
for i in range(len(partArr)):
    for j in partArr[i]:
        result[j] = i
result = np.array(result)
qe = calculateQE(M, partArr)
print("Quantization Error is : ",qe)
np.savetxt(sys.argv[4], result, delimiter=',', fmt='%10f')