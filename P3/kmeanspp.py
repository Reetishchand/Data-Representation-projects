import numpy as np
from numpy import newaxis, delete
from numpy.random import randint
from numpy.random import seed
import sys


def calculateQE(input, centers, nearest):
    error = 0
    for i in range(len(input)):
        temp = (input[i] - centers[nearest[i]]) ** 2
        error += temp.sum()
    return error

def findNearestCenter(input, centers):
    return  np.sqrt(((input - centers[:, np.newaxis]) ** 2).sum(axis=2))


def chooseInitialCenters(input, kValue):
    inital = [input[0]]
    for kValue in range(1, kValue):
        lis=[]
        for every in input:
            minArr=[]
            for each in inital:
                val = np.inner(each - every, each - every)
                minArr.append(val)
            lis.append(min(minArr))
        lis = np.array(lis)

        total = lis.sum()
        probList = lis / total
        cpList = probList.cumsum()
        any,any = np.random.rand(), 0
        for index, value in enumerate(cpList):
            if any < value:
                temp = index
                break
        inital.append(input[temp])
    inital = np.array(inital)
    return inital


def updateNewCenters(input, nearest, centers):
    return np.array([input[nearest == each].mean(axis=0) for each in range(centers.shape[0])])

np.seterr(divide='ignore', invalid='ignore')
seed(1)
if len(sys.argv) != 5:
    print("invalid arguments")
    print("python <input> <K> <R> <output>")
    sys.exit()

noOfClusters, noOfIterations = int(sys.argv[2]), int(sys.argv[3])
with open(sys.argv[1],"r") as current:
  input = np.loadtxt(current, delimiter=",")

centers = chooseInitialCenters(input, noOfClusters)
for _ in range(0, noOfIterations + 1):
    nearest = findNearestCenter(input, centers)
    nearest = np.argmin(nearest, axis=0)
    centers = updateNewCenters(input, nearest, centers)

error = calculateQE(input, centers, nearest)

print("Quantization Error : ",error)
np.savetxt(sys.argv[4], nearest, delimiter=',', fmt ='%10f')