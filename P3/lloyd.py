import numpy as np
from numpy.random import seed
import sys
from numpy.random import randintz
from numpy import newaxis, delete


def calculateQE(input, centers, nearest):
    error = 0
    for i in range(len(input)):
        temp = (input[i] - centers[nearest[i]]) ** 2
        error += temp.sum()
    return error

def findNearestCenter(input, centers):
    return  np.sqrt(((input - centers[:, np.newaxis]) ** 2).sum(axis=2))


def chooseInitialCenters(input, kValue):
    inital = input.copy()
    np.random.shuffle(inital)
    return inital[:kValue]



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