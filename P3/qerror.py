import sys
import numpy as np
from numpy.random import seed
seed(1)

def getQE(input, clusters, iter):
    temp = []
    error = 0
    packs,count = np.unique(clusters, return_counts=True)
    for each in packs:
        lis=[]
        for z1,z2 in enumerate(M):
            if clusters[z1] == each:
                lis.append(z2)
        temp.append(np.mean(lis,axis=0))

    for each in range(iter):
        s1,s2=input[each],temp[int(clusters[each])-1]
        f = np.sum(s1-s2)
        f = f*f
        error +=f
    return error

if len(sys.argv) != 3:
    print('invalid arguments')
    print('<program> <data> <labels>')
    sys.exit()
input,clsuters = np.loadtxt(sys.argv[1], delimiter=','),np.loadtxt(sys.argv[2], delimiter=',')
rows, cols = input.shape
M,C = np.matrix(input),np.matrix(clsuters)
error = getQE(input, clsuters, rows)
print('Quantization Error:',error)