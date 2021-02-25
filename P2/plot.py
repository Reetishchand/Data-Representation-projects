import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if  len(sys.argv) != 2 and len(sys.argv) != 3 :
    print('input format must be : python3 plot.py <dataplots>.txt <label>.txt')
    sys.exit()

M = np.genfromtxt(sys.argv[1], delimiter=',', autostrip=True)
assert(M.shape[1] >= 2)
M1 = M[:, 0]
M2 = M[:, 1]

if(len(sys.argv) == 3) :
    Y = np.genfromtxt(sys.argv[2], dtype='str')

graph, plot = plt.subplots()
plot.scatter(M1, M2)

if(len(sys.argv) == 3) :
    for i, j in enumerate(Y):
        plot.annotate(j, (M1[i], M2[i]))

# save to pdf
title = sys.argv[1] + 'Plot.pdf'
output = PdfPages(title)
output.savefig(graph)
output.close()