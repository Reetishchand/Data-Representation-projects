import numpy as np

Xt = np.matrix('1 2; 3 4; 5 6; 7 8; 9 9') # 5 x 2 matrix
print("Xt=", Xt)

# compute B = X X'
B = np.dot(Xt.T,Xt)
print("B=", B)

# another method to calculate B = X X' = sum xi xi'
B = sum([np.dot(xt.T,xt) for xt in Xt])
print("B=", B)

# another method to calculate B = X X' = sum xi xi'
n,m = Xt.shape
B = np.zeros((m,m))
for i, xt in enumerate(Xt) :
    B += np.dot(xt.T,xt)
print("B=", B)

# calculate mu = mean xi
mu = np.mean(Xt, axis=0)
print("mu=", mu)

# another method for calculating mu
n = sum([1 for xt in Xt])
mu = sum([xt for xt in Xt]) / n
print("mu=", mu)

y = np.array([1, 2, 1, 1, 2])
# calculate B1,m1,mu1  over xi where yi==1

B1 = np.zeros((m,m))
s1 = np.zeros((1,m))
m1 = 0

for i, xt in enumerate(Xt) :
    if y[i] == 1 :
        B1 += np.dot(xt.T,xt)
        s1 += xt
        m1 += 1
print("B1=", B1, "\n s1=", s1, "\n m1=", m1)

mu1 = s1/m1

# Calculate C1, the covariance matrix over xi where yi = 1
C1 = np.zeros((m,m))
for i, xt in enumerate(Xt) :
    if y[i] == 1 :
        ct = xt - mu1
        C1 += np.dot(ct.T,ct)
print("C1=", C1)





