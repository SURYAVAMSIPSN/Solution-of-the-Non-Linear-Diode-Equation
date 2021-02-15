# Matrix Examples
# author: olhartin@asu.edu updated by sdm

import numpy as np                 # needed for arrays
from numpy.linalg import solve     # linear algebra

# Ax=b                             # given this equation
# AT Ax = AT b                     # left mult by A transpose
# x = inv(AT)*ATb                  # left mult by the inverse of AT

# problem is 2 variables but 3 equations...
A = np.array([[2,3],[1,5],[4,1]],float)   # example array 3x2
b = np.array([1,2,3],float)               # answers... too many!

# note that things stay on the same side for "solve"
# so we'll use the second step above and let solve finish it
ATA = np.matmul(np.transpose(A),A)        # compute A-transpose * A
ATb = np.matmul(np.transpose(A),b)        # compute A-transpose * b
x = solve(ATA,ATb)                        # let solve get the answer
print(x)
print("\n\n")

