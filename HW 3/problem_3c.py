"""
    Name - Mudit Balooja
    Roll no - 2019258
"""

# Q3
import math
import numpy as np
import matplotlib.pyplot as plt

# values of n
n_vals = np.arange(5,101,5)
#n_vals = [5]

# for storing the condition numbers
result1 = []
result2 = []
result3 = []
result4 = []

# for each n
for n in n_vals:
    # diff Vandermonde matrices
    V1 = np.zeros((n,n))
    V2 = np.zeros((n,n))
    V3 = np.zeros((n,n))
    V4 = np.zeros((n,n))

    # equispaced nodes
    arr = np.linspace(-1,1,n)

    # chebyshev nodes
    arr2 = []
    for i in range(n):
        arr2.append(math.cos(math.pi * ((2*i+1)/(2*n))))
    
    for i in range(n):
        for j in range(n):
            V1[i][j] = arr[i] ** j # eq nodes with mono basis
            V2[i][j] = (arr2[i])**j # cheby nodes with mono basis
            V3[i][j] = math.cos(j * np.arccos(arr[i])) # Eq nodes with cheby poly
            V4[i][j] = math.cos(j * np.arccos(arr2[i])) # cheby nodes with cheby poly

    result1.append(np.linalg.cond(V1))
    result2.append(np.linalg.cond(V2))
    result3.append(np.linalg.cond(V3))
    result4.append(np.linalg.cond(V4))
plt.title("Behaviour of condition number")
plt.semilogy(n_vals, result1, label = "part (i)")
plt.semilogy(n_vals, result2, label = "part (ii)")
plt.semilogy(n_vals, result3, label = "part (iii)")
plt.semilogy(n_vals, result4, label = "part (iv)")
plt.legend()
plt.show()