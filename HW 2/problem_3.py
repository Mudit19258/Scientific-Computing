# Problem 3

import numpy as np

# Gaussian Elimination without partial pivoting for factorizing a
# linear system A x = b into A = L * U and b = L^{-1}x * b
#
# Inputs: Matrix A, Vector b
#
# Outputs: Solution x
#
def GE(A, b):
    # Your function definition goes here
    """
        This is the forward substitution part
    """
    n = A.shape[0]
    # python is 0 indexed, so small change here 
    for k in range(0,n-1):
        for i in range(k+1,n):
            temp = A[i][k]/A[k][k]
            A[i][k] = temp
            for j in range(k+1,n):
                A[i][j] -= (temp*A[k][j])
            b[i] -= (temp*b[k])
    """
        This is the backward substitution part
    """
    x = np.zeros((n))
    x[n-1] = b[n-1]/A[n-1, n-1]
    # Again 0 indexing in python
    for i in range(n-2, -1, -1):
        sum = b[i]
        for j in range(i+1, n):
            sum -= (A[i][j]*x[j])
        x[i] = sum/A[i][i]
    return x


# Gaussian Elimination with partial pivoting for factorizing a
# linear system A x = b into P * A = L * U and b = L^{-1}x * P * b
#
# Inputs: Matrix A, Vector b
#
# Outputs: Solution x
#
# Note: The permutation matrix is not tracked
def GE_pp(A, b):
    # Your function definition goes here   
    n = A.shape[0] 
    s = np.zeros(n)
    #l = list(range(n))
    l = np.array([i for i in range(n)])
    for i in range(0, n):
        Smax = 0
        #l[i] = i
        for j in range(0,n):
            Smax = max(Smax, A[i][j])
        s[i] = Smax

    for k in range(n-1):
        Rmax = 0
        for i in range(k,n):
            r = abs(A[l[i]][k]/s[l[i]])
            if r > Rmax:
                Rmax = r
                j = i
        temp = l[k]
        l[k] = l[j]
        l[j] = temp

        for i in range(k+1,n):
            Amult = A[l[i]][k]/A[l[k]][k]
            A[l[i]][k] = Amult
            for j in range(k+1,n):
                A[l[i],j] = A[l[i],j] - Amult * A[l[k],j]

    x = np.zeros(n)
    x[n-1] = b[l[n-1]]/A[l[i]][n-1]

    for k in range(0, n-1):
        for i in range(k+1,n):
            b[l[i]] -= A[l[i],k] * b[l[k]]
    
    for i in range(n-1, -1, -1):
        sum = b[l[i]]
        for j in range(i+1, n):
            sum -= A[l[i],j] * x[j]
        x[i] = sum/A[l[i],i]

    return x


A = np.array([
                [1,1,2*pow(10,9)],
                [2, -1, pow(10,9)],
                [1,2,0]
            ],dtype=float)
b = np.array([1,1,1],dtype=float)

# Part (a)
x_pp = GE_pp(A.copy(),b.copy())
print("\n (a) Output with Gaussian Elimination with partial pivoting")
print("\nA:")
print(A)
print("\nb:")
print(b)
print("\nx:")
print(x_pp)

# preforming row equilibriation
b[0] = b[0]/max(abs(A[0]))
b[1] = b[1]/max(abs(A[1]))
b[2] = b[2]/max(abs(A[2]))

A[0] = A[0]/max(abs(A[0]))
A[1] = A[1]/max(abs(A[1]))
A[2] = A[2]/max(abs(A[2]))
print("\n (b) Output with row equilibrated and Gaussian elimination without partial pivoting")
print("\nA:")
print(A)
print("\nb:")
print(b)
op_x = GE(A.copy(),b.copy())
print("\nx:")
print(op_x)
