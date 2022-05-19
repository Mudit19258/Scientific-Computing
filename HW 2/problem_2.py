# Problem 2: Gaussian elimination without and with partial
# pivoting.

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
        Smax = 0.0
        #l[i] = i
        for j in range(0,n):
            Smax = max(Smax, A[i][j])
        s[i] = Smax

    for k in range(n-1):
        Rmax = 0.0
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

n = 10 # 10,20,30,40. Default: 10
matrix_choice = 1 # 1, 2 or 3. Default: random
def choice1(n):
    np.random.seed(0)
    A = np.random.rand(n, n)
    return A
def choice2(n):
    A = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            A[i][j] = 1.0/(i+j+1)
    return A

def choice3(n):
    A = np.empty(shape=(n,n))
    for i in range(n):
        for j in range(n):
            if i >= j:
                A[i][j] = 1
            else:
                A[i][j] = -1
    return A

if matrix_choice == 2:
    # Hilbert Matrix
    A = choice2(n)
    print(A)
     
elif matrix_choice == 3:
    # Your code for generating matrix  
    A = choice3(n)
    print(A)
else:
    # Random matrix example
    np.random.seed(0)
    A = np.random.rand(n, n)
    print(A)

x_star = np.ones(n)
b = np.dot(A, x_star)
    
# Computations as sought
print ("condition number: %g" % np.linalg.cond(A))

# Solve without pivoting
x_npp = GE(A.copy(), b.copy())
# this is relative measurements here, changed according to sir's instructions on discord
error_npp = np.linalg.norm(x_star - x_npp)/np.linalg.norm(x_star)
residual_npp = np.linalg.norm(np.dot(A, x_npp) - b)/(np.linalg.norm(A)*np.linalg.norm(x_npp))
print ("\nNo partial pivoting:")
print ("Error = %1.6g  Residual = %1.6g" %(error_npp, residual_npp))

# Solve with partial pivoting
# To be completed by you
x_pp = GE_pp(A.copy(), b.copy())
error_pp = np.linalg.norm(x_star - x_pp)/np.linalg.norm(x_star)
residual_pp = np.linalg.norm(np.dot(A, x_pp) - b)/(np.linalg.norm(A)*np.linalg.norm(x_pp))
print ("\nPartial Pivoting:")
print ("Error = %1.6g  Residual = %1.6g" %(error_pp, residual_pp))

# Solve using numpy.linalg's solve
# To be completed by you
x_inbuilt = np.linalg.solve(A.copy(), b.copy())
error_inbuilt = np.linalg.norm(x_star - x_inbuilt)/np.linalg.norm(x_star)
residual_inbuilt = np.linalg.norm(np.dot(A, x_inbuilt) - b)/(np.linalg.norm(A)*np.linalg.norm(x_inbuilt))
print ("\nUsing Inbuilt functions:")
print ("Error = %1.6g  Residual = %1.6g" %(error_inbuilt, residual_inbuilt))

from tabulate import tabulate
import pandas
N_values = [10,20,30,40]
Rows = ["","N = 10", "N = 20", "N = 30", "N = 40"]
Columns = ["Condition Number", "Error(un-pivoted)", "Residual(un-pivoted)", "Error(partially pivoted)", "Residual(partially pivoted)", "Error(inbuilt)", "Residual(inbuilt)"]
r,c = 7,4
Data = np.empty(shape=(r+1,c+1),dtype=object)
for i in range(7):
    Data[i][0] = Columns[i]
for i in range(1,5):
    A_random = choice1(N_values[i-1])
    x_star = np.ones(N_values[i-1])
    b_random = np.dot(A_random, x_star)

    x_npp = GE(A_random.copy(), b_random.copy())
    error_npp = np.linalg.norm(x_star - x_npp)/np.linalg.norm(x_star)
    residual_npp = np.linalg.norm(np.dot(A_random, x_npp) - b_random)/(np.linalg.norm(A_random)*np.linalg.norm(x_npp))

    x_pp = GE_pp(A_random.copy(), b_random.copy())
    error_pp = np.linalg.norm(x_star - x_pp)/np.linalg.norm(x_star)
    residual_pp = np.linalg.norm(np.dot(A_random, x_pp) - b_random)/np.linalg.norm(A_random) * np.linalg.norm(x_pp)

    x_inbuilt = np.linalg.solve(A_random.copy(), b_random.copy())
    error_inbuilt = np.linalg.norm(x_star - x_inbuilt)/np.linalg.norm(x_star)
    residual_inbuilt = np.linalg.norm(np.dot(A_random, x_inbuilt) - b_random)/(np.linalg.norm(A_random)*np.linalg.norm(x_inbuilt))

    Data[0][i] = "%g"%(np.linalg.cond(A_random))
    Data[1][i] = "{:1.6g}".format(error_npp)
    Data[2][i] = "%1.6g"%residual_npp
    Data[3][i] = "{:1.6g}".format(error_pp)
    Data[4][i] = "%1.6g"%residual_pp
    Data[5][i] = "{:1.6g}".format(error_inbuilt)
    Data[6][i] = "%1.6g"%residual_inbuilt

print("\nRANDOM VALUES MATRIX")
print(tabulate(Data,headers=Rows,tablefmt='orgtbl'))

print("\nHILBERT MATRIX")
Data = np.empty(shape=(r+1,c+1),dtype=object)
for i in range(7):
    Data[i][0] = Columns[i]
for i in range(1,5):
    A_random = choice2(N_values[i-1])
    x_star = np.ones(N_values[i-1])
    b_random = np.dot(A_random, x_star)

    x_pp = GE_pp(A_random.copy(), b_random.copy())
    error_pp = np.linalg.norm(x_star - x_pp)/np.linalg.norm(x_star)
    residual_pp = np.linalg.norm(np.dot(A_random, x_pp) - b_random)/(np.linalg.norm(A_random)*np.linalg.norm(x_pp))

    x_npp = GE(A_random.copy(), b_random.copy())
    error_npp = np.linalg.norm(x_star - x_npp)/np.linalg.norm(x_star)
    residual_npp = np.linalg.norm(np.dot(A_random, x_npp) - b_random)/(np.linalg.norm(A_random)*np.linalg.norm(x_npp))

    x_inbuilt = np.linalg.solve(A_random.copy(), b_random.copy())
    error_inbuilt = np.linalg.norm(x_star - x_inbuilt)/np.linalg.norm(x_star)
    residual_inbuilt = np.linalg.norm(np.dot(A_random, x_inbuilt) - b_random)/(np.linalg.norm(A_random)*np.linalg.norm(x_inbuilt))

    Data[0][i] = "%g"%(np.linalg.cond(A_random))
    Data[1][i] = "{:1.6g}".format(error_npp)
    Data[2][i] = "%1.6g"%residual_npp
    Data[3][i] = "{:1.6g}".format(error_pp)
    Data[4][i] = "%1.6g"%residual_pp
    Data[5][i] = "{:1.6g}".format(error_inbuilt)
    Data[6][i] = "%1.6g"%residual_inbuilt

print(tabulate(Data,headers=Rows,tablefmt='orgtbl'))

print("\n3rd CHOICE MATRIX")
Data = np.empty(shape=(r+1,c+1),dtype=object)
for i in range(7):
    Data[i][0] = Columns[i]
for i in range(1,5):
    A_random = choice3(N_values[i-1])
    x_star = np.ones(N_values[i-1])
    b_random = np.dot(A_random, x_star)

    x_pp = GE_pp(A_random.copy(), b_random.copy())
    error_pp = np.linalg.norm(x_star - x_pp)/np.linalg.norm(x_star)
    residual_pp = np.linalg.norm(np.dot(A_random, x_pp) - b_random)/(np.linalg.norm(A_random)*np.linalg.norm(x_pp))

    x_npp = GE(A_random.copy(), b_random.copy())
    error_npp = np.linalg.norm(x_star - x_npp)/np.linalg.norm(x_star)
    residual_npp = np.linalg.norm(np.dot(A_random, x_npp) - b_random)/(np.linalg.norm(A_random)*np.linalg.norm(x_npp))

    x_inbuilt = np.linalg.solve(A_random.copy(), b_random.copy())
    error_inbuilt = np.linalg.norm(x_star - x_inbuilt)/np.linalg.norm(x_star)
    residual_inbuilt = np.linalg.norm(np.dot(A_random, x_inbuilt) - b_random)/(np.linalg.norm(A_random)*np.linalg.norm(x_inbuilt))

    Data[0][i] = "%g"%(np.linalg.cond(A_random))
    Data[1][i] = "{:1.6g}".format(error_npp)
    Data[2][i] = "%1.6g"%residual_npp
    Data[3][i] = "{:1.6g}".format(error_pp)
    Data[4][i] = "%1.6g"%residual_pp
    Data[5][i] = "{:1.6g}".format(error_inbuilt)
    Data[6][i] = "%1.6g"%residual_inbuilt

print(tabulate(Data,headers=Rows,tablefmt='orgtbl'))