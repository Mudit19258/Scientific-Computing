"""
    Name - Mudit Balooja
    Roll no - 2019258
"""
# Q4 (c)
import numpy as np
import matplotlib.pyplot as plt
import random

# initializing the points
x = [0.]*6
y = [0.]*6
# seeding so we don't get random everytime
random.seed(123)
for i in range(0,6):
    x[i] = random.random()
    y[i] = random.random()
# sorting x coordinates
x.sort()

# length
n = len(x) - 1
# saves the difference between consecutive x coords
h = [0.0]*(n+1)

# h(i) = x(i+1) - x(i) for i in 0 to n-1 
for i in range(1, n+1):
    h[i] = x[i] - x[i-1]
    
# Setting matices for tri diagonal system
# (n+1)*(n+1) matrix
U = np.zeros((n+1,n+1))
# (n+1)*1 matrix
V = np.zeros((n+1, 1))

for i in range(n+1):
    # base cases

    # when i == 0
    if i == 0:
        V[i][0] = 0.0
    # when i == n
    elif i == n:
        V[i][0] = 0.0
    # filled according to equations
    else:
        temp1 = (y[i + 1] - y[i])/h[i + 1]
        temp2 = (y[i] - y[i - 1])/h[i]
        V[i][0] = 6.0 * (temp1 - temp2)

for j in range(n+1):
    # the base cases

    # when j == 0
    if j == 0:
        U[j][j] = 1.0
    # when j == n
    elif j == n:
        U[j][j] = 1.0
    # filled as per equations
    else:
        U[j][j] = 2.0 * (h[j] + h[j + 1])
        U[j][j+1] = h[j + 1]
        U[j][j-1] = h[j]
# Inverse 
Uinv = np.linalg.inv(U)
# matrix multiplication
Z = Uinv @ V

# interpolating and plotting
for i in range(n):
    interpolate_x = np.linspace(x[i], x[i+1])
    # this is fi(x)
    interpolate_y = (Z[i+1][0] - Z[i][0]) / (6.0 * h[i+1])*(interpolate_x - x[i])**3 \
                        + (Z[i][0]/2.0) * (interpolate_x - x[i])**2 \
                        + ((y[i+1] - y[i])/h[i+1] - ((2.0 * Z[i][0] * h[i + 1] + Z[i + 1][0] * h[i + 1]))/6.0) * (interpolate_x - x[i]) \
                        + y[i]
    plt.plot(interpolate_x, interpolate_y)
# Title
plt.title("Interpolation using natural cubic splines")
# scattering the points
plt.scatter(x,y)
# show
plt.show()