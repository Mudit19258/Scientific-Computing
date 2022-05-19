"""
    Name - Mudit Balooja
    Roll no - 2019258
"""
import numpy as np
import scipy.optimize as opt
import math
import matplotlib.pyplot as plt
from math import *

# Implementing newton's method for non-linear
def Newton(f, J, x0, xstar, tol = 1e-12, maxit = 500):
    Fx = f(x0,xstar)
    F_norm = np.linalg.norm(Fx, ord = 2)  
    iterations = 0
    while abs(F_norm) > tol and iterations < maxit:
        delta = np.linalg.solve(J(x0), -Fx)
        x0 = x0 + delta
        Fx = f(x0,xstar)
        F_norm = np.linalg.norm(Fx, ord = 2)
        iterations += 1
    return x0

# Find r, theta, phi
def stc(rtp):
    r, theta, phi = rtp
    x = r*sin(theta)*cos(phi)
    y = r*sin(theta)*sin(phi)
    z = r*cos(theta)
    return np.array([x,y,z])

def cts(xyz):
    x, y, z = xyz
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
    return np.array([r,theta,phi])


def function(rtp,xyz):
    x, y, z = xyz
    r, theta, phi = rtp
    return np.array([r*sin(theta)*cos(phi) - x,
            r*sin(theta)*sin(phi) - y,
            r*cos(theta) - z])

def jacobian(rtp):
    r, theta, phi = rtp
    return np.array([[sin(theta)*cos(phi), r*cos(theta)*cos(phi), (-1)*r*sin(theta)*sin(phi)],
            [sin(theta)*sin(phi), r*cos(theta)*sin(phi), r*sin(theta)*cos(phi)],
            [cos(theta), (-1)*r*sin(theta), 0]])

# Inbuilt
# sol = opt.root(function_exercise, x0=np.array([1.0,2.0,1.0]), jac=jacobian_exercise)
# root = sol.x
# print(root)

# prevent changing everytime
np.random.seed(123)

for i in range(10):
    xstar = np.random.randn(3)

    ans = Newton(function, jacobian, np.array([2.0,1.0,0.0]), xstar)

    X = stc(ans)

    res = np.subtract(X, np.array(xstar))
    rel_residual = np.linalg.norm(res, ord = 2)/np.linalg.norm(xstar, ord = 2)
    
    #print(rel_residual)

    var7 = cts(np.array(xstar))
    res2 = np.subtract(var7, np.array(ans))

    rel_error = np.linalg.norm(res2, ord = 2)/np.linalg.norm(var7, ord = 2)
    #print(rel_error)

    print("n = "+str(i))
    print("Relative Residual = "+str(rel_residual))
    print("Relative Error = "+str(rel_error))
    print("\n")


