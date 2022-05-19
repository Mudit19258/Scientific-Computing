"""
    Name - Mudit Balooja
    Roll no - 2019258
"""

# Q1
# (b)
import sys
from math import log,cos,sin

# Implementing newton's method
def Newton(function, derivative, x0, tolerance):
    fx = function(x0)
    iterations = 0
    x_list = []

    while abs(fx) > tolerance and iterations < 100:
        try:
            x0 = x0 - float(fx)/derivative(x0)
        except ZeroDivisionError:
            print("Zero div error!!")
            sys.exit(1)

        fx = function(x0)
        iterations += 1
        x_list.append(x0)

    return x_list

# Rate of Convergence
def rate(x):
    q = [log(abs(x[n+1] - x[n])/abs(x[n] - x[n-1]))/log(abs(x[n] - x[n-1])/abs(x[n-1] - x[n-2])) for n in range(2, len(x)-1, 1)]
    return q

# part (i)  
Fx1 = lambda x: x**2 - 1
DFx1 = lambda x: 2*x
ans = Newton(Fx1, DFx1, x0 = 10**6, tolerance = 1e-10)
print("Part (i)")
print(ans[len(ans)-1])
convergence_rate = rate(ans)
print(convergence_rate[len(convergence_rate)-1])
print("\n")
# part (ii)
Fx2 = lambda x: (x-1)**4
DFx2 = lambda x: 4*(x-1)**3
ans = Newton(Fx2, DFx2, x0 = 10, tolerance = 1e-10)
print("Part (ii)")
print(ans[len(ans)-1])
convergence_rate = rate(ans)
print(convergence_rate[len(convergence_rate)-1])
print("\n")
# part (iii)
Fx3 = lambda x: x - cos(x)
DFx3 = lambda x: 1 + sin(x)
ans = Newton(Fx3, DFx3, x0 = 1, tolerance = 1e-10)
print("Part (iii)")
print(ans[len(ans)-1])
convergence_rate = rate(ans)
print(convergence_rate[len(convergence_rate)-1])