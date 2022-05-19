# Problem 7a: Implementation of the formula (n + 1/n)**n for
# estimation of the number e.

import numpy as np
import matplotlib.pyplot as plt

def e_by_limit(n):
    return (1 + 1/n)**n

N = 17
err_n = [abs(np.exp(1) - e_by_limit(10**k))/np.exp(1)
             for k in range(1, N + 1)]

# Plot log of err_n versus n.
x_n = [10**k for k in range(1, N + 1)]
plt.figure()
plt.loglog(x_n, err_n, 'bo')
plt.grid(True)
plt.ylabel("$log (|e - (1 + 1/n)^n|/e)$")
plt.xlabel("$n$")
plt.title("Error in computing e using the limit formula", fontsize=14)
plt.savefig("problem7a.pdf")



