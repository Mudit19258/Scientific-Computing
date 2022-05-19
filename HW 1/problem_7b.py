# Problem 7b: Implementation of the Taylor series formula for
# estimation of the number e.

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spsp

e_sum = [0., 1.]; n = 1
eps_M = np.finfo(float).eps

while e_sum[-1] - e_sum[-2] > eps_M:
    e_sum.append(e_sum[-1] + 1/spsp.factorial(n))
    n += 1
# Note that e_sum[-1] and e_sum[-2] will be the same values after
# exiting the loop

error = abs(np.exp(1) - e_sum[-1])/np.exp(1)
print("\nThe computation converged after summing %1d terms."%(n - 1))
print("The relative error in computing e via its truncated "
          + "Taylor series is %1.16e."%error)





