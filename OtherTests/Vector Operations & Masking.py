"""
Created on Wed Sep 23 14:35:04 2020.

@author: jkn354
"""

from numpy import array
from numpy import arange
from numpy import maximum
import numpy as np
from scipy.special import factorial as fac

a = array([1, 2, 3, 0])
j = array(arange(1, 5))
a**j / fac(j)

# Works
maximum(arange(6), 3.5)

# NOT AS EXPECTED
A = np.array([[1,2,3],
              [0],
              [3,4,5]])
B = np.array([[1,1,1],
              [0],
              [2,2,2]])
A+B

s = 3
A = np.indices((s+1, s+1))
mask_triu = A[0] <= A[1]  # upper trianular matrix ('=' including diagonal)
mask_tril = A[0] >= A[1]  # lower trianular matrix ('=' including diagonal)
M = np.zeros([s+1, s+1])
M[mask_triu] = A[0, mask_triu] - A[1, mask_triu]
