"""
Created on Sun Mar 29 10:41:12 2020.

@author: Jeroen
"""

from numpy import round, arange
import numpy as np

for i in arange(3):
    print(1 + 4 if i == 2 else 0)

for i in arange(3):
    print(1 + (4 if i == 2 else 0))

V = np.ones(3)
V_t = V*3
V_t[1] = -1
print('V', V, 'V_t', V_t)

V = np.zeros((3, 3))
W = np.random.randint(10, size=(4, 3, 3))
W[2] = V
V[0, 0] = 1
print(W, V)
