"""
Created on Sun Mar 29 10:41:12 2020.

@author: Jeroen
"""

from numpy import arange
import numpy as np
import matplotlib.pyplot as plt

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

alpha = arange(2, 20, 0.1)
s = [2, 10]
mu = [1/4, 1/2]
rho = [0.5, 0.8, 0.9]
for rho_i in rho:
    p_w_d = []
    legends = []
    for mu_i in mu:
        for s_i in s:
            p_w_d.append(tuple(np.exp(-s_i * mu_i * (1 - rho_i) * alpha)))
            legends.append('mu=' + str(mu_i) + ', rho=' + str(rho_i) +
                           ', s=' + str(s_i) + ', smu(1-rho)=' +
                           str(np.around(s_i * mu_i * (1 - rho_i), 2)))
    for y in p_w_d:
        plt.plot(alpha, y)
    plt.legend(legends)
    plt.title('P(W>D), rho='+str(rho_i))
    plt.ylim([0, 1])
    plt.xlabel('alpha')
    plt.ylabel('P(W>D)')
    plt.grid()
    plt.show()
print('>=', -np.log(0.9), '?')


J = 2
gamma = arange(10, 20, 1)
rho = 0.9
S = [2, 5, 10]  # arange(2, 10, 4)
mu = [0.25, 0.33, 0.5]
rho = [0.8, 0.9]
for rho_i in rho:
    legends = []
    for s in S:
        for mu_i in mu:
            size = []
            for gamma_i in gamma:
                D = np.ceil(-np.log(0.001) / (s * mu_i * (1 - rho_i)) * gamma_i)
                dim = tuple(np.repeat([D + 1, s + 1], J))
                size.append(np.prod(dim))
            plt.plot(gamma, size)
            legends.append('s=' + str(s) + ', mu=' + str(mu_i))
    plt.legend(legends)
    plt.title('Size, rho=' + str(rho_i))
    plt.xlabel('gamma')
    plt.ylabel('size')
    plt.grid()
    plt.show()

import numpy as np
from sklearn.model_selection import ParameterGrid

S_GRID = [2, 5, 10]
MU_1_GRID = [1/4]
MU_2_GRID = np.array([1, 1.5, 2])*MU_1_GRID
RHO_GRID = [0.5, 0.6, 0.7, 0.8]  # 0.9?
RHO_IMB = [1/3, 1, 3]
param_grid = {'S': S_GRID,
              'mu_1': MU_1_GRID,
              'mu_2': MU_2_GRID,
              'rho': RHO_GRID,
              'imbalance': RHO_IMB}

grid = ParameterGrid(param_grid)

# # write
# with open('dict.csv', 'w') as csv_file:
#     writer = csv.writer(csv_file)
#     for key, value in mydict.items():
#        writer.writerow([key, value])
#
# # read back
# with open('dict.csv') as csv_file:
#     reader = csv.reader(csv_file)
#     mydict = dict(reader)
