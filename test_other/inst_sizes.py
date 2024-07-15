"""
docstring todo...
"""

from utils import plotting
import numpy as np
from utils import tools

ID = 'test'  # 'plot_J2', 'plot_J3'
max_t_prob = 0.9
del_t_prob = True
max_size = 2e6
del_size = True

if ID == 'plot_J1':
    param_grid = {'J': [1],
                  'S': list(range(2, 10, 2)),
                  'D': [0],
                  'gamma': list(np.arange(100, 300, 25) / 10),
                  'mu': [[4]],
                  'load': list(np.arange(50, 95, 5) / 100)}
elif ID == 'plot_J2':
    mu = 2
    param_grid = {'J': [2],
                  'S': list(range(2, 10, 2)),
                  'D': [0],
                  'gamma': list(np.arange(100, 300, 25) / 10),
                  'mu': [[mu, mu], [mu, 2 * mu], [mu, 3 * mu]],
                  'load': list(np.arange(70, 95, 5) / 100),
                  'imbalance': [[1 / 2, 1], [1, 1], [2, 1]]}
elif ID == 'plot_J3':
    mu = 4
    param_grid = {'J': [3],
                  'S': list(range(2, 6, 2)),
                  'D': [0],
                  'gamma': list(np.arange(100, 250, 25) / 10),
                  'mu': [[mu, mu, mu], [mu, 1.5 * mu, 2 * mu]],
                  'load': list(np.arange(70, 90, 5) / 100),
                  'imbalance': [[1 / 3, 2 / 3, 1], [1, 1, 1],
                                [1, 2 / 3, 1 / 3]]}
elif ID == 'test':
    mu = 3
    # param_grid = {'J': [2],
    #               'S': [2, 4, 6],
    #               'D': [0],  # D=y*y_multi if < 0, formula if 0, value if > 0
    #               'gamma': [20],
    #               'mu': [[mu, mu], [2*mu, 2*mu], [mu, 2*mu], [2*mu, 4*mu]],
    #               'load': [0.7, 0.8, 0.9],
    #               'imbalance': [[1/3, 1], [1, 1], [3, 1]]}
else:
    print('Error: ID not recognized')
    exit(0)

grid = tools.get_instance_grid(param_grid, max_t_prob=max_t_prob,
                               max_size=max_size,
                               del_t_prob=del_t_prob,
                               del_size=del_size)

lab = np.array([np.sum(xi) for xi in grid.lab])
mu = [sum(lab) / sum(np.array(lab) / np.array(mu))
      for lab, mu in zip(grid.lab, grid.mu)]
S = grid.S.values
smu_rho = S * mu * (1 - lab / (S * lab))
grid['smu(1-rho)'] = smu_rho

x, x_lab = grid.gamma, 'gamma'
y, y_lab = smu_rho, 'smu(1-rho)'
cols = ['gamma', 'D', 'size', 'cap_prob', 'smu(1-rho)']
vmax = {'gamma': None, 'D': None, 'size': None, 'cap_prob': None,
        'smu(1-rho)': None}
vmin = {'gamma': None, 'D': None, 'size': 0, 'cap_prob': 0, 'smu(1-rho)': None}
for c_lab in cols:
    if c_lab not in [x_lab, y_lab]:
        plotting.plot_xyc(x, y, grid[c_lab], title=ID,
                          x_lab=x_lab, y_lab=y_lab, c_lab=c_lab,
                          vmin=vmin[c_lab], vmax=vmax[c_lab])
