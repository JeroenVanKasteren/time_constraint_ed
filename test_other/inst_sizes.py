"""
docstring todo...
"""

from utils import plotting, tools, TimeConstraintEDs as Env
import numpy as np
from utils import tools

ID = 'plot_J1'  # 'plot_J2', 'plot_J3'
FILEPATH_INSTANCE = 'results/instances_' + ID + '.csv'
max_t_prob = 0.9
del_t_prob = True
max_size = 2e6
del_size = True

instance_columns = ['J', 'S', 'gamma', 'D',
                    'mu', 'lab', 'load', 'imbalance'
                                         't', 'c', 'r',
                    'max_t_prob']

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
else:
    print('Error: ID not recognized')
    exit(0)

grid = tools.get_instance_grid(param_grid)

lab = np.array([np.sum(xi) for xi in grid.lab])
mu = [sum(lab) / sum(np.array(lab) / np.array(mu))
      for lab, mu in zip(grid.lab, grid.mu)]
S = grid.S.values
smu_rho = S * mu * (1 - lab / (S * lab))
grid['smu(1-rho)'] = smu_rho

x, x_lab = grid.gamma, 'gamma'
y, y_lab = smu_rho, 'smu(1-rho)'
cols = ['gamma', 'D', 'size', 'cap_prob', 'smu(1-rho)']
vmax = {'gamma': None, 'D': None, 'size': 10e6, 'cap_prob': None,
        'smu(1-rho)': None}
vmin = {'gamma': None, 'D': None, 'size': 0, 'cap_prob': 0, 'smu(1-rho)': None}
for c_lab in cols:
    if c_lab not in [x_lab, y_lab]:
        plotting.plot_xyc(x, y, grid[c_lab], title=ID,
                          x_lab=x_lab, y_lab=y_lab, c_lab=c_lab,
                          vmin=vmin[c_lab], vmax=vmax[c_lab])

inst = grid.iloc[8, :]
lab = sum(inst.lab)
mu = lab / np.sum(inst.lab[0] / inst.mu[0])
pi_0 = Env.get_pi_0(inst.gamma, inst.S, inst.load, lab)
prob_delay = Env.get_tail_prob(inst.gamma, inst.S, inst.load,
                               lab, mu, pi_0, 0)
print(pi_0)
print(prob_delay)
print(Env.get_tail_prob(inst.gamma, inst.S, inst.load,
                        lab, mu, pi_0, 1))
print(Env.get_tail_prob(inst.gamma, inst.S, inst.load,
                        lab, mu, pi_0, inst.D))
print(int(np.ceil(-np.log(0.001 / prob_delay) /
                  (inst.S * mu - lab) * inst.gamma)))

gamma = 1e3
pi_0 = Env.get_pi_0(inst.gamma, inst.S, inst.load, lab)
prob_delay = Env.get_tail_prob(inst.gamma, inst.S, inst.load,
                               lab, mu, pi_0, 0)
print(pi_0)
print(prob_delay)
print(Env.get_tail_prob(gamma, inst.S, inst.load,
                        lab, mu, pi_0, 0))
print(Env.get_tail_prob(gamma, inst.S, inst.load,
                        lab, mu, pi_0, 1*gamma))
print(Env.get_tail_prob(gamma, inst.S, inst.load,
                        lab, mu, pi_0, inst.D*gamma))
print(np.log(1e-3/prob_delay) /
      np.log(1 - (inst.S*mu - lab)/(inst.S*mu + gamma)))
