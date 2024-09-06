import pandas as pd
import numpy as np
from utils import plotting, tools, TimeConstraintEDs as Env
from sklearn.model_selection import ParameterGrid

param_grid = {'J': [1],
              'S': [2, 4, 6],
              'gamma': [10, 20, 30, 40, 1e3],
              'mu': [[0.5], [1], [2], [4], [6]],
              'load': [0.7, 0.8, 0.9],
              't': [[1]]}

grid = pd.DataFrame(ParameterGrid(param_grid))

grid['tail_rate'] = .0
grid['tail_erlang'] = .0
grid['tail_mms'] = .0

for i, inst in grid.iterrows():
    if i == 199:
        pass
    lab = Env.get_lambda(inst.mu[0], inst.S, inst.load, np.array([1]))
    pi_0 = Env.get_pi_0(inst.gamma, inst.S, inst.load, lab)
    tail_prob = Env.get_tail_prob(inst.gamma, inst.S, inst.load, lab,
                                  inst.mu[0], pi_0, inst.t[0]*inst.gamma)
    grid.loc[i, 'tail_rate'] = tail_prob

    grid.loc[i, 'tail_erlang'] = \
        (pi_0 / (1 - inst.load)
         * (lab + inst.gamma) / (inst.gamma + lab * pi_0)
         * np.exp(-inst.gamma / (inst.S * inst.mu[0] + inst.gamma)
                  * inst.S * inst.mu[0] * (1 - inst.load) * inst.t[0]))

    _, _, _, prob_late = tools.get_erlang_c(
        inst.mu[0], inst.load, inst.S, inst.t, [1])
    grid.loc[i, 'tail_mms'] = prob_late[0]

grid['gap_rate'] = .0
grid['gap_erlang'] = .0
grid['gap_diff'] = .0
gammas = param_grid['gamma']
gap = {}
diff_gap = {}
x_ticks = []
for gamma in gammas:
    x_ticks.extend(['rate $\gamma$=' + str(gamma),
                    'Erlang $\gamma$=' + str(gamma)])
    subset = grid[grid.gamma == gamma]
    gap['rate_' + str(gamma)] = ((subset.tail_mms - subset.tail_rate)
                                 / subset.tail_mms * 100)
    gap['erlang_' + str(gamma)] = ((subset.tail_mms - subset.tail_erlang)
                                   / subset.tail_mms * 100)
    diff_gap[gamma] = (gap['rate_' + str(gamma)]
                       - gap['erlang_' + str(gamma)])
    grid.loc[grid['gamma'] == gamma, 'gap_rate'] = gap['rate_' + str(gamma)]
    grid.loc[grid['gamma'] == gamma, 'gap_erlang'] = gap['erlang_' + str(gamma)]
    grid.loc[grid['gamma'] == gamma, 'gap_diff'] = diff_gap[gamma]

plotting.multi_boxplot(gap, gap.keys(),
                       'Discr. gap',
                       x_ticks,
                       'gap (%)',
                       violin=False,
                       rotation=20,
                       bottom=0.12,
                       left=0.2)
                       # y_lim=[-1000, 10])
plotting.multi_boxplot(diff_gap, diff_gap.keys(),
                       'Discr. gap difference relative to M/M/s',
                       gammas,
                       'diff in gap-%',
                       violin=False,
                       rotation=0,
                       left=0.2)
                       # y_lim=[-10, 500])
