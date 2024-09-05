import pandas as pd
import numpy as np
from utils import plotting, tools, TimeConstraintEDs as Env
from sklearn.model_selection import ParameterGrid

param_grid = {'J': [1],
              'S': [2, 4, 6],
              'gamma': [10, 20, 30, 40, 50],
              'mu': [[0.5], [1], [2], [4], [6]],
              'load': [0.7, 0.8, 0.9],
              't': [[1]]}

grid = pd.DataFrame(ParameterGrid(param_grid))

grid['tail_rate'] = .0
grid['tail_erlang'] = .0
grid['tail_mms'] = .0

for i, inst in grid.iterrows():
    inst_dict = inst.to_dict()
    env = Env(sim=True, **inst_dict)
    grid.loc[i, 'tail_rate'] = env.target_prob

    grid.loc[i, 'tail_erlang'] = \
        (env.pi_0 / (1 - env.rho)
         * (env.lab + env.gamma) / (env.gamma + env.lab * env.pi_0)
         * np.exp(-env.gamma / (env.S * env.mu + env.gamma)
                  * env.S * env.mu * (1 - env.rho) * env.t))

    _, _, _, prob_late = tools.get_erlang_c(
        env.mu, env.rho, env.S, env.t, [1])
    grid.loc[i, 'tail_mms'] = prob_late[0][0]

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
plotting.multi_boxplot(gap, gap.keys(),
                       'Discr. gap',
                       x_ticks,
                       'gap (%)',
                       violin=False,
                       rotation=20,
                       bottom=0.12)
plotting.multi_boxplot(diff_gap, diff_gap.keys(),
                       'Discr. gap difference relative to M/M/s',
                       gammas,
                       'diff in gap-%',
                       violin=False,
                       rotation=0)
