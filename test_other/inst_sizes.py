from utils import plotting, tools, TimeConstraintEDs as Env

import numpy as np
import os
from utils import tools
from utils import instances_sim
import pandas as pd

ID = 'sim'  # 'J2', 'J3', 'J2_D_gam', 'J1_D', 'sim'
FILEPATH_INSTANCE = 'results/instances_' + ID + '.csv'
max_target_prob = 0.9
remove_max_t_prob = True
max_size = 2e6
remove_max_size = True

instance_columns = ['J', 'S', 'gamma', 'D',
                    'mu', 'lab', 'load', 'imbalance'
                    't', 'c', 'r',
                    'max_t_prob']

if ID == 'plot_J1':  # Boxplot of D/gamma for different gamma and J
    mu = 1 / 3
    param_grid = {'J': [1],
                  'S': list(range(2, 10)),
                  'D': [0],
                  'gamma': [10, 15, 20, 25],
                  'mu': [[mu, mu], [mu, 2 * mu]],
                  'load': [0.6, 0.7, 0.8, 0.9, 0.95],
                  'imbalance': [[1 / 3, 1], [1, 1], [3, 1]]}
elif ID == 'plot_J2':
    mu = 1 / 3
    param_grid = {'J': [2],
                  'S': list(range(2, 10)),
                  'D': [0],
                  'gamma': [10, 15, 20, 25],
                  'mu': [[mu, mu], [mu, 2*mu]],
                  'load': [0.6, 0.7, 0.8, 0.9, 0.95],
                  'imbalance': [[1/3, 1], [1, 1], [3, 1]]}
elif ID == 'plot_J3':
    param_grid = {'J': [3],
                  'S': list(range(2, 10)),
                  'D': [0],
                  'gamma': [10, 15, 20, 25],
                  'mu': [[mu, mu, mu], [mu, 1.5*mu, 2*mu]],
                  'load': [0.6, 0.7, 0.8, 0.9, 0.95],
                  'imbalance': [[1/3, 2/3, 1], [1, 1, 1], [1, 2/3, 1/3]]}
    grid = tools.get_instance_grid(param_grid)
