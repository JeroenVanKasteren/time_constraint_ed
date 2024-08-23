"""
Load and visualize results of simulation.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import numpy as np
import os
from utils import plotting, tools, TimeConstraintEDs as Env, PolicyIteration

FILEPATH_INSTANCE = 'results/'
FILEPATH_READ = 'results/read/'
FILEPATH_PICKLES = 'results/simulation_pickles/'
FILEPATH_V = 'results/value_functions/'
INSTANCES_ID = 'J1'
FILEPATH_INSTANCES = 'results/instances_' + INSTANCES_ID + '.csv'

instance_names = [f for f in os.listdir(FILEPATH_INSTANCE)
                  if f.startswith('instance_sim_')]
inst = tools.inst_load(FILEPATH_INSTANCE + instance_names[0])
methods = inst['method'].values

# --------------------- Plotting ---------------------
instances = [instance_names[i - 1] for i in range(1, 7)]
plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods, 'g', False)
plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods, 'perc')
instances = [instance_names[i - 1] for i in range(7, 12)]
plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods, 'g', False)
plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods, 'perc')
instances = [instance_names[i - 1] for i in range(12, 15)]
plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods, 'g', False)
plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods, 'perc')

# method = 'fcfs'  # method = 1
# instance_name = instance_names[10]
# method, row_id, inst, pickle = utils.tools.load_result(method, instance_name)
#
# utils.plotting.plot_convergence(pickle['kpi'], method,
#                                 inst.loc[row_id, 'start_K'],
#                                 inst.loc[row_id, 'batch_T'],
#                                 m=100)

# start = 0
# start = round_significance(random.randint(0, len(kpi_df)-size), 2)
# utils.plotting.plot_waiting(inst.loc[row_id], pickle['kpi'], 1000, start)
