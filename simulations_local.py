import os
from utils import tools, Simulation as Sim

FILEPATH_INSTANCE = 'results/'
FILEPATH_PICKLES = 'results/simulation_pickles/'
FILEPATH_RESULT = 'results/simulation_pickles/result_'

instance_names = [f for f in os.listdir(FILEPATH_INSTANCE)
                  if f.startswith('instance_sim_')]
continue_run = False
N = int(1e5)

# interested = [instance_names[i - 1] for i in [3, 9, 10, 11, 12]]
# inst_to_sim = [instance_names[i - 1] for i in [3]]
# inst_to_sim = [instance_names[i - 1] for i in range(len(instance_names) + 1)]
inst_to_sim = [instance_names[i - 1] for i in range(7, len(instance_names) + 1)]
for instance_name in inst_to_sim:
    inst = tools.inst_load(FILEPATH_INSTANCE + instance_name)
    methods = inst['method'].values
    for method in methods:
        simulation = Sim(inst=inst,
                         inst_id=instance_name[-6:-4],
                         method=method,
                         N=N)
        simulation.run(continue_run=continue_run)
