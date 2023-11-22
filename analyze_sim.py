"""
Load and visualize results of simulation.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from utils import tools

FILEPATH_INSTANCE = 'results/'
FILEPATH_READ = 'results/read/'
FILEPATH_PICKLES = 'results/simulation_pickles/'

instance_names = [f for f in os.listdir(FILEPATH_INSTANCE)
                  if f.startswith('instance_sim_')]
# inst_nrs = [name.split('_')[2][:-4] for name in instance_names]

inst = tools.inst_load(FILEPATH_INSTANCE + instance_names[0])
methods = inst['method'].values
# methods = ['ospi', 'cmu', 'fcfs', 'sdf', 'sdfprior']

"""
instance_name = instance_names[7]
inst = tools.inst_load(FILEPATH_INSTANCE + instance_name)
instance_id = instance_name.split('_')[2][:-4]
methods = inst['method'].values
method = methods[4]
row_id = 4
"""

performances = {method: [[], []] for method in methods}
for instance_name in instance_names:
    inst = tools.inst_load(FILEPATH_INSTANCE + instance_name)
    for row_id, method in enumerate(methods):
        performances[method][0].extend([inst.loc[row_id, 'g']])
        performances[method][1].extend([inst.loc[row_id, 'conf_int']])

x = np.arange(len(methods))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
for method, g, conf_int in performances.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, g, width, label=method, yerr=(g-conf_int,
                                                             g+conf_int))
    ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set_ylabel('g')
ax.set_title('long term average reward')
ax.set_xticks(x + width, methods)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 250)

plt.show()

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

start_K = 1e4
batch_T = 1e4

plt.scatter(kpi_df['time']/60, kpi_df['g'])
plt.xlabel('Running time (hours)')
plt.ylabel('g')
plt.title('g vs. time')
# plt.legend(['vi', 'ospi'])
plt.show()


# K analyses
# import matplotlib.pyplot as plt
# MA = kpi_df['wait'].rolling(window=T).mean().iloc[T::T]
# plt.scatter(np.arange(len(MA)), MA)
# plt.show()
#
# plt.scatter(np.arange(len(MA)), MA.cumsum()/np.arange(len(MA)))
# plt.show()
#
# plt.scatter(np.arange(len(kpi_df)), kpi_df['g'])
# plt.show()

