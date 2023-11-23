"""
Load and visualize results of simulation.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
from utils import tools

FILEPATH_INSTANCE = 'results/'
FILEPATH_READ = 'results/read/'
FILEPATH_PICKLES = 'results/simulation_pickles/'

instance_names = [f for f in os.listdir(FILEPATH_INSTANCE)
                  if f.startswith('instance_sim_')]
instance_names = instance_names[1:6]
inst_nrs = [name.split('_')[2][:-4] for name in instance_names]

inst = tools.inst_load(FILEPATH_INSTANCE + instance_names[0])
methods = inst['method'].values

"""
instance_name = instance_names[7]
inst = tools.inst_load(FILEPATH_INSTANCE + instance_name)
instance_id = instance_name.split('_')[2][:-4]
methods = inst['method'].values
method = methods[4]
row_id = 4
"""

performances = {method: [[], []] for method in methods}
min_y, max_y = 0, 0
for instance_name in instance_names:
    inst = tools.inst_load(FILEPATH_INSTANCE + instance_name)
    for row_id, method in enumerate(methods):
        performances[method][0].extend([inst.loc[row_id, 'g']])
        performances[method][1].extend([inst.loc[row_id, 'conf_int']])
    min_y = np.min([min_y, np.min(inst['g']-inst['conf_int'])])
    max_y = np.max([max_y, np.max(inst['g']+inst['conf_int'])])
min_y, max_y = np.floor(min_y*1.2), np.ceil(max_y*1.2)

x = np.arange(len(instance_names))  # the label locations
width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
for method, [g, conf_int] in performances.items():
    g, conf_int = np.array(g), np.array(conf_int)
    offset = width * multiplier
    rects = ax.bar(x + offset, g, width, label=method, yerr=conf_int)
    ax.bar_label(rects, padding=3, fmt='{0:.3f}', fontsize=6, rotation=90)
    multiplier += 1

ax.set_ylabel('g')
ax.set_title('long term average reward')
ax.set_xticks(x + width, inst_nrs)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(min_y, max_y)

plt.show()

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

instance_name = instance_names[0]
inst = tools.inst_load(FILEPATH_INSTANCE + instance_name)
instance_id = instance_name.split('_')[2][:-4]
methods = inst['method'].values
row_id = 0
method = methods[row_id]
file = 'result_' + instance_id + '_' + method + '.pkl'
arr_times, fil, heap, kpi_df, s, time = (
    pkl.load(open(FILEPATH_PICKLES + file, 'rb')))

start_K = 1e4

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

