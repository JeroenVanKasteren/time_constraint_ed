"""
Load and visualize results.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import matplotlib.pyplot as plt
import pandas as pd
from utils import tools

INSTANCE_ID = '01'
FILEPATH_INSTANCE = 'results/instances_' + INSTANCE_ID + '.csv'

inst = tools.inst_load(FILEPATH_INSTANCE)
inst['vi_time'] = inst['vi_time'].map(
    lambda x: x if pd.isnull(x) else tools.get_time(x))
inst['ospi_time'] = inst['ospi_time'].map(
    lambda x: x if pd.isnull(x) else tools.get_time(x))
inst_conv = inst[pd.notnull(inst['ospi_g']) & pd.notnull(inst['vi_g'])]
inst_part = inst[pd.notnull(inst['ospi_g']) | pd.notnull(inst['vi_g'])]
inst_tmp = inst[pd.notnull(inst['ospi_g_tmp']) & pd.notnull(inst['vi_g_tmp'])]

tools.solved_and_left(inst)

# plt.hist(inst_conv['ospi_opt_gap'])
inst_conv.boxplot(column='ospi_opt_gap', by='gamma')
plt.ylabel('Opt_gap')
plt.title('Optimality Gap versus queues')
plt.show()
# https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
# plt.violinplot(inst_conv['ospi_opt_gap'], showmedians=True)

plt.scatter(inst_conv['vi_time']/(60*60), inst_conv['ospi_opt_gap'])
plt.scatter(inst_conv['ospi_time']/(60*60), inst_conv['ospi_opt_gap'])
plt.xlabel('Running time (hours)')
plt.ylabel('Optimality gap')
plt.title('Running time vs. gap')
plt.legend(['vi', 'ospi'])
plt.show()

inst_conv.boxplot(column=['vi_time', 'ospi_time'], by='load')
plt.ylabel('Running time (sec.)')
plt.suptitle('Running time (sec.) vs. Load')
plt.show()

plt.scatter(inst_conv['ospi_opt_gap']/(60*60), inst_conv['size'])
plt.xlabel('Opt_gap')
plt.ylabel('State space size')
plt.title('Opt gap vs. State space size')
plt.show()

plt.scatter(inst_conv['vi_time']/(60*60), inst_conv['size'])
plt.scatter(inst_conv['ospi_time']/(60*60), inst_conv['size'])
plt.xlabel('Running time (hours)')
plt.ylabel('State space size')
plt.title('Running time vs. State space size')
plt.legend(['vi', 'ospi'])
plt.show()

inst_conv.boxplot(column='ospi_opt_gap', by='load')
plt.xlabel('load')
plt.ylabel('Optimality Gap')
plt.title('Load vs. Optimality Gap')
plt.show()

inst_conv[inst_conv['J'] == 2].boxplot(column='ospi_opt_gap', by='S')
plt.title('Optimality Gap vs servers')
plt.show()

inst['solved'] = pd.notnull(inst['ospi_opt_gap'])
inst.boxplot(column='size', by='solved')
plt.ylabel('Size')
plt.title('Size per solved')
plt.show()

inst_unsolved = inst[pd.isnull(inst['ospi_g']) | pd.isnull(inst['vi_g'])]
unique_loads = inst['load'].unique()
solved = inst_conv['load'].value_counts().reindex(unique_loads, fill_value=0)
unsolved = inst_unsolved['load'].value_counts().reindex(unique_loads,
                                                        fill_value=0)
print('unsolved: ', unsolved)
print('solved:', solved)

plt.bar(unique_loads, solved, width=0.05)
plt.bar(unique_loads, unsolved, width=0.05, bottom=solved)
plt.show()
