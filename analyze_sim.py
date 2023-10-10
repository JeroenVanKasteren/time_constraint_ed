"""
Load and visualize results of simulation.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import tools

INSTANCE_ID = '01'
FILEPATH_INSTANCE = 'results/instances_' + INSTANCE_ID + '.csv'
# FILEPATH_READ = 'results/read/'
# FILEPATH_RESULT = 'results/'

start_K = 1e3
batch_T = 1e4

kpi_df = pd.DataFrame(kpi_np, columns=['time', 'class', 'wait'])
kpi_df = kpi_df[kpi_df['time'] != 0]
kpi_df = kpi_df.reindex(columns=[*kpi_df.columns.tolist(),
                                 'reward', 'g', 'target', 'avg_wait'])

inst = pd.read_csv(FILEPATH_INSTANCE)
cols = ['t', 'c', 'r', 'lab', 'mu']
inst.loc[:, cols] = inst.loc[:, cols].applymap(tools.strip_split)
inst['vi_time'] = inst['vi_time'].map(
    lambda x: x if pd.isnull(x) else tools.get_time(x))
inst['ospi_time'] = inst['ospi_time'].map(
    lambda x: x if pd.isnull(x) else tools.get_time(x))
inst_part = inst[pd.notnull(inst['ospi_g']) | pd.notnull(inst['vi_g'])]
inst_conv = inst[pd.notnull(inst['ospi_g']) & pd.notnull(inst['vi_g'])]

print('Solved vi: ' + str(inst['vi_g'].count()) + '\n' +
      'left vi: ' + str(len(inst) - inst['vi_g'].count()) + '\n' +
      'Solved ospi: ' + str(inst['ospi_g'].count()) + '\n' +
      'left ospi: ' + str(len(inst) - inst['ospi_g'].count()) + '\n' +
      'Solved both: ' + str(inst['opt_gap'].count()))

# Mark bad results where weighted average of cap_prob is too big >0.05
# results['w_cap_prob'] = (results['cap_prob'] * results['lambda']
#                          / results['lambda'].apply(sum)).apply(sum)

plt.hist(inst_conv['opt_gap'])
plt.show()

inst_conv.boxplot(column='opt_gap', by='gamma')
plt.ylabel('Opt_gap')
plt.title('Optimality Gap versus queues')
plt.show()
# https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html

plt.violinplot(inst_conv['opt_gap'], showmedians=True)
plt.show()

plt.scatter(inst_conv['vi_time']/(60*60), inst_conv['opt_gap'])
plt.scatter(inst_conv['ospi_time']/(60*60), inst_conv['opt_gap'])
plt.xlabel('Running time (hours)')
plt.ylabel('Optimality gap')
plt.title('Running time vs. gap')
plt.legend(['vi', 'ospi'])
plt.show()

inst_conv.boxplot(column=['vi_time', 'ospi_time'], by='load')
plt.ylabel('Running time (sec.)')
plt.title('Running time vs. Load')
plt.show()

plt.scatter(inst_conv['opt_gap']/(60*60), inst_conv['size'])
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

inst_conv.boxplot(column='opt_gap', by='load')
plt.xlabel('load')
plt.ylabel('Optimality Gap')
plt.title('Load vs. Optimality Gap')
plt.show()

inst_conv[inst_conv['J'] == 2].boxplot(column='opt_gap', by='S')
plt.title('Optimality Gap vs servers')
plt.show()

inst['solved'] = pd.notnull(inst['opt_gap'])
inst.boxplot(column='size', by='solved')
plt.ylabel('Optimality Gap')
plt.title('Size vs. Optimality Gap')
plt.show()

inst_unsolved = inst[pd.isnull(inst['ospi_g']) | pd.isnull(inst['vi_g'])]
unique_loads = inst['load'].unique()
print('unsolved: ', inst_unsolved['load'].value_counts().reindex(unique_loads, fill_value=0))
print('solved:', inst_conv['load'].value_counts().reindex(unique_loads, fill_value=0))

plt.bar(inst['load'].unique(), inst_unsolved['load'].value_counts().reindex(unique_loads, fill_value=0))
plt.bar(inst['load'].unique(), inst_conv['load'].value_counts().reindex(unique_loads, fill_value=0))
plt.legend()
plt.show()

