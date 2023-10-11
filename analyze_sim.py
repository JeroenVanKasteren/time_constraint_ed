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

plt.scatter(kpi_df['time']/60, kpi_df['g'])
plt.xlabel('Running time (hours)')
plt.ylabel('g')
plt.title('g vs. time')
# plt.legend(['vi', 'ospi'])
plt.show()