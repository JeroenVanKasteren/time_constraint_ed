"""
Process all unread result files.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import pandas as pd
import os
from utils import tools

INSTANCE_ID = '03'  # instance set
FILEPATH_INSTANCE = 'results/instances_' + INSTANCE_ID + '.csv'
FILEPATH_READ = 'results/read/'
FILEPATH_RESULT = 'results/'
OPT_METHOD = 'vi'

inst = pd.read_csv(FILEPATH_INSTANCE)
tools.remove_empty_files(FILEPATH_READ)
tools.remove_empty_files(FILEPATH_RESULT)

# Process all unread result files
for file in os.listdir(FILEPATH_RESULT):
    if not file.startswith('result_' + INSTANCE_ID):
        continue
    print(file)
    result = pd.read_csv(FILEPATH_RESULT + file, index_col=0)
    i = int(result.loc['Unnamed: 0'][0])  # instance
    method = file.split('_')[3]
    inst.loc[i, method + '_iter'] = result.loc[method + '_iter'][0]
    inst.loc[i, method + '_time'] = result.loc[method + '_time'][0]
    inst.loc[i, method + '_g_tmp'] = result.loc[method + '_g_tmp'][0]
    if pd.isnull(inst.loc[i, method + '_g']):
        inst.loc[i, method + '_job_id'] = result.loc[method + '_job_id'][0]
        inst.loc[i, method + '_attempts'] += 1
        # if _g empty returns nan
        inst.loc[i, method + '_g'] = result.loc[method + '_g'][0]
    else:
        print('Instance', INSTANCE_ID + '_' + str(i),
              'already solved. Redundant job with id:',
              result.loc[method + '_job_id'][0], flush=True)
    os.rename(FILEPATH_RESULT + file, FILEPATH_READ + file)

# Calculate opt_gaps
methods = [column.split('_')[0] for column in inst.columns
           if column.endswith('job_id')]
methods.remove('vi')
for method in methods:
    for i in inst.index:
        inst.loc[i, method + '_opt_gap_tmp'] = \
            tools.opt_gap(inst.loc[i, method + '_g_tmp'],
                          inst.loc[i, OPT_METHOD + '_g_tmp'])
        inst.loc[i, method + '_opt_gap'] = \
            tools.opt_gap(inst.loc[i, method + '_g'],
                          inst.loc[i, OPT_METHOD + '_g'])

tools.solved_and_left(inst)
inst.to_csv(FILEPATH_INSTANCE, index=False)
