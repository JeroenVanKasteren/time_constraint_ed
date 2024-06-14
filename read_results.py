"""
Process all unread result files.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import pandas as pd
import os
from utils import tools

INSTANCE_ID = '02'
FILEPATH_INSTANCE = 'results/instances_' + INSTANCE_ID + '.csv'
FILEPATH_READ = 'results/read/'
FILEPATH_RESULT = 'results/'

inst = pd.read_csv(FILEPATH_INSTANCE)
tools.remove_empty_files(FILEPATH_READ)
tools.remove_empty_files(FILEPATH_RESULT)

# Process all unread result files
for file in os.listdir(FILEPATH_RESULT):
    if not file.startswith('result_' + INSTANCE_ID):
        continue
    result = pd.read_csv(FILEPATH_RESULT + file, index_col=0)
    index = int(result.loc['Unnamed: 0'][0])
    method = file.split('_')[3]
    inst.loc[index, method + '_iter'] = result.loc[method + '_iter'][0]
    inst.loc[index, method + '_time'] = result.loc[method + '_time'][0]
    inst.loc[index, method + '_g_tmp'] = result.loc[method + '_g_tmp'][0]
    if pd.isnull(inst.loc[index, method + '_g']):
        inst.loc[index, method + '_job_id'] = result.loc[method + '_job_id'][0]
        inst.loc[index, method + '_attempts'] += 1
        inst.loc[index, method + '_g'] = result.loc[method + '_g'][0]
    else:
        print('Instance', INSTANCE_ID + '_' + str(index),
              'already solved. Redundant job with id:',
              result.loc[method + '_job_id'][0], flush=True)
    if (pd.notnull(inst.loc[index, method + '_g_tmp']) &
            pd.notnull(inst.loc[index, 'vi_g_tmp'])):
        inst.loc[index, method + '_opt_gap_tmp'] = \
            (abs(float(inst.loc[index, method + '_g_tmp'])
                 - float(inst.loc[index, 'vi_g_tmp']))
             / float(inst.loc[index, 'vi_g_tmp']))
    if (pd.notnull(inst.loc[index, method + '_g']) &
            pd.notnull(inst.loc[index, 'vi_g'])):
        inst.loc[index, method + '_opt_gap'] = \
            (abs(float(inst.loc[index, method + '_g'])
                 - float(inst.loc[index, 'vi_g']))
             / float(inst.loc[index, 'vi_g']))
    os.rename(FILEPATH_RESULT + file, FILEPATH_READ + file)

tools.solved_and_left(inst)
inst.to_csv(FILEPATH_INSTANCE, index=False)
