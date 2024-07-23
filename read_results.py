"""
Process all unread result files.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import pandas as pd
import os
import re
from utils import tools

# Debug
# args = {'instance': 'J2'}
# args = tools.DotDict(args)
args = tools.load_args()

FILEPATH_INSTANCES = 'results/instances_' + args.instance + '.csv'
FILEPATH_READ = 'results/read/'
FILEPATH_RESULT = 'results/'
OPT_METHOD = 'vi'

insts = pd.read_csv(FILEPATH_INSTANCES)
# tools.remove_empty_files(FILEPATH_READ)
# tools.remove_empty_files(FILEPATH_RESULT)

methods = ['_'.join(column.split('_')[:-2]) for column in insts.columns
           if column.endswith('job_id')]

# Process all unread result files
for file in os.listdir(FILEPATH_RESULT):
    if ((re.match('result_' + args.instance + r'_\d', file) is None) or
            (file.find('sim') >= 0)):
        continue
    result = pd.read_csv(FILEPATH_RESULT + file, index_col=0)
    i = int(result.loc['Unnamed: 0'].values[0])  # instance

    method = [method for method in methods
              if file.find('_' + method + '_job') >= 0][0]
    insts.loc[i, method + '_iter'] = int(result.loc[method + '_iter'].values[0])
    insts.loc[i, method + '_time'] = result.loc[method + '_time'].values[0]
    insts.loc[i, method + '_g_tmp'] = float(result.loc[method +
                                                       '_g_tmp'].values[0])
    if pd.isnull(insts.loc[i, method + '_g']):
        insts.loc[i, method + '_job_id'] = int(result.loc[method +
                                                          '_job_id'].values[0])
        insts.loc[i, method + '_attempts'] += 1
        # if _g empty returns nan
        insts.loc[i, method + '_g'] = float(result.loc[method + '_g'].values[0])
    else:
        print('Instance', args.instance + '_' + str(i),
              'already solved. Redundant job with id:',
              result.loc[method + '_job_id'].values[0], flush=True)
    os.rename(FILEPATH_RESULT + file, FILEPATH_READ + file)

# Calculate opt_gaps
methods.remove(OPT_METHOD)
for method in methods:
    for i in insts.index:
        insts.loc[i, method + '_opt_gap_tmp'] = \
            tools.opt_gap(insts.loc[i, method + '_g_tmp'],
                          insts.loc[i, OPT_METHOD + '_g_tmp'])
        insts.loc[i, method + '_opt_gap'] = \
            tools.opt_gap(insts.loc[i, method + '_g'],
                          insts.loc[i, OPT_METHOD + '_g'])

tools.solved_and_left(insts)
insts.to_csv(FILEPATH_INSTANCES, index=False)
