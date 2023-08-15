"""
Process all unread result files
Load and visualize results.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import pandas as pd
import os

INSTANCE_ID = '01'
FILEPATH_INSTANCE = 'results/instances_' + INSTANCE_ID + '.csv'
FILEPATH_READ = 'results/read/'
FILEPATH_RESULT = 'results/'

inst = pd.read_csv(FILEPATH_INSTANCE)

for file in os.listdir(FILEPATH_RESULT):
    if not file.startswith('result_' + INSTANCE_ID):
        continue
    result = pd.read_csv(FILEPATH_RESULT + file, index_col=0)
    method = 'vi' if 'vi' in file else 'ospi'
    inst.loc[int(result[0]), method + '_job_id'] = result[method + '_job_id']
    inst.loc[int(result[0]), method + '_attempts'] += 1
    inst.loc[int(result[0]), method + '_time'] = result[method + '_time']
    inst.loc[int(result[0]), method + '_iter'] = result[method + 'iter']
    inst.loc[int(result[0]), method + '_g'] = result[method + 'g']

    if(pd.notnull(inst.loc[int(result[0]), 'ospi_g']) &
            pd.notnull(inst.loc[int(result[0]), 'vi_g'])):
        inst.loc[int(result[0]), 'opt_cap'] = (inst.loc[int(result[0]), 'ospi_g'] /
                                                  inst.loc[int(result[0]), 'vi_g'])

    os.rename(FILEPATH_RESULT + file, FILEPATH_READ + file)

inst.to_csv(FILEPATH_INSTANCE)
