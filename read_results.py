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

# Remove empty ones
for file in os.listdir(FILEPATH_READ):
    if os.path.getsize(os.path.join(FILEPATH_READ, file)) == 0:
        os.remove(os.path.join(FILEPATH_READ, file))

# Process all unread result files
for file in os.listdir(FILEPATH_RESULT):
    if not file.startswith('result_' + INSTANCE_ID):
        continue
    result = pd.read_csv(FILEPATH_RESULT + file, index_col=0)
    index = int(result.loc['Unnamed: 0'][0])
    method = 'vi' if 'vi' in file else 'ospi'
    if pd.isnull(inst.loc[index, method + '_g']):
        inst.loc[index, method + '_job_id'] = result.loc[method + '_job_id'][0]
        inst.loc[index, method + '_attempts'] += 1
        inst.loc[index, method + '_time'] = result.loc[method + '_time'][0]
        inst.loc[index, method + '_iter'] = result.loc[method + '_iter'][0]
        inst.loc[index, method + '_g'] = result.loc[method + '_g'][0]
    else:
        print('Instance', INSTANCE_ID + '_' + str(index),
              'already solved. Redundant job with id:',
              result.loc[method + '_job_id'][0])
    if(pd.notnull(inst.loc[index, 'ospi_g']) &
            pd.notnull(inst.loc[index, 'vi_g'])):
        inst.loc[index, 'opt_gap'] = (abs(float(inst.loc[index, 'ospi_g'])
                                          - float(inst.loc[index, 'vi_g']))
                                      / float(inst.loc[index, 'vi_g']))
    os.rename(FILEPATH_RESULT + file, FILEPATH_READ + file)

inst.to_csv(FILEPATH_INSTANCE, index=False)
