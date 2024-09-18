import numpy as np
import os

FILEPATH_V = 'results/value_functions/'

instances = ['J1', 'J2', 'J2_D_gam']

for file in os.listdir(FILEPATH_V):
    if ((file.startswith('v_J') or file.startswith('w_J'))
            and file.endswith('.npz')):
        m = np.float64(np.load(FILEPATH_V + file)['arr_0'])
        np.savez(FILEPATH_V + file, m)
