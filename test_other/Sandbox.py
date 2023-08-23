"""
Sandbox
"""

import pandas as pd

inst = pd.read_csv('results/instances_01.csv')
print('Solved ospi: ' + str(inst['ospi_g'].count()) + '\n' +
      'left ospi: ' + str(len(inst) - inst['ospi_g'].count()) + '\n' +
      'Solved vi: ' + str(inst['vi_g'].count()) + '\n' +
      'left vi: ' + str(len(inst) - inst['vi_g'].count()) + '\n' +
      'Solved both: ' + str(inst['opt_gap'].count()))
