"""
Load and visualize results.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('Resutls/results.csv')
# columns=['J', 'S', 'D', 'gamma', 'eps',
# 't', 'c', 'r', 'lambda', 'mu', 'Rho', 'cap_prob',
# 'VI', 'OSPI', 'gap']
results.boxplot(column='gap', by='J')
plt.title('Optimality Gap')
plt.show()
# https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
