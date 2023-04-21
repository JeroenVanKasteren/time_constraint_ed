"""
Load and visualize results.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import pandas as pd
import matplotlib.pyplot as plt

columns = ['id', 'index', 'Date', 'seed', 'J', 'S', 'D', 'gamma', 'eps',
           't', 'c', 'r', 'lambda', 'mu', 'Rho', 'cap_prob',
            'vi_converged', 'ospi_converged', 'time', 'VI', 'OSPI', 'gap']

results = pd.read_csv('Results/results.csv', names=columns)
results_conv = results[results['vi_converged'] & results['ospi_converged']]
results_conv.boxplot(column='gap', by='J')
plt.title('Optimality Gap')
plt.show()
# https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
