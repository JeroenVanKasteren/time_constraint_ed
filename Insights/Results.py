"""
Load and visualize results.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columns = ['id', 'index', 'Date', 'seed', 'J', 'S', 'D', 'size', 'size_i',
           'gamma', 'eps', 't', 'c', 'r', 'lambda', 'mu', 'load', 'cap_prob',
           'cap_bool', 'vi_converged', 'ospi_converged', 'time', 'VI',
           'OSPI', 'gap']

results = pd.read_csv('Results/results.csv', names=columns)


def strip_split(x):
    return np.array([float(i) for i in x.strip('[]').split()])


cols = ['lambda', 'mu', 'cap_prob']
results.loc[:, cols] = results.loc[:, cols].applymap(strip_split)

# Mark bad results where weighted average of cap_prob is too big >0.05
# results['cap_bool'] = (results['cap_prob'] * results['lambda'] /
#                        results['lambda'].apply(sum)).apply(sum)
# results.to_csv('Results/results.csv', header=False, index=False)

results_conv = results[results['vi_converged'] & results['ospi_converged']]

plt.hist(results_conv['gap'])
plt.show()
results_conv.boxplot(column='gap', by='J')
plt.title('Optimality Gap versus queues')
plt.show()
# https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html

plt.violinplot(results_conv['gap'], showmedians=True)
plt.show()

plt.scatter(results_conv['time']/(60*60), results_conv['gap'])
plt.xlabel('Running time (hours)')
plt.ylabel('Optimality gap')
plt.title('Running time vs. gap')
plt.show()

plt.scatter(results_conv['time']/(60*60), results_conv['load'])
plt.xlabel('Running time (hours)')
plt.ylabel('Load')
plt.title('Running time vs. Load')
plt.show()

plt.scatter(results_conv['time']/(60*60), results_conv['cap_prob'].apply(max))
plt.xlabel('Running time (hours)')
plt.ylabel('max cap_prob')
plt.title('Running time vs. max cap_prob')
plt.show()

plt.scatter(results_conv['time']/(60*60), results_conv['size'])
plt.xlabel('Running time (hours)')
plt.ylabel('State space size')
plt.title('Running time vs. State space size')
plt.show()

plt.scatter(results_conv['time']/(60*60),
            results_conv['cap_prob'].apply(np.mean))
plt.xlabel('Running time (hours)')
plt.ylabel('average cap_prob')
plt.title('Running time vs. average cap_prob')
plt.show()

plt.scatter(results_conv['D'], results_conv['load'])
plt.xlabel('D')
plt.ylabel('load')
plt.title('D vs. load')
plt.show()

plt.scatter(results_conv['load'], results_conv['gap'])
plt.xlabel('load')
plt.ylabel('Optimality Gap')
plt.title('Load vs. Optimality Gap')
plt.show()

plt.scatter(results_conv['cap_prob'].apply(max), results_conv['gap'])
plt.xlabel('max cap_prob')
plt.ylabel('Optimality Gap')
plt.title('average cap_prob vs. max cap_prob')
plt.show()

plt.scatter(results_conv['cap_prob'].apply(np.mean), results_conv['gap'])
plt.xlabel('average cap_prob')
plt.ylabel('Optimality Gap')
plt.title('average cap_prob vs. Optimality Gap')
plt.show()

results_conv[results_conv['J'] == 2].boxplot(column='gap', by='S')
plt.title('Optimality Gap vs servers')
plt.show()
