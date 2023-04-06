"""
Load and visualize results.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob

columns = ['Date', 'J', 'S', 'D', 'gamma', 'eps',
           't', 'c', 'r', 'lambda', 'mu', 'Rho', 'cap_prob',
           'VI', 'OSPI', 'gap']

results = pd.read_csv('Results/results.csv', names=columns)

results.boxplot(column='gap', by='J')
plt.title('Optimality Gap')
plt.show()
# https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html





def merge_per_folder(folder_path, output_filename):
    """Merges first lines of text files in one folder, and
    writes combined lines into new output file

    Parameters
    ----------
    folder_path : str
        String representation of the folder path containing the text files.
    output_filename : str
        Name of the output file the merged lines will be written to.
    """
    # make sure there's a slash to the folder path
    folder_path += "" if folder_path[-1] == "/" else "/"
    # get all text files
    txt_files = glob.glob(folder_path + "*.txt")
    # get first lines; map to each text file (sorted)
    output_strings = map(read_first_line, sorted(txt_files))
    output_content = "".join(output_strings)
    # write to file
    with open(folder_path + output_filename, 'wt') as outfile:
        outfile.write(output_content)