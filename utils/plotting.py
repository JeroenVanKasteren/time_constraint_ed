"""
Colors:
https://matplotlib.org/stable/gallery/color/named_colors.html
https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib
https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
https://moonbooks.org/Articles/How-to-change-imshow-axis-values-labels-in-matplotlib-/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import utils
from matplotlib import colors


def plot_pi(env, PI_learner, Pi, zero_state, **kwargs):
    if zero_state:
        state = np.zeros(len(env.dim_i), 'int').astype(object)
        state[0] = 1 if 'smu' in kwargs else 0
    elif 'state' in kwargs:  # state = array([0,0,0,0]).astype(object)
        state = kwargs.get('state')
    else:  # Select a random valid states
        state = np.concatenate(([0],
                                env.x_states[
                                    np.random.randint(len(env.x_states))],
                                env.s_states[np.random.randint(
                                    len(env.s_states))])).astype(object)
        state[0] = 1 if 'smu' in kwargs else 0

    states = state.copy()
    print_states = state.astype('str')
    if ('i' in kwargs) & ('j' not in kwargs):
        i = kwargs.get('i')
        states[1 + i] = slice(None)  # x_i
        states[1 + i + env.J] = slice(None)  # s_i
        Pi_i = Pi[tuple(states)]
        print_states[1 + i] = ':'
        print_states[1 + i + env.J] = ':'
        max_ticks = 5
        title = 'Policy, queue: ' + str(i + 1) + ', ' + str(print_states)
        if 'learner' in kwargs:
            title = kwargs.get('learner') + ', ' + title
        x_label = 'Servers occupied by queue ' + str(i + 1)
        y_label = 'Waiting time state FIL queue ' + str(i + 1)
        x_ticks = np.arange(0, Pi_i.shape[1], max(1, np.ceil(Pi_i.shape[1]
                                                            / max_ticks)))
        y_ticks = np.arange(0, Pi_i.shape[0], max(1, np.ceil(Pi_i.shape[0]/10)))
    else:
        if ('i' in kwargs) & ('j' in kwargs):
            i = kwargs.get('i')
            j = kwargs.get('j')
        elif env.J == 2:
            i, j = [0, 1]
        else:
            i, j = np.sort(np.random.choice(env.J, 2, replace=False))
        states[1 + i] = slice(None)  # x_i
        states[1 + j] = slice(None)  # x_j
        Pi_i = Pi[tuple(states)]
        print_states[1 + i] = ':'
        print_states[1 + j] = ':'
        max_ticks = 10
        title = 'Policy, ' + str(print_states)
        x_label = 'Waiting time state FIL queue ' + str(j + 1)
        y_label = 'Waiting time state FIL queue ' + str(i + 1)
        x_ticks = np.arange(0, Pi_i.shape[1], max(1, np.ceil(Pi_i.shape[1]/10)))
        y_ticks = np.arange(0, Pi_i.shape[0], max(1, np.ceil(Pi_i.shape[0]/10)))
    if 'name' in kwargs:
        title = kwargs.get('name') + ', ' + title
    cols = ['black', 'grey', 'lightyellow', 'lightgrey']
    queues = ['darkblue', 'indigo', 'darkmagenta', 'mediumvioletred', 'crimson']
    cols.extend(queues[0:env.J])
    cmap = colors.ListedColormap(cols)  # Color list
    dic = {}
    for i in range(env.J):
        dic['Queue ' + str(i + 1)] = queues[i]
    dic['Keep Idle'] = cols[2]
    dic['None Waiting'] = cols[3]
    dic['Servers Full'] = cols[1]
    dic['Not Evaluated'] = cols[0]
    patches = [mpatches.Patch(edgecolor='black', facecolor=v, label=k)
               for k, v in dic.items()]

    bounds = [PI_learner.NOT_EVALUATED, PI_learner.SERVERS_FULL,
              PI_learner.KEEP_IDLE, PI_learner.NONE_WAITING]
    bounds.extend(np.arange(env.J + 1) + 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(Pi_i, origin='lower', cmap=cmap, norm=norm)
    plt.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax = plt.gca()
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticks(np.arange(Pi_i.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(Pi_i.shape[0] + 1) - 0.5, minor=True)
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.25)
    plt.show()


def plot_v(env, V, zero_state, **kwargs):
    if zero_state:
        state = np.zeros(len(env.dim), 'int').astype(object)
    elif 'state' in kwargs:  # state = array([0,0,0,0]).astype(object)
        state = kwargs.get('state')
    else:  # Select a random valid states
        state = np.concatenate(
            (env.x_states[np.random.randint(len(env.x_states))],
             env.S_states[np.random.randint(len(env.S_states))])).astype(object)
    states = state.copy()
    print_states = state.astype('str')
    if ('i' in kwargs) & ('j' not in kwargs):
        i = kwargs.get('i')
        states[i] = slice(None)  # x_i
        states[i + env.J] = slice(None)  # s_i
        print_states[i] = ':'
        print_states[i + env.J] = ':'
        max_ticks = 5
        title = 'V(x), queue: ' + str(i + 1) + ', ' + str(print_states)
        x_label = 'Servers occupied by queue ' + str(i + 1)
        y_label = 'Waiting time state FIL queue ' + str(i + 1)
    else:
        if ('i' in kwargs) & ('j' in kwargs):
            i = kwargs.get('i')
            j = kwargs.get('j')
        elif env.J == 2:
            i, j = [0, 1]
        else:
            i, j = np.sort(np.random.choice(env.J, 2, replace=False))
        states[i] = slice(None)  # x_i
        states[j] = slice(None)  # x_j
        print_states[i] = ':'
        print_states[j] = ':'
        max_ticks = 10
        title = 'V(x), ' + str(print_states)
        x_label = 'Waiting time state FIL queue ' + str(j + 1)
        y_label = 'Waiting time state FIL queue ' + str(i + 1)
    if 'name' in kwargs:
        title = kwargs.get('name') + ', ' + title
    V_i = V[tuple(states)]
    plt.imshow(V_i, origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, V_i.shape[1],
                            max(1, np.ceil(V_i.shape[1] / max_ticks))))
    ax.set_yticks(np.arange(0, V_i.shape[0], max(1, np.ceil(V_i.shape[0]/10))))
    ax.set_xticks(np.arange(V_i.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(V_i.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.1)
    plt.show()


def plot_multi_bar(filepath, instance_names, methods, kpi):
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    performances = {method: [[], []] for method in methods}
    min_y, max_y = 0, 0
    inst_nrs = [name.split('_')[2][:-4] for name in instance_names]
    for instance_name in instance_names:
        inst = utils.tools.inst_load(filepath + instance_name)
        for row_id, method in enumerate(methods):
            performances[method][0].extend([inst.loc[row_id, kpi]])
            performances[method][1].extend([inst.loc[row_id, 'ci_' + kpi]])
        min_y = np.min([min_y, np.min(inst[kpi] - inst['ci_' + kpi])])
        max_y = np.max([max_y, np.max(inst[kpi] + inst['ci_' + kpi])])
    min_y, max_y = (utils.tools.round_significance(min_y * 1.1),
                    utils.tools.round_significance(max_y * 1.1))
    x = np.arange(len(instance_names))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for method, [value, conf_int] in performances.items():
        value, conf_int = np.array(value), np.array(conf_int)
        offset = width * multiplier
        rects = ax.bar(x + offset, value, width, label=method, yerr=conf_int)
        ax.bar_label(rects, padding=3, fmt='{0:.3f}', fontsize=6, rotation=90)
        multiplier += 1
    ax.set_ylabel('g')
    ax.set_title('long term average reward')
    ax.set_xticks(x + width, inst_nrs)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(min_y, max_y)
    plt.show()

def plot_waiting(inst):
    x = np.arange(inst.J[0])
    ys = [i + x + (i * x) ** 2 for i in range(inst.J[0])]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ys)))
    for i in range(inst.loc[row_id, 'J']):
        mask = (kpi_df_tmp['class'] == i)
        plt.scatter(kpi_df_tmp.loc[mask, 'time']/60, kpi_df_tmp.loc[mask, 'wait'],
                    marker='x', label=i, color=colors[i])
        plt.axhline(y=inst.t[0][i], color=colors[i], linestyle='-')

    plt.xlabel('Time (hours)')
    plt.ylabel('wait')
    plt.title('Waiting time per class')
    plt.legend(loc='upper left')
    plt.show()
