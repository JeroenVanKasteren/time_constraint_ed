"""
Colors:
https://matplotlib.org/stable/gallery/color/named_colors.html
https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib
https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
https://moonbooks.org/Articles/How-to-change-imshow-axis-values-labels-in-matplotlib-/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D as lines
import numpy as np
import utils
from matplotlib import colors


def choosing_classes(env, **kwargs):
    if ('i' in kwargs) & ('j' in kwargs):
        return kwargs.get('i'), kwargs.get('j')
    elif env.J == 2:
        return [0, 1]
    else:
        return np.sort(np.random.choice(env.J, 2, replace=False))

def plot_pi(env, Pi, zero_state, **kwargs):
    """
    Plot policy Pi.

    :param env: Environment object to extract dimension
    :type env: utils.env.TimeConstraintEDs
    :param Pi: Numpy matrix, Policy to plot
    :type Pi: numpy.ndarray
    :param zero_state: Boolean, if True, plot x_i = 0 and s_i = 0
    :type zero_state: bool
    :key smu: bool, take policy after a departure if True
    :key state: Numpy array, state to plot (array([0,0,0,0]).astype(object))
    :key i: int, queue i to plot
    :key j: int, queue j to plot
    :key learner: String, name of the learner
    :key name: String, name of the plot
    :key t: Numpy array, target per class
    :key d_cap: int, time cap for plot
    :return: -

    If zero_state is True, take the state x_i = 0 and s_i = 0.
    If zero_state is False, take the given state.
    If zero_state is False and state is not given, take a random state.

    If smu is given, plot the policy after a departure.
    If i given and j not, take x_i and s_i as axis.
    If i and j are given or only 2 queues (J=2), take x_i and x_j as axis.
    Otherwise, choose 2 random queues.
    """

    if zero_state:
        state = np.zeros(len(env.dim_i), 'int').astype(object)
        state[0] = env.J + 1 if 'smu' in kwargs else 0
    elif 'state' in kwargs:
        state = kwargs.get('state')
    else:  # Select a random valid states
        x, s = (np.random.randint(len(env.x_states)),
                np.random.randint(len(env.s_states)))
        state = np.concatenate(([0],
                                env.x_states[x],
                                env.s_states[s])).astype(object)
        state[0] = env.J + 1 if 'smu' in kwargs else 0

    states = state.copy()
    print_states = state.astype('str')
    d_cap = kwargs.get('d_cap', 0)
    t = kwargs.get('t', [0])
    if ('i' in kwargs) & ('j' not in kwargs):
        i = kwargs.get('i')
        states[1 + i + env.J] = slice(None)  # s_i
        print_states[1 + i + env.J] = ':'
        max_ticks = 5
        title = 'Policy, queue: ' + str(i + 1) + ', ' + str(print_states)
        x_label = 'Servers occupied by queue ' + str(i + 1)
        if 't' in kwargs:
            t_x, t_y = [0, t[i]], [t[i], t[i]]
    else:
        i, j = choosing_classes(env, **kwargs)
        states[1 + j] = slice(d_cap) if 'D' in kwargs else slice(None)  # x_j
        print_states[1 + j] = ':'
        max_ticks = 10
        title = 'Policy, ' + str(print_states)
        x_label = 'Waiting time state FIL queue ' + str(j + 1)
        if 't' in kwargs:
            t_x, t_y = [0, t[i], t[i]], [t[i], t[i], 0]
    print_states[1 + i] = ':'
    y_label = 'Waiting time state FIL queue ' + str(i + 1)
    states[1 + i] = slice(d_cap) if 'D' in kwargs else slice(None)  # x_i
    pi_i = Pi[tuple(states)]
    x_ticks = np.arange(0, pi_i.shape[1], max(1, np.ceil(pi_i.shape[1]
                                                         / max_ticks)))
    y_ticks = np.arange(0, pi_i.shape[0], max(1, np.ceil(pi_i.shape[0]
                                                         / max_ticks)))
    if 'name' in kwargs:
        title = kwargs.get('name') + ', ' + title
    cols = ['black', 'grey', 'lightyellow', 'lightgrey']
    queues = ['blue', 'crimson', 'darkgreen', 'gold', 'teal']
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

    bounds = [env.NOT_EVALUATED, env.SERVERS_FULL,
              env.KEEP_IDLE, env.NONE_WAITING]
    bounds.extend(np.arange(env.J + 1) + 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(pi_i, origin='lower', cmap=cmap, norm=norm)
    if 't' in kwargs:
        lines(xdata=t_x, ydata=t_y, linewidth=0.5,
              linestyle='-.', color='green')
    plt.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax = plt.gca()
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticks(np.arange(pi_i.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(pi_i.shape[0] + 1) - 0.5, minor=True)
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
    if 'cap_d' in kwargs:
        cap_d = kwargs.get('cap_d')
        ax.set_ylim(0, cap_d)
        if not (('i' in kwargs) & ('j' not in kwargs)):
            ax.set_xlim(0, cap_d)
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
    t = kwargs.get('t', [0])
    if ('i' in kwargs) & ('j' not in kwargs):
        i = kwargs.get('i')
        states[i + env.J] = slice(None)  # s_i
        print_states[i + env.J] = ':'
        max_ticks = 5
        title = 'V(x), queue: ' + str(i + 1) + ', ' + str(print_states)
        x_label = 'Servers occupied by queue ' + str(i + 1)
        y_label = 'Waiting time state FIL queue ' + str(i + 1)
        if 't' in kwargs:
            t_x, t_y = [0, t[i]], [t[i], t[i]]
    else:
        i, j = choosing_classes(env, **kwargs)
        states[j] = slice(None)  # x_j
        print_states[i] = ':'
        print_states[j] = ':'
        max_ticks = 10
        title = 'V(x), ' + str(print_states)
        x_label = 'Waiting time state FIL queue ' + str(j + 1)
        y_label = 'Waiting time state FIL queue ' + str(i + 1)
        if 't' in kwargs:
            t_x, t_y = [0, t[i], t[i]], [t[i], t[i], 0]
    if 'name' in kwargs:
        title = kwargs.get('name') + ', ' + title
    states[i] = slice(None)  # x_i
    print_states[i] = ':'
    V_i = V[tuple(states)]

    plt.imshow(V_i, origin='lower')
    if 't' in kwargs:
        lines(xdata=t_x, ydata=t_y, linewidth=0.5,
              linestyle='-.', color='green')
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


def plot_multi_bar(filepath, instance_names, methods, kpi, normalize=False,
                   width=0.1):
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    performances = {method: [[], []] for method in methods}
    min_y, max_y = 0, 0
    inst_nrs = [name.split('_')[2][:-4] for name in instance_names]
    for instance_name in instance_names:
        inst = utils.tools.inst_load(filepath + instance_name)
        optimal = sum(inst.r[0] * inst.lab[0])
        for row_id, method in enumerate(methods):
            if normalize:
                performances[method][0].extend(
                    [inst.loc[row_id, kpi] / optimal])
                performances[method][1].extend(
                    [inst.loc[row_id, 'ci_' + kpi] / optimal])
            else:
                performances[method][0].extend([inst.loc[row_id, kpi]])
                performances[method][1].extend([inst.loc[row_id, 'ci_' + kpi]])
        min_y = np.min([min_y, np.min(inst[kpi] - inst['ci_' + kpi])])
        max_y = np.max([max_y, np.max(inst[kpi] + inst['ci_' + kpi])])
    if normalize:
        min_y, max_y = 0, 1
    else:
        min_y, max_y = (utils.tools.round_significance(min_y * 1.1),
                        utils.tools.round_significance(max_y * 1.1))
    x = np.arange(len(instance_names))  # the label locations
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for method, [value, conf_int] in performances.items():
        value, conf_int = np.array(value), np.array(conf_int)
        offset = width * multiplier
        rects = ax.bar(x + offset, value, width, label=method, yerr=conf_int)
        ax.bar_label(rects, padding=3, fmt='{0:.3f}', fontsize=6, rotation=90)
        multiplier += 1
    ax.set_ylabel('g')
    if kpi == 'perc':
        ax.set_ylabel('Percentage of arrivals served on time')
    else:
        if normalize:
            ax.set_ylabel('Optimality gap of g')
        else:
            ax.set_ylabel('Long term average reward')
    ax.set_xticks(x + width, inst_nrs)
    ax.legend(loc='lower left', ncols=3)
    ax.set_ylim(min_y, max_y)
    plt.show()


def plot_waiting(inst_row, kpi_df_full, size, start):
    kpi_df = kpi_df_full[start:start + size]
    x = np.arange(inst_row.J)
    ys = [i + x + (i * x) ** 2 for i in range(inst_row.J)]
    colors_tmp = plt.cm.rainbow(np.linspace(0, 1, len(ys)))
    for i in range(inst_row.J):
        mask = (kpi_df['class'] == i)
        plt.scatter(kpi_df.loc[mask, 'time']/60,
                    kpi_df.loc[mask, 'wait'],
                    marker='x', label=i, color=colors_tmp[i])
        plt.axhline(y=inst_row.t[i], color=colors_tmp[i], linestyle='-')
    plt.xlabel('Time (hours)')
    plt.ylabel('wait')
    plt.title('Waiting time per class')
    plt.legend(loc='upper right')
    plt.show()


def plot_convergence(kpi_df, method, k, t, m=100):
    MA, times = utils.tools.moving_average(kpi_df, k, m, t)
    plt.scatter(times.cumsum() / 60, MA/times, label='Moving Average')
    plt.scatter(times.cumsum() / 60,
                MA.cumsum() / times.cumsum(),
                label='g')
    plt.xlabel('Running time (hours)')
    plt.ylabel('g')
    plt.title('g vs. time for ' + method)
    plt.legend(loc='upper right')
    plt.show()


# def fil_plot():
#     arrivals = kpi[ kpi['event'] = arrival]
#     time = time of service initiation (end of waiting)
#     arrivals['arrival_time'] = time - waiting time
#     end point (blue)
#     x = time
#     y = waiting time
#
#     time_last = time of service initiation of last arrival
#     diff = time_last - arrival time
#     diff > 0: current arrival has waited
#     diff < 0: current arrival did not wait
#
#     Start point
#     x = where(diff > 0, time_last, arrival_time)
#     y = where(diff < 0, diff, 0)