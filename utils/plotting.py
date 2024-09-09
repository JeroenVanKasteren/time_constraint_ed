"""
Colors:
https://matplotlib.org/stable/gallery/color/named_colors.html
https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib
https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
https://moonbooks.org/Articles/How-to-change-imshow-axis-values-labels-in-matplotlib-/
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D as lines
import numpy as np
import utils
from matplotlib import colors


def choosing_classes(env, **kwargs):
    """Assumed J > 1."""
    if ('i' in kwargs) & ('j' in kwargs):
        return kwargs.get('i'), kwargs.get('j')
    elif env.J == 2:
        return [0, 1]
    else:
        return np.sort(np.random.choice(env.J, 2, replace=False))


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


def multi_boxplot(gap, keys, title, x_ticks, y_label, violin=False,
                  rotation=20, left=0.1, bottom=0.1,
                  **kwargs):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=left, bottom=bottom)
    for i, key in enumerate(keys):
        if violin:
            if len(gap[key]) == 0:  # Edge case for empty data
                gap[key][0] = 0
            ax.violinplot(gap[key], positions=[i], showmedians=True)
        else:
            ax.boxplot(gap[key], positions=[i], tick_labels=[key])
    if 'y_lim' in kwargs:
        ax.set_ylim(kwargs.get('y_lim'))
    if kwargs.get('log_y', False):
        ax.set_yscale('log')
    plt.axhline(0)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    if 'x_label' in kwargs:
        ax.set_xlabel(kwargs.get('x_label'))
    if violin:
        ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks, rotation=rotation)
    plt.show()


def plot_gap(data, methods,
             meth_v, comp_m, comp_v, ref_m, ref_v, title,
             violin=False, multi_xyc=False, title_xyc='',
             x_lab='size', y_lab='smu(1-rho)',
             rotation=20, left=0, bottom=0):
    """
    Boxplot (optimality) gap of value (v) for methods in comparison to value of
    a comparison method, relative to value of reference method.
    (A method is not compared against itself.)
    """

    gap = {}
    for method in methods:
        if comp_m + comp_v == method + meth_v:
            continue
        subset_cols = list({comp_m + comp_v, method + meth_v, ref_m + ref_v})
        subset = data[subset_cols].dropna()
        gap[method] = ((subset[comp_m + comp_v] - subset[method + meth_v])
                       / subset[ref_m + ref_v]) * 100
        if multi_xyc:
            subset_cols.extend([x_lab, y_lab])
            subset = data[subset_cols].dropna()
            plot_xyc(subset[x_lab],
                     subset[y_lab],
                     gap[method],
                     title=title_xyc + ' ' + comp_m + ' vs ' + method,
                     x_lab=x_lab,
                     y_lab=y_lab,
                     c_lab='gap (%)')
    multi_boxplot(gap, gap.keys(), title, gap.keys(),
                  'gap (%)', violin=violin,
                  rotation=rotation, left=left, bottom=bottom)


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


def state_selection(env,
                    dim_i=False,
                    x=None,  # None, 'random', array
                    s=None,  # None, 'random', number, array
                    dep_arr=0,  # state[0] = i (arrival) / env.J (departure)
                    wait_perc=None):  # percentage of target, float or array
    if x == 'random':
        x = env.x_states[np.random.randint(len(env.x_states))]
    elif wait_perc is not None:
        x = (np.ones(env.J) * env.t * env.gamma * wait_perc).astype(int)
    elif x is None:
        x = np.zeros(env.J)

    if s == 'random':
        s = env.s_states[np.random.randint(len(env.s_states))]
    elif s is None:
        s = np.zeros(env.J)
    elif type(s) is int:  # Divide servers equally, with s spare servers
        s = (np.arange(env.J) < (env.S - s) % env.J) + ((env.S - s) // env.J)

    assert len(x) == env.J, 'x has wrong dimension'
    assert len(s) == env.J, 's has wrong dimension'
    if dim_i:  # if a policy (# dim_i uneven, dim even)
        state = np.concatenate([x, s]).astype(object)
    else:
        state = np.concatenate(([dep_arr], x, s))
    return state


def plot_heatmap(env, state, **kwargs):
    """
    Plot policy Pi or value V or W.

    :param env: Environment object to extract dimension
    :type env: utils.env.TimeConstraintEDs
    :param state: State (i, x, s)
    :type state: numpy.ndarray
    :key Pi/V/W: numpy.ndarray, data to plot
    :key i: int, queue i to plot
    :key j: int, queue j to plot
    :key title: String, title of the plot
    :key t: Numpy array, target per class
    :key d_cap: int, time cap for plot
    :return: -

    If i given and j not, take x_i and s_i as axis.
    If i and j are given or only 2 queues (J=2), take x_i and x_j as axis.
    Otherwise, choose 2 random queues.
    """
    title = kwargs.get('title', '')
    d_cap = kwargs.get('d_cap', 0)
    t = kwargs.get('t', [0])
    event = 'V' not in kwargs

    states = state.copy()
    print_states = [''] * (event + 2 * env.J)
    for i in range(env.J):
        print_states[event + i] = '$x_{' + str(i + 1) + '}$=' + str(states[i])
        print_states[event + i + env.J] = ('$s_{' + str(i + 1) + '}=' +
                                           str(states[i]))
    if (event == 1) and (states[0] < env.J):
        print_states[0] = '$arr_{' + str(states[0] + 1) + '}$'
    elif (event == 1) and (states[0] == env.J):
        print_states[0] = 'dep'

    if (('i' in kwargs) & ('j' not in kwargs)) or env.J == 1:
        i = kwargs.get('i', 0)
        x_label = 'Servers occupied by queue ' + str(i + 1)
        states[event + i + env.J] = slice(None)  # s_i
        print_states[event + i + env.J] = '$\\forall  s_{' + str(i + 1) + '}$'
        max_ticks = kwargs.get('max_ticks', 5)
        if 't' in kwargs:
            t_x, t_y = [-0.5, env.S + 0.5], [t[i] + 0.5, t[i] + 0.5]
    else:
        i, j = choosing_classes(env, **kwargs)
        x_label = 'Waiting time state FIL queue ' + str(j + 1)
        states[event + j] = slice(d_cap) if d_cap > 0 else slice(None)  # x_j
        print_states[event + j] = '$\\forall x_{' + str(j + 1) + '}$'  # x axis
        max_ticks = kwargs.get('max_ticks', 10)
        if 't' in kwargs:
            t_x = [0, t[i] + 0.5, t[i] + 0.5]
            t_y = [t[i] + 0.5, t[i] + 0.5, 0]
    y_label = 'Waiting time state FIL queue ' + str(i + 1)
    states[event + i] = slice(d_cap) if d_cap > 0 else slice(None)  # x_i
    print_states[event + i] = '$\\forall x_{' + str(i + 1) + '}$'  # y axis
    title = title + ' [' + ', '.join(print_states) + ']'

    if 'V' in kwargs:
        data = kwargs.get('V')[tuple(states)]
    elif 'Pi' in kwargs:
        data = kwargs.get('Pi')[tuple(states)]
    elif 'W' in kwargs:
        data = kwargs.get('W')[tuple(states)]

    fig, ax = plt.subplots()
    if 't' in kwargs:
        ax.plot(t_x, t_y, linewidth=2, linestyle='--',
                color='green')
    if 'Pi' in kwargs:
        color_list = ['black', 'grey', 'lightyellow', 'lightgrey']
        queues = ['blue', 'crimson', 'darkgreen', 'gold', 'teal']
        color_list.extend(queues[0:env.J])
        cmap = colors.ListedColormap(color_list)  # Color list
        dic = {'Not Evaluated': color_list[0],
               'Servers Full': color_list[1],
               'Keep Idle': color_list[2],
               'None Waiting': color_list[3]}
        for i in range(env.J):
            dic['Queue ' + str(i + 1)] = queues[i]
        patches = [mpatches.Patch(edgecolor='black', facecolor=v, label=k)
                   for k, v in dic.items()]
        bounds = ([env.NOT_EVALUATED, env.SERVERS_FULL,
                   env.KEEP_IDLE, env.NONE_WAITING]
                  + list(range(1, env.J + 2)))
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(data, origin='lower', cmap=cmap, norm=norm,
                  aspect='auto',  # allows rectangles (instead of only squares)
                  interpolation='none')  # no interpolation
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5))
    else:  # 'V' or 'W' in kwargs
        im = ax.imshow(data, origin='lower', cmap='coolwarm', aspect='auto',
                       interpolation='none')  # no interpolation
        plt.colorbar(im)

    x_ticks = np.arange(0, data.shape[1],
                        max(1, np.ceil(data.shape[1] / max_ticks)))
    y_ticks = np.arange(0, data.shape[0],
                        max(1, np.ceil(data.shape[0] / max_ticks)))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
    if d_cap > 0:
        ax.set_ylim(0, d_cap)
        if not (('i' in kwargs) & ('j' not in kwargs)):
            ax.set_xlim(0, d_cap)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.2)
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


def plot_xyc(x, y, c, title='', x_lab='x', y_lab='y', c_lab='', c_rot=270,
             c_map='coolwarm', **kwargs):
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=c, cmap=mpl.colormaps[c_map],
                    vmin=kwargs.get('vmin', min(c)),
                    vmax=kwargs.get('vmax', max(c)))
    cbar = fig.colorbar(sc, ax=ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(c_lab, rotation=c_rot)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.title.set_text(title)
    plt.show()
