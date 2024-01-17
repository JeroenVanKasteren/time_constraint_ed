import numpy as np
from utils import tools, TimeConstraintEDs as Env


def generate_instance(inst, sim_id):
    if sim_id in [1, 2, 3, 4, 10]:
        inst['J'] = 3
        inst['S'] = 5
        inst['gamma'] = 1
        inst['D'] = 180
        inst['r'] = [np.array([1, 1, 1]) for _ in range(len(inst))]
        inst['c'] = [np.array([1, 1, 1]) for _ in range(len(inst))]
        inst['imbalance'] = [np.array([18, 94, 172]) / 284 for r in
                             range(len(inst))]
        if sim_id == 1:
            inst['load'] = 0.75
        else:
            inst['load'] = 0.85
        if sim_id == 4:
            inst['t'] = [np.array([2, 2, 2]) / 60 for _ in range(len(inst))]
        elif sim_id == 10:
            inst['t'] = [np.array([60] * 3) for _ in range(len(inst))]
        else:
            inst['t'] = [np.array([10, 60, 120]) for _ in range(len(inst))]
        if sim_id in [3, 10]:
            inst['mu'] = [np.array([1, 1, 1]) / 60 for _ in range(len(inst))]
        else:
            inst['mu'] = [np.array([1 / 2.19, 1 / 2, 1 / 0.51]) / 60 for _ in
                          range(len(inst))]
    elif sim_id in [5, 6]:
        inst['J'] = 4
        inst['S'] = 10
        inst['t'] = [np.array([1] * 4) * 60 for _ in range(len(inst))]
        inst['D'] = 120
        inst['gamma'] = 1
        inst['c'] = [np.array([1, 1, 0.5, 0.5]) for _ in range(len(inst))]
        inst['r'] = [np.array([1] * 4) for _ in range(len(inst))]
        inst['mu'] = [np.array([1, 2, 1, 2]) / 60 for _ in range(len(inst))]
        inst['imbalance'] = [np.array([1 / 4, 1 / 4, 3 / 4, 3 / 4]) / 2
                             for _ in range(len(inst))]
        if sim_id == 5:
            inst['load'] = 0.8
        else:
            inst['load'] = 0.9
    elif sim_id in [7, 8]:
        inst['J'] = 6
        inst['S'] = 10
        inst['t'] = [np.array([1] * 6) * 60 for _ in range(len(inst))]
        inst['D'] = 120
        inst['gamma'] = 1
        inst['c'] = [np.array([1, 1, 1, 1, 1, 1]) for _ in range(len(inst))]
        inst['r'] = [np.array([1, 1, 1, 1, 1, 1]) for _ in range(len(inst))]
        inst['mu'] = [np.array([1, 2, 3, 4, 5, 6]) for _ in range(len(inst))]
        inst['imbalance'] = [np.array([1, 2, 3, 4, 5, 6]) / 21 for r in
                             range(len(inst))]
        if sim_id == 7:
            inst['load'] = 0.8
        else:
            inst['load'] = 0.9
    elif sim_id == 9:
        inst['J'] = 1
        inst['S'] = 5
        inst['t'] = [np.array([1] * 1) * 60 for _ in range(len(inst))]
        inst['D'] = 180
        inst['gamma'] = 1
        inst['c'] = [np.array([1]) for _ in range(len(inst))]
        inst['r'] = [np.array([1]) for _ in range(len(inst))]
        inst['mu'] = [np.array([1 / 30]) for _ in range(len(inst))]
        inst['load'] = 0.85
        inst['imbalance'] = [np.array([1]) for _ in range(len(inst))]
    if sim_id in [11, 12]:
        if sim_id == 11:
            row = 8
        elif sim_id == 12:
            row = 58
        inst_vi = tools.inst_load('results/instances_01.csv')
        for c_name in ['J', 'S', 'D', 'gamma', 'load']:
            inst[c_name] = inst_vi.loc[row][c_name]
        for c_name in ['t', 'c', 'r', 'mu', 'lab']:
            inst[c_name] = [inst_vi.loc[row][c_name] for _ in range(len(inst))]
    else:
        env = Env(J=inst['J'][0], S=inst['S'][0], D=inst['D'][0],
                  gamma=inst['gamma'][0],
                  t=inst['t'][0], c=inst['c'][0], r=inst['r'][0],
                  mu=inst['mu'][0],
                  load=inst['load'][0], imbalance=inst['imbalance'][0],
                  sim=True)
        inst['lab'] = [env.lab for _ in range(len(inst))]
    return inst
