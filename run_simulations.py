import numpy as np
from utils import tools, Simulation as Sim

FILEPATH_INSTANCE = 'results/instances_'
FILEPATH_PICKLES = 'results/simulation_pickles/'
FILEPATH_RESULT = 'results/simulation_pickles/result_'

# local simulation
instance_name = 'J2'
# inst = tools.inst_load(FILEPATH_INSTANCE + instance_name + '_sim.csv')
# for array_id in range(1, 3):  # len(inst) + 1):

# local / Debug
array_id = 1
args = {'job_id': 1,
        'array_id': array_id,  # array_id if local simulation
        'time': '0-00:10:00',
        'instance': instance_name,
        'method': 'not specified',
        'x': 0,
        'max_iter': '1e5',
        'continue_run': True}
args = tools.DotDict(args)
# args = tools.load_args()
args.max_iter = np.inf if args.max_iter == 'inf' else float(args.max_iter)
if args.method == 'not specified':
    methods = ['ospi', 'cmu_t_min', 'cmu_t_max', 'fcfs', 'sdf',
               'sdfprior', 'l_max', 'l_min']
else:
    methods = [args.method]

if np.isinf(args.max_iter):  # arrivals to simulate
    N = int(1e4)
else:
    N = int(args.max_iter)

inst = tools.inst_load(FILEPATH_INSTANCE + args.instance + '_sim.csv')
if args.array_id - 1 + args.x >= len(inst):
    print('No more instances to simulate.  Index:',
          args.array_id - 1 + args.x, flush=True)
    exit(0)
inst = inst.iloc[args.array_id - 1 + args.x]
inst_id = args.instance + '_' + str(args.array_id - 1 + args.x)

for method in methods:
    simulation = Sim(inst=inst,
                     inst_id=inst_id,
                     method=method,
                     N=N,
                     time=args.time)
    simulation.run(continue_run=args.continue_run)
