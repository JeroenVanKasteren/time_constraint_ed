import os

FILEPATH_INSTANCE = 'results/instance_sim_'
FILEPATH_PICKLES = 'results/simulation_pickles/'
FILEPATH_RESULT = 'results/simulation_pickles/result_'

instance_names = [f for f in os.listdir(FILEPATH_INSTANCE)
                  if f.startswith('instance_sim_')]
inst = utils.tools.inst_load(FILEPATH_INSTANCE + instance_names[0])
methods = inst['method'].values

# interested = [instance_names[i - 1] for i in [3, 9, 10, 11, 12]]
interested = [instance_names[i - 1] for i in [3]]
for instance_name in interested:
    inst = utils.tools.inst_load(FILEPATH_INSTANCE + instance_name)
    pickle_file = 'result_' + instance + '_' + method + '.pkl'
    if pickle_file in os.listdir(FILEPATH_PICKLES):
        arr_times, fil, heap, kpi, s, time = pkl.load(open(FILEPATH_PICKLES +
                                                           pickle_file, 'rb'))
        n_done = np.sum(kpi[:, 0] > 0)
        n_left = N - n_done
        if n_left > 2:
            if len(kpi) < N:
                kpi = np.concatenate((kpi, np.zeros((N - len(kpi) + 1, 3))))
            arr_times, fil, heap, kpi, s, time = simulate_multi_class_system(
                arr_times=arr_times,
                fil=fil,
                heap=heap,
                kpi=kpi,
                n_admit=n_done,
                s=s,
                time=time,
                sims=N)
    else:
        arr_times, fil, heap, kpi, s, time = simulate_multi_class_system()
        n_left = N
    if n_left > 0:
        time = clock() - env.start_time
        print(f'Sims done: {np.sum(kpi[:, 0] > 0)} (N={N}, n_left={n_left}). '
              f'Total time: {tools.sec_to_time(time)}, '
              f'time per 10,000 iterations: '
              f'{tools.sec_to_time(time / n_left * 1e4)}.')
        pkl.dump([arr_times, fil, heap, kpi, s, time], open(FILEPATH_PICKLES +
                                                            pickle_file, 'wb'))
    else:
        print(f'Already done {N} sims.')
