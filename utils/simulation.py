import heapq as hq
import numpy as np
import os
import pandas as pd
import pickle as pkl
from time import perf_counter as clock
from utils import TimeConstraintEDs as Env, OneStepPolicyImprovement
from utils import tools


class Simulation:

    FILEPATH_INSTANCE = 'results/instance_sim_'
    FILEPATH_PICKLES = 'results/simulation_pickles/'
    FILEPATH_RESULT = 'results/simulation_pickles/result_'

    def __init__(self, **kwargs):
        inst = kwargs.get('inst')
        self.inst_id = kwargs.get('inst_id')
        self.method = kwargs.get('method')
        self.N = kwargs.get('N', 1e5)
        max_time = kwargs.get('time', '0-00:10:00')
        self.time_check = kwargs.get('time_check', 1e5)

        self.env = Env(J=inst.J, S=inst.S, D=inst.D,
                       gamma=inst.gamma, t=inst.t, c=inst.c, r=inst.r,
                       mu=inst.mu, lab=inst.lab,
                       seed=42,
                       max_time=max_time,
                       max_iter=self.N, sim='yes')
        self.J = self.env.J
        self.order = tools.fixed_order(self.env, self.method)

        if self.method[:4] == 'ospi':
            self.learner = OneStepPolicyImprovement()
            self.eye = np.eye(inst.J, dtype=int)
            self.v = self.get_v(self.env)
        self.arrival_times, self.service_times = \
            tools.generate_times(self.env, self.N)

    def get_v(self, env):
        """Get the value function for each class.
        Mimics get_v_app_... of OneStepPolicyImprovement."""
        if self.method == 'ospi_dp':
            return self.learner.calc_v_app_dp(env)

        v = np.zeros((env.J, env.D + 1, env.S + 1))
        s_ceil = np.ceil(env.s_star).astype(int)
        s_int = env.s_star.astype(int)
        for i in range(env.J):
            v_app_i = self.learner.get_v_app_i(env, i)
            if self.method in ["ospi", "ospi_cons"]:
                for x in range(env.D + 1):  # Value indep. of s
                    v[i, x, :] = v_app_i[x + s_ceil[i]]
                if self.method == 'ospi':
                    for s in range(s_ceil[i]):  # overwrite x=0, s<s*
                        v[i, 0, s] = v_app_i[s]
            else:  # ospi_lin, ospi_abs
                for x in range(env.D + 1):  # x_i = x
                    for s in range(env.S + 1):  # s_i = s
                        if s < env.s_star[i]:
                            if x == 0:
                                v[i, 0, s] = v_app_i[s]
                            else:  # x > 0
                                if self.method == 'ospi_abs':
                                    v[i, x, s] = (v_app_i[s]
                                                  + (v_app_i[x + s_ceil[i]]
                                                     - v_app_i[s_ceil[i]]))
                                else:
                                    dx = self.learner.calc_dx(env, i, x, s,
                                                              v_app_i)
                                    v[i, x, s] = (v_app_i[x + s_ceil[i] - 1]
                                                  + dx
                                                  * (s + env.S + 1 - s_int[i]))
                        else:  # s*_i <= s <= S
                            dx = self.learner.calc_dx(env, i, x, s, v_app_i)
                            v[i, x, s] = (v_app_i[x + s_ceil[i]]
                                          + dx * (s - s_int[i]))
        return v

    def get_v_app_dp(self, env, j, x, s):
        h, n = self.learner.calc_h(env, j, s)
        f = self.learner.calc_f(env, j, x, h, n)
        assert 0 <= f <= 1
        s_i = int(env.s_star[j])
        return f * self.v[j, x, s_i - n] + (1 - f) * self.v[j, x, s_i - n - 1]

    def ospi(self, fil, i, x, s):
        """
        Real time One-step policy improvement.
        Var i indicates which class just arrived, i = J if no class arrived.
        """
        x = np.minimum(np.round(x * self.env.gamma, 0),
                       self.env.D - 1).astype(int)
        pi = self.J
        # Arrival or gamma-jump for i, except after departure
        x_next = x if i == self.J else x + np.eye(self.J, dtype=int)[i]
        # v_sum is a container for the sum of the value functions
        v_sum = np.zeros(self.J)
        w_max = 0
        for j in range(self.J):
            if self.method == 'ospi_dp':
                # w_max represents not admitting any class
                w_max += self.get_v_app_dp(self.env, j, x_next[j], s)
                # v_sum uses only value of admitted class changes in p_xy V(y)
                v_sum += self.get_v_app_dp(self.env, j, x_next[j], s)
                v_sum[j] -= self.get_v_app_dp(self.env, j, x_next[j], s)
            else:
                w_max += self.v[j, x_next[j], s[j]]
                v_sum += self.v[j, x_next[j], s[j]]
                v_sum[j] -= self.v[j, x_next[j], s[j]]
        for j in range(self.J):  # Class to admit
            if fil[j]:
                if x[j] > self.env.gamma * self.env.t[j]:
                    w = self.env.r[j] - self.env.c[j]
                else:
                    w = self.env.r[j]
                if self.method == 'ospi_dp':
                    for y in range(x[j] + 1):
                        w += (self.env.p_xy[j, x[j], y]
                              * (v_sum[j]
                                 - self.get_v_app_dp(self.env, j, y,
                                                     s + self.eye[j])))
                else:
                    w += sum(self.env.p_xy[j, x[j], :x[j] + 1]
                             * (v_sum[j] - self.v[j, :x[j] + 1, s[j] + 1]))
                if w > w_max:
                    pi = j
                    w_max = w
        return pi

    def policy(self, fil, i, x, s):
        """Return the class to admit, assumes at least one FIL."""
        if self.method[:4] == 'ospi':
            return self.ospi(fil, i, x, s)
        elif self.method == 'fcfs':  # argmax(x)
            return np.nanargmax(np.where(fil, x, np.nan))
        elif self.method == 'sdf':  # argmin(t - x)
            return np.nanargmin(np.where(fil, self.env.t - x, np.nan))
        elif self.method == 'sdfprior':
            y = self.env.t - x  # Time till deadline
            on_time = y >= 0
            if np.any(fil & on_time):
                return np.nanargmin(np.where(fil & on_time,
                                             self.env.t - x, np.nan))
            else:  # FCFS
                return np.nanargmin(np.where(fil, x, np.nan))
        elif self.method in ['cmu_t_min', 'cmu_t_max', 'l_max', 'l_min']:
            return np.nanargmin(np.where(fil, self.order, np.nan))
        else:
            raise ValueError(f'Unknown policy: {self.method}')

    def admission(self, arr, arr_times, dep, fil, heap, i, kpi, n_admit, s,
                  time, x):
        """
        Assumes that sum(s)<S.
        if pi < J: Take class pi into service, add its departure & new arrival
        """
        pi = self.policy(fil, i, x, s)
        if pi < self.J:
            kpi[n_admit, :] = time, pi, x[pi]
            n_admit += 1
            s[pi] += 1
            fil[pi] = 0
            # print(f'Dep: {time + self.service_times[pi][dep[pi]]:.2f} of {pi} '
            #       f'Arr: {time + self.arrival_times[pi][arr[pi]]:.2f} of {pi}')
            hq.heappush(heap, (time + self.service_times[pi][dep[pi]],
                               pi, 'departure'))
            hq.heappush(heap,
                        (arr_times[pi] + self.arrival_times[pi][arr[pi]],
                         pi, 'arrival'))
            arr[pi] += 1
            dep[pi] += 1
        else:  # Idle
            # print(f'Idle: {time + 1/self.env.gamma:.2f}')
            hq.heappush(heap, (time + 1/self.env.gamma, i, 'idle'))
        return arr, dep, fil, heap, kpi, n_admit, s

    def simulate_multi_class_system(self, **kwargs):
        """Simulate a multi-class system.
        arr & dep keep track of arrivals and departures per class, both finished
        and planned.
        """
        arr_times = kwargs.get('arr_times', np.zeros(self.J))
        fil = kwargs.get('fil', np.zeros(self.J, dtype=int))
        heap = kwargs.get('heap', [])  # (time, class, event)
        kpi = kwargs.get('kpi', np.zeros((self.N + 1, 3)))
        n_admit = kwargs.get('n_admit', 0)
        s = kwargs.get('s', np.zeros(self.J, dtype=int))
        time = kwargs.get('time', 0.0)
        arr = kwargs.get('arr', np.zeros(self.J, dtype=int))
        dep = kwargs.get('dep', np.zeros(self.J, dtype=int))
        if len(heap) == 0:
            for i in range(self.J):  # initialize the event list
                # print(f'Arr: {self.arrival_times[i][0]:.2f} of {i} ')
                hq.heappush(heap, (self.arrival_times[i][0], i, 'arrival'))
                arr[i] += 1
        while n_admit < self.N:
            event = hq.heappop(heap)  # get next event
            time = event[0] if event[0] > time else time
            i = event[1]
            type_event = event[2]
            # print(f'fil: {fil}, s: {s}, time: {time:.2f}, event:', type_event, i)
            if type_event in ['arrival', 'idle']:  # arrival of FIL by design
                if type_event == 'arrival':
                    fil[i] = 1
                    arr_times[i] = event[0]
                if sum(s) < self.env.S:
                    x = time - arr_times
                    arr, dep, fil, heap, kpi, n_admit, s = self.admission(
                        arr, arr_times, dep, fil, heap, i, kpi, n_admit, s,
                        time, x)
            elif type_event == 'departure':
                s[i] -= 1  # ensures that sum(s) < S
                if sum(fil) > 0:
                    x = np.where(fil, time - arr_times, 0)
                    arr, dep, fil, heap, kpi, n_admit, s = self.admission(
                        arr, arr_times, dep, fil, heap, i, kpi, n_admit, s,
                        time, x)
            if (n_admit % self.time_check) == 0 and n_admit > 0:
                time_per = tools.sec_to_time((clock() - self.env.start_time)
                                             / n_admit * self.time_check)
                print(f'Sims done: {n_admit} (N={self.N}). Total time: '
                      f'{tools.sec_to_time(clock() - self.env.start_time)}, '
                      f'time per {self.time_check} iterations: '
                      f'{time_per}.')
                if (clock() - self.env.start_time) > self.env.max_time:
                    print(f'Time limit {self.env.max_time} reached, '
                          f'stop simulation.')
                    break
        return {'arr': arr, 'arr_times': arr_times, 'dep': dep, 'fil': fil,
                'heap': heap, 'kpi': kpi, 's': s, 'time': time}

    def run(self, continue_run=True):
        pickle_file = 'result_' + self.inst_id + '_' + self.method + '.pkl'
        if pickle_file in os.listdir(self.FILEPATH_PICKLES) and continue_run:
            pickle = pkl.load(open(self.FILEPATH_PICKLES + pickle_file, 'rb'))
            kpi = pickle['kpi']
            if isinstance(kpi, pd.DataFrame):
                kpi = kpi[:, :3].to_numpy()
            n_done = np.sum(kpi[:, 0] > 0)
            n_left = self.N - n_done
            if n_left > 2:
                if len(kpi) < self.N:
                    kpi = np.concatenate((kpi,
                                          np.zeros((self.N - len(kpi) + 1, 3))))
                to_pickle = self.simulate_multi_class_system(
                    arr=pickle['arr'],
                    arr_times=pickle['arr_times'],
                    dep=pickle['dep'],
                    fil=pickle['fil'],
                    heap=pickle['heap'],
                    kpi=kpi,
                    n_admit=n_done,
                    s=pickle['s'],
                    time=pickle['time'])
        else:
            to_pickle = self.simulate_multi_class_system()
            n_left = self.N
        if n_left > 0:
            time = clock() - self.env.start_time
            sims_done = np.sum(to_pickle['kpi'][:, 0] > 0)
            print(f'Sims done: {sims_done} '
                  f'(N={self.N}, n_left={n_left}). '
                  f'Total time: {tools.sec_to_time(time)}, '
                  f'time per 10,000 iterations: '
                  f'{tools.sec_to_time(time / n_left * 1e4)}.')
            pkl.dump(to_pickle, open(self.FILEPATH_PICKLES + pickle_file, 'wb'))
        else:
            print(f'Already done {self.N} sims.')
