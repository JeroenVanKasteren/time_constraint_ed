"""
This file contains the simulation of the multi-class queueing system.

TODO
How to simulate when OSPI idles?

Convert to a class when it works.
"""

# import argparse
import heapq as hq
import numba as nb
import numpy as np
# import pandas as pd
# from time import perf_counter as clock
from utils import TimeConstraintEDs as Env
from utils import OneStepPolicyImprovement as Ospi


N = 1000  # arrivals to simulate
batch_size = 100  # batch size for KPI
policy = 'fcfs'  # 'fcfs' 'sdf' 'sdf_prior' 'cmu' 'ospi'

J = 3
S = 5
gamma = 30
D = 100
t = env.t
lab = env.lab
# load: float = sum(lab / mu) / s.S
# s.imbalance = kwargs.get('imbalance',
#                          s.rng.uniform(s.imbalance_MIN,
#                                        s.imbalance_MAX, s.J))
# s.load = kwargs.get('load', s.rng.uniform(s.load_MIN, s.load_MAX))
# lab = mu * s.S * s.load * s.imbalance / sum(s.imbalance)

p_xy = env.p_xy
regret = np.max(r) - r + c
cmu = c * mu

if any((t % (1 / s.gamma) != 0) | (t < 1 / s.gamma)):
    t = np.floor(t * s.gamma) / s.gamma
    if s.trace:
        print('Rounded t down to nearest multiple of 1/gamma.\n')
lab = array(lab, float)
mu = array(mu, float)
t = array(t, float)
c = array(kwargs.get('c', array([1] * s.J)), float)
r = array(kwargs.get('r', array([1] * s.J)), float)

s.a = array(lab / mu, float)
s.s_star = array(s.server_allocation(), float)
s.rho = array(s.a / s.s_star, float)
s.pi_0 = s.get_pi_0(s.gamma, s.s_star, s.rho, s.lab)
s.tail_prob = s.get_tail_prob(s.gamma, s.s_star, s.rho, s.lab, s.mu,
                              s.pi_0, s.gamma * s.t)
s.g = s.get_g_app(s.pi_0, s.tail_prob)
s.tau = float(s.S * max(s.mu) + sum(np.maximum(s.lab, s.gamma)))

if 'D' in kwargs:
    s.D: int = kwargs.get('D')
else:
    s.D: int = s.get_D()
s.cap_prob_i = s.get_tail_prob(s.gamma, s.s_star, s.rho,
                               s.lab, s.mu, s.pi_0, s.D)
mu = sum(s.lab) / sum(s.lab / s.mu)
pi_0 = s.get_pi_0(s.gamma, s.S, s.load, sum(s.lab))
s.cap_prob = s.get_tail_prob(s.gamma, s.S, s.load, sum(s.lab),
                             mu, pi_0, s.D)
s.target_prob = s.get_tail_prob(s.gamma, s.S, s.load, sum(s.lab), mu,
                                pi_0, max(s.t))

s.p_xy = s.trans_prob(s.J, s.D, s.lab, s.gamma)

env = Env(J=J, S=S, D=D, gamma=gamma, t=t, c=c, r=r, P=inst.P,
          lab=inst.lab, mu=inst.mu, max_time=args.time,
          convergence_check=10, seed=42)

# @nb.njit
# def update_mean(mean, x, n):
#    """Welford's method to update the mean."""
#    return mean + (x - mean) / n  # avg_{n-1} = avg_{n-1} + (x_n - avg_{n-1})/n


def generate_times(J, N, lab, mu):
    """Generate exponential arrival and service times."""
    arrival_times = np.empty((J, N + 1), dtype=np.float32)  # +1, last arrival
    service_times = np.empty((J, N + 1), dtype=np.float32)
    for i in range(J):
        arrival_times[i, :] = env.rng.exponential(1 / lab[i], N + 1)
        service_times[i, :] = env.rng.exponential(1 / mu[i], N + 1)
    return arrival_times, service_times


def get_v_app(env):
    """Get the approximate value function for a given state."""
    v = np.zeros((env.J, env.D + 1))
    for i in range(env.J):
        v[i, ] = Ospi.get_v_app_i(env, i)
    return v


heap_type = nb.typeof((0.0, 0, 'event'))  # (time, class, event)
eye = np.eye(J, dtype=int)
v = get_v_app(env)
arrival_times, service_times = generate_times(J, N, lab, mu)
kpi = np.zeros((N+1, 3))  # time, class, waited

@nb.njit
def ospi(x, i):
    """One-step policy improvement.
    i indicates which class just arrived, i = J if no class arrived.
    """
    x = np.minimum(np.round(x / gamma), D)
    pi = J
    x_next = x if i == J else x + eye[i]
    v_sum = np.zeros(J)
    w_max = 0
    for j in range(J):
        w_max += v[j, x_next[j]]
        v_sum += v[j, x_next[j]]
        v_sum[j] -= v[j, x_next[j]]
    for j in range(J):  # Class to admit
        if (x[j] > 0) or (j == i):
            w = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
            w += p_xy[j, x[j], :x[j]+1] * (v_sum[j] + v[j, :x[j]+1])
            if w > w_max:
                pi = j
                w_max = w
    return pi


@nb.njit
def policy(x, i):
    if policy == 'ospi':
        return ospi(x, i)
    mask = x > 0  # If waiting
    mask[i] = True  # or just arrived
    if policy == 'fcfs':  # argmax(x)
        return np.nanargmax(np.where(mask, x, np.nan))
    elif policy == 'sdf':  # argmin(t - x)
        return np.nanargmin(np.where(mask, t - x, np.nan))
    elif policy == 'sdf_prior':
        y = t - x  # Time till deadline
        on_time = y >= 0
        if np.any(on_time):
            np.nanargmin(np.where(mask & on_time, t - x, np.nan))
        else:  # FCFS
            np.nanargmax(np.where(mask, x, np.nan))
    elif policy == 'cmu':
        return np.nanargmax(np.where(mask, cmu, np.nan))


@nb.njit
def admission(x, s, i, time, arr, n_admit, dep, arr_times, heap, kpi):
    """Assumes that sum(s)<S."""
    pi = policy(x, i)
    if pi < J:  # Take class pi into service, add its departure & new arrival
        kpi[n_admit, :] = time, pi, x[pi]
        n_admit += 1
        s += 1
        hq.heappush(heap, (time + service_times[pi, dep[pi]], pi, 'departure'))
        hq.heappush(heap,
                    (arr_times[pi] + arrival_times[pi, arr[pi]], pi, 'arrival'))
    else:  # Idle
        hq.heappush(heap,
                    (time + 1/gamma, i, 'idle'))
    return heap, kpi, n_admit, s


@nb.njit
def simulate_multi_class_system(kpi):
    """Simulate a multi-class system."""
    # initialize the system
    time = 0.0
    s = 0
    arr = np.ones(J, dtype=np.int32)
    n_admit = 0
    dep = np.zeros(J, dtype=np.int32)
    avg_wait = np.zeros(J, dtype=np.float32)
    arr_times = np.zeros(J, dtype=np.float32)
    heap = nb.typed.List.empty_list(heap_type)
    # initialize the event list
    for i in range(J):
        hq.heappush(heap, (arrival_times[i, 0], i, 'arrival'))
    # run the simulation
    while arr.sum() < N:
        # get next event
        event = hq.heappop(heap)
        time = event[0] if event[0] < time else time
        i = event[1]
        type_event = event[2]
        if type_event in ['arrival', 'idle']:  # arrival of FIL by design
            if type_event == 'arrival':
                arr[i] += 1
                arr_times[i] = event[0]
            if s < S:
                x = time - arr_times
                heap, kpi, n_admit, s = \
                    admission(x, s, i, time, arr, n_admit, dep, arr_times, heap,
                              kpi)
        elif type_event == 'departure':
            dep[i] += 1
            s[i] -= 1  # ensures that sum(s) < S
            x = time - arr_times
            heap, kpi, n_admit, s = \
                admission(x, s, i, time, arr, n_admit, dep, arr_times, heap,
                          kpi)
    return kpi

simulate_multi_class_system(kpi)

if __name__ == '__main__':
    main()
