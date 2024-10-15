"""
Description.

V_f is vectorized over all x (and not over all queues)
Quad_vec limits vectorizing over queues as the limits of the integration
cannot be vectors. Additionally, V then is a matrix with specific dimensions.
It is easier (and more readable and understandable) when we loop over J.

# -s <= x < 0,
g + tau*V(x) = lmbda * (r + V(x+1)) + (x + S)*mu*V(max(x-1,0)) + \
                (tau - lmbda - (x+S)*mu)*V(x)
note V[max(x+S-1,0)] maps V(-s-1) -> V(-s)=0
Moreover, when x=-s then (x+S)*mu=0
"""

import numpy as np
from utils import (tools,
                   plotting,
                   TimeConstraintEDs as Env,
                   OneStepPolicyImprovement as OSPI)

FILEPATH_RESULTS = 'results/'
FILEPATH_V = 'results/value_functions/'
np.set_printoptions(precision=4, linewidth=150, suppress=True)
tolerance = 1e-4
instance_id = 'J1'
weight_error = False
learner = OSPI()


def v_app_i_test(env, i, verbose=False):
    """Testing Poisson Equation equality."""
    s = env.s_star
    s_int = s.astype(int)
    lab = env.lab
    mu = env.mu
    tau = np.maximum(lab, env.gamma) + s * mu

    v = learner.get_v_app_i(env, i)
    x = np.arange(-s_int[i], env.D + 1)
    LHS = env.g[i] + tau[i] * v[x + s_int[i]]
    RHS = np.zeros(s_int[i] + env.D + 1)
    # -s <= x < 0, note V(-s)=0
    x = np.arange(-s_int[i], 0 + 1)
    RHS[x + s_int[i]] = (lab[i] * v[x + s_int[i] + 1]
                         + (x + s_int[i])
                         * mu[i] * v[np.maximum(x + s_int[i] - 1, 0)]
                         + (tau[i] - lab[i] - (x + s_int[i]) * mu[i])
                         * v[x + s_int[i]])
    x = np.arange(-s_int[i], 0)
    RHS[x + s_int[i]] +=  lab[i] * env.r[i]
    # x >= 1
    x = np.arange(1, env.D)
    RHS[x + s_int[i]] = (env.gamma * v[x + s_int[i] + 1]
                     + s[i] * mu[i]
                     * (env.r[i] + np.sum(env.p_xy[i, 1:env.D, :env.D]
                                          * v[s_int[i]:env.D + s_int[i]],
                                          1))
                     + (tau[i] - env.gamma - s[i] * mu[i]) * v[x + s_int[i]])
    # x > t * gamma
    x = np.arange(env.t[i] * env.gamma + 1, env.D).astype(int)
    RHS[x + s_int[i]] -= s[i] * mu[i] * env.c[i]
    # x = D, Note V[D+1] does not exist
    RHS[env.D + s_int[i]] = np.nan
    # Check if Poisson Equations hold
    x = np.arange(-s_int[i], env.D)
    all_close = np.allclose(LHS[x + s_int[i]], RHS[x + s_int[i]],
                            atol=tolerance)
    if verbose:
        print(f'g: {env.g[i]:0.4f}, '
              f'pi_0: {env.pi_0[i]:0.4f}, '
              f'tail_prob: {env.tail_prob[i]:0.4f}')
        print("gamma*t: ", env.gamma * env.t[i])
        print("LHS==RHS? ", all_close)
        print("x, LHS, RHS, V: \n",
              np.c_[np.arange(-s_int[i], env.D + 1), LHS, RHS, v])
    return all_close


def get_weighting_factor(env):
    """Calculate weighting factor of error (by tail prob)."""
    x_v = np.arange(env.D + 1)
    p_x_leq_t = np.ones((env.J, env.D + 1))
    w_factor = np.ones(env.dim)
    for j in range(env.J):
        p_x_leq_t *= env.get_tail_prob(env.gamma, env.S, env.rho[j], env.lab[j],
                                       env.mu[j], env.pi_0[j], x_v)
        states = [slice(None)] * (env.J * 2)
        for x in range(env.D + 1):
            states[j] = x
            w_factor[tuple(states)] *= p_x_leq_t[j, x]
    return w_factor


# Load in instance_set and extract methods used
instance_name = 'instances_' + instance_id
inst_set = tools.inst_load(FILEPATH_RESULTS + instance_name + '.csv')

# env = Env(J=1, S=3, load=0.75, gamma=20., D=100, P=1e3, e=1e-4)
# env = Env(J=2, S=2, gamma=20., P=1e3, e=1e-5,
#           lab=np.array([0.6726, 0.1794]), mu=np.array([0.8169, 0.2651]))

if instance_id == 'J1':
    for i, inst in inst_set.iterrows():
        env_i = Env(J=inst.J, S=inst.S, D=inst.D,
                    gamma=inst.gamma, t=inst.t, c=inst.c, r=inst.r,
                    mu=inst.mu, lab=inst.lab)
        if not v_app_i_test(env_i, 0):
            v_app_i_test(env_i, 0, verbose=True)
            break
    print("All tests passed (if nothing else printed).")

perc_improv = {'base': [], 'linear': [], 'abs': [], 'dp': []}
perc_improv_opt = {'base': [], 'linear': [], 'abs': [], 'dp': []}
for i, inst in inst_set.iterrows():
    env_i = Env(J=inst.J, S=inst.S, D=inst.D,
                gamma=inst.gamma, t=inst.t, c=inst.c, r=inst.r,
                mu=inst.mu, lab=inst.lab)

    w_factor = get_weighting_factor(env_i) if weight_error else 1

    file = '_'.join([instance_id, str(i), 'fcfs.npz'])
    v_fcfs = np.load(FILEPATH_V + 'v_' + file)['arr_0']
    file = '_'.join([instance_id, str(i), 'vi.npz'])
    v_vi = np.load(FILEPATH_V + 'v_' + file)['arr_0']

    v_app_old = learner.get_v_app_cons(env_i)
    error = np.sum(abs(v_app_old - v_fcfs)*w_factor)
    error_opt = np.sum(abs(v_app_old - v_vi)*w_factor)

    for method in perc_improv.keys():
        if method == 'base':
            v_app = learner.get_v_app(env_i)
        elif method == 'linear':
            v_app = learner.get_v_app_lin(env_i, type='linear')
        elif method == 'abs':
            v_app = learner.get_v_app_lin(env_i, type='abs')
        elif method == 'dp':
            v_app = learner.get_v_app_dp(env_i)
        m_error = np.sum(abs(v_app - v_fcfs)*w_factor)
        m_error_opt = np.sum(abs(v_app - v_vi)*w_factor)
        perc_improv[method].append((error - m_error) / error * 100)
        perc_improv_opt[method].append((error_opt - m_error_opt)
                                       / error_opt * 100)
        # if (error_opt - m_error_opt) / error_opt * 100 < -100:
        #     print(i, method, (error_opt - m_error_opt) / error_opt * 100)

plotting.multi_boxplot(perc_improv, perc_improv.keys(),
                       'Weighted improvement $V_{app}$ by interpolation, '
                       'rel to fcfs, inst: ' + instance_id,
                       perc_improv.keys(),
                       'improvement vs conservative (%)')

plotting.multi_boxplot(perc_improv_opt, perc_improv_opt.keys(),
                       'Weighted improvement $V_{app}$ by interpolation, '
                       'rel to VI, inst: ' + instance_id,
                       perc_improv_opt.keys(),
                       'improvement vs conservative (%)')

# Analyses checking v_app_dp
# i = 49
# inst = inst_set.iloc[i]
# env_i = Env(J=inst.J, S=inst.S, D=inst.D,
#             gamma=inst.gamma, t=inst.t, c=inst.c, r=inst.r,
#             mu=inst.mu, lab=inst.lab)
#
# w_factor = get_weighting_factor(env_i)
# file = '_'.join([instance_id, str(i), 'vi.npz'])
# v_vi = np.load(FILEPATH_V + 'v_' + file)['arr_0']
# v_app_old = learner.get_v_app_cons(env_i)
# error_opt = np.sum(abs(v_app_old - v_vi) * w_factor)
#
# v_app_lin = learner.get_v_app_lin(env_i, type='linear')
# v_dp = learner.calc_v_app_dp(env_i)
# v_app_dp = learner.get_v_app_dp(env_i)
#
# m_error_opt = np.sum(abs(v_app_lin - v_vi) * w_factor)
# print('lin improv: ', (error_opt - m_error_opt) / error_opt * 100)
# m_error_opt = np.sum(abs(v_app_dp - v_vi) * w_factor)
# print('dp improv: ', (error_opt - m_error_opt) / error_opt * 100)
#
# state = plotting.state_selection(env_i,
#                                  dim=True,
#                                  s=1,
#                                  wait_perc=0.7)
# plotting.plot_heatmap(env_i, state,
#                       V=v_vi,
#                       title='VI',
#                       t=inst.t * inst.gamma)
#
# plotting.plot_heatmap(env_i, state,
#                       V=v_app_lin,
#                       title='V_app_lin',
#                       t=inst.t * inst.gamma)
# plotting.plot_heatmap(env_i, state,
#                       V=v_app_dp,
#                       title='V_app_dp',
#                       t=inst.t * inst.gamma)
