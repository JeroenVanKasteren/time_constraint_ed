"""
Sandbox Value Iteration
"""

import numpy as np
from Env_and_Learners import TimeConstraintEDs as Env, PolicyIteration, \
    ValueIteration
from Insights import plot_pi, plot_v
from time import perf_counter as clock

np.set_printoptions(precision=4, linewidth=150, suppress=True)

# -------------------------- Value Iteration --------------------------------
pi_learner = PolicyIteration()
env = Env(J=2, S=2, gamma=5, D=10, P=1e3, e=1e-5, seed=42,
          max_time='0-00:10:30', convergence_check=1, print_modulo=1,
          max_iter=100)
vi_learner = ValueIteration(env, pi_learner)

name = 'Value Iteration'
V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
W = np.zeros(env.dim_i, dtype=np.float32)
Pi = pi_learner.init_pi(env)

count = 0
stopped = False
converged = False

start_time = clock()
while not (stopped | converged):  # Update each state.
    W = pi_learner.init_w(env, V, W)
    V = V.reshape(env.size)
    W = W.reshape(env.size_i)
    W = vi_learner.get_w(V, W, env.J, env.D, env.gamma, env.d_i1, env.d_i2,
                         env.d_f1, env.P_xy)
    V = V.reshape(env.dim)
    W = W.reshape(env.dim_i)
    V_t = pi_learner.get_v(env, V, W)
    if count % env.convergence_check == 0:
        converged, stopped, g = pi_learner.convergence(env, V_t, V, count, name)
    V = V_t - V_t[tuple([0] * (env.J * 2))]  # Rescale and Save V_t
    count += 1
env.time_print(clock() - start_time)

# Determine policy via Policy Improvement.
W = pi_learner.init_w(env, V, W)
V = V.reshape(env.size)
W = W.reshape(env.size_i)
Pi = Pi.reshape(env.size_i)
Pi, _ = pi_learner.policy_improvement(V, W, Pi, env.J, env.D, env.gamma,
                                      env.KEEP_IDLE, env.d_i1, env.d_i2,
                                      env.d_f1, env.P_xy)
V = V.reshape(env.dim)
Pi = Pi.reshape(env.dim_i)

# print("V", V)
# print("Pi", Pi)
print("g", g)

if env.J > 1:
    plot_pi(env, env, Pi, zero_state=True)
    plot_pi(env, env, Pi, zero_state=False)
for i in range(env.J):
    plot_pi(env, env, Pi, zero_state=True, i=i)
    plot_pi(env, env, Pi, zero_state=True, i=i, smu=True)
    plot_v(env, V, zero_state=True, i=i)
