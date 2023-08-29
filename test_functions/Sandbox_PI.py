"""
Sandbox Policy Iteration
"""

import numpy as np
from env_and_learners import TimeConstraintEDs as Env, PolicyIteration
from Insights import plot_pi, plot_v
from time import perf_counter as clock

np.set_printoptions(precision=4, linewidth=150, suppress=True)

# -------------------------- Policy Iteration --------------------------
pi_learner = PolicyIteration()
env = Env(J=2, S=2, load=0.75, gamma=20., D=10, P=1e3, e=1e-5,
          convergence_check=10, print_modulo=10, seed=42)
name = 'Policy Iteration'
V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
W = np.zeros(env.dim_i, dtype=np.float32)
Pi = pi_learner.init_pi(env)
Pi = Pi.reshape(env.size_i)

g = 0
count = 0
stable = False

start_time = clock()
while not stable:
    V, g, _ = pi_learner.policy_evaluation(env, V, W, Pi, g,
                                           'Policy Evaluation of PI', count)
    W = pi_learner.init_w(env, V, W)
    V = V.reshape(env.size)
    W = W.reshape(env.size_i)
    Pi, stable = pi_learner.policy_improvement(V, W, Pi, env.J, env.D,
                                               env.gamma, env.KEEP_IDLE,
                                               env.d_i1, env.d_i2, env.d_f1,
                                               env.p_xy)
    V = V.reshape(env.dim)
    W = W.reshape(env.dim_i)
    if count > env.max_iter:
        break
    count += 1
env.time_print(clock() - start_time)

Pi = Pi.reshape(env.dim_i)

print("V", V)
print("Pi", Pi)
print("g", g)

if env.J > 1:
    plot_pi(env, env, Pi, zero_state=True)
    plot_pi(env, env, Pi, zero_state=False)
for i in range(env.J):
    plot_pi(env, env, Pi, zero_state=True, i=i)
    plot_pi(env, env, Pi, zero_state=True, i=i, smu=True)
    plot_v(env, V, zero_state=True, i=i)
