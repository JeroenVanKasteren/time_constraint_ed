"""
Sandbox One-step policy improvement

Note that V(x) for x<=0 is used to calculate V(0), which is used in V(x) x>0.
"""
#
# import numpy as np
# from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma
# from scipy.integrate import quad_vec
# from Sandbox_PI import init_w, init_pi, policy_improvement, policy_evaluation
# from env_and_learners.test_env import Env
# from Plotting import plot_pi, plot_v
#
# np.set_printoptions(precision=4, linewidth=150, suppress=True)
#
# np.random.seed(42)
# env = Env(J=2, S=2, gamma=20., P=1e3, e=1e-5, trace=True,
#           lab=np.array([0.6726, 0.1794]), mu=np.array([0.8169, 0.2651]),
#           convergence_check=10, print_modulo=10)
# # env = Env(J=1, S=1, mu=array([3]), lab=array([1]), t=array([1]), P=1e3,
# #           gamma=1, D=5, e=1e-4, trace=True, print_modulo=100,
# #           max_iter=5)
#
# # ----------------------- One Step Policy Improvement ----------------------
# name = 'One-step Policy Improvement'
# W = np.zeros(env.dim_i, dtype=np.float32)
# Pi = init_pi(env)
# g = 0
#
# env.timer(True, name, env.trace)
# V_app = get_v_app(env)
# W = init_w(env, V_app, W)
# V_app = V_app.reshape(env.size)
# W = W.reshape(env.size_i)
# Pi = Pi.reshape(env.size_i)
# Pi, _ = policy_improvement(V_app, W, Pi, env.J, env.D, env.gamma, env.KEEP_IDLE,
#                            env.d_i1, env.d_i2, env.d_f, env.p_xy)
# env.timer(False, name, env.trace)
#
# env.timer(True, name, env.trace)
# V_app = V_app.reshape(env.dim)
# W = W.reshape(env.dim_i)
# _, g = policy_evaluation(env, V_app, W, Pi, g, name, count=0)
# env.timer(False, name, env.trace)
#
# Pi = Pi.reshape(env.dim_i)
#
# print("V", V_app)
# print("Pi", Pi)
# print("g", g)
#

