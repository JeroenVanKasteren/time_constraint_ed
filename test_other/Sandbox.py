# # run firt 110 lines of analyze.py
#
# from utils import ValueIteration
#
# v_learner = ValueIteration()
#
# inst_id = 42
# inst = inst_set_gammas.loc[inst_id]
# env = Env(J=inst.J, S=inst.S, D=inst.D,
#           gamma=inst.gamma, t=inst.t, c=inst.c, r=inst.r,
#           mu=inst.mu, lab=inst.lab)
# zero_state = tuple(np.zeros(len(env.dim), dtype=int))
#
# # load in ref V, w, Pi, v_app
# file = '_'.join([instance_id, str(inst_id), ref_method + '.npz'])
# g_ref = round(inst[ref_method + g_tmp], dec)
# V = np.load(FILEPATH_V + 'v_' + file)['arr_0']
# Pi = None
# if 'pi_' + file in os.listdir(FILEPATH_V):
#     Pi = np.load(FILEPATH_V + 'pi_' + file)['arr_0']
#     w = np.load(FILEPATH_V + 'w_' + file)['arr_0']
# elif calculate_pi:
#     pi_learner = PolicyIteration()
#     pi_learner.one_step_policy_improvement(env, V)
#     Pi = pi_learner.Pi
#     w = pi_learner.W
# method = 'check'
#
# def get_w(V, W, J, D, gamma, d_i, d_i2, d_f1, p_xy):
#     """W given policy."""
#     sizes_x = d_i['sizes_i'][1:J + 1]
#     sizes_s = d_i['sizes_i'][J + 1:J * 2 + 1]
#     sizes_x_n = d_i['sizes'][0:J]  # sizes Next state
#     sizes_s_n = d_i['sizes'][J:J * 2]
#     r = d_f1['r']
#     c = d_f1['c']
#     t = d_f1['t']
#     for x_i in range(len(d_i2['x'])):
#         for s_i in range(len(d_i2['s'])):
#             for i in range(J + 1):
#                 x, s = d_i2['x'][x_i], d_i2['s'][s_i]
#                 state = (i * d_i['sizes_i'][0]
#                          + np.sum(x * sizes_x + s * sizes_s))
#                 for j in range(J):
#                     if (x[j] > 0) or (j == i):
#                         w = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
#                         i_not_admitted = 0
#                         if (i < J) and (i != j) and (x[i] < D):
#                             i_not_admitted = sizes_x_n[i]
#                         for y in range(x[j] + 1):
#                             next_state = (np.sum(
#                                 x * sizes_x_n + s * sizes_s_n)
#                                           - (x[j] - y) * sizes_x_n[j]
#                                           + i_not_admitted
#                                           + sizes_s_n[j])
#                             w += p_xy[j, x[j], y] * V[next_state]
#                         if w > W[state]:
#                             W[state] = w
#     return W
#
# V = np.zeros(env.dim, dtype=np.float32)
# for i in range(100):
#     V = V.reshape(env.size)
#     w = w.reshape(env.size_i)
#     Pi = Pi.reshape(env.size_i)
#     w = get_w(V, w, env.J, env.D, env.gamma,
#               env.d_i1, env.d_i2, env.d_f1, env.p_xy)
#     V = V.reshape(env.dim)
#     w = w.reshape(env.dim_i)
#     V_t = pi_learner.get_v(env, V, w)
#     conv, _, g = pi_learner.convergence(env, V_t, V, 0, '')
#     print(conv, g)
#     V = V_t - V_t[tuple([0] * (env.J * 2))]  # Rescale V_t
#
# # When stuck? How to measure?
#
# def get_w(V, W, Pi, J, D, gamma,
#           d_i1, d_i2, d_f1, p_xy):
#     """W given policy."""
#     sizes_x = d_i1['sizes_i'][1:J + 1]
#     sizes_s = d_i1['sizes_i'][J + 1:J * 2 + 1]
#     sizes_x_n = d_i1['sizes'][0:J]  # sizes Next state
#     sizes_s_n = d_i1['sizes'][J:J * 2]
#     r = d_f1['r']
#     c = d_f1['c']
#     t = d_f1['t']
#     for x_i in range(len(d_i2['x'])):
#         for s_i in range(len(d_i2['s'])):
#             for i in range(J + 1):
#                 x = d_i2['x'][x_i]
#                 s = d_i2['s'][s_i]
#                 state = (i * d_i1['sizes_i'][0] +
#                          + np.sum(x * sizes_x + s * sizes_s))
#                 if Pi[state] > 0:
#                     j = Pi[state] - 1
#                     W[state] = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
#                     i_not_admitted = 0
#                     if (i < J) and (i != j) and (x[i] < D):
#                         i_not_admitted = sizes_x_n[i]
#                     for y in range(x[j] + 1):
#                         next_state = (np.sum(x * sizes_x_n + s * sizes_s_n)
#                                       - (x[j] - y) * sizes_x_n[j]
#                                       + i_not_admitted
#                                       + sizes_s_n[j])
#                         W[state] += p_xy[j, x[j], y] * V[next_state]
#     return W
#
# V = np.zeros(env.dim, dtype=np.float32)
# for i in range(100):
#     V = V.reshape(env.size)
#     w = w.reshape(env.size_i)
#     Pi = Pi.reshape(env.size_i)
#     w = get_w(V, w, Pi, env.J, env.D, env.gamma,
#               env.d_i1, env.d_i2, env.d_f1, env.p_xy)
#     V = V.reshape(env.dim)
#     w = w.reshape(env.dim_i)
#     V_t = pi_learner.get_v(env, V, w)
#     conv, _, g = pi_learner.convergence(env, V_t, V, 0, '')
#     print(conv, g)
#     V = V_t - V_t[tuple([0] * (env.J * 2))]  # Rescale V_t
