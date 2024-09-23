
pi_0 = env.get_pi_0(env.s_star, env.rho)
tail_prob = env.get_tail_prob(env.S, env.rho, pi_0, env.gamma*env.t)
g = (env.r - env.c*tail_prob) * (env.lab + pi_0*env.lab**2/env.gamma)
P_xy = env.p_xy[0]  # [0]
V = V_f(env, g)

x = np.arange(-S, D+1)
LHS = g + env.tau * V[x+S]
RHS = np.zeros(S + D + 1)
x = np.arange(-S, 0)  # -s <= x < 0
RHS[x+S] = (lab*(env.r+V[x+S+1]) + (x + S)*mu*V[np.maximum(x+S-1, 0)]
            + (env.tau - lab - (x+S)*mu)*V[x+S])
x = 0  # x = 0
RHS[x+S] = (lab*V[x+S+1] + (x + S)*mu*V[np.maximum(x+S-1, 0)]
            + (env.tau - lab - (x + S) * mu) * V[x+S])
x = np.arange(1, D)  # x>=1
RHS[x+S] = (gamma*V[x+S+1]
            + S*mu*(env.r + np.sum(P_xy[1:D, :D]*V[S:D+S], 1))
            + (env.tau - gamma - S*mu)*V[x+S])
x = np.arange(env.t*gamma+1, D).astype(int)  # x>t*gamma
RHS[x+S] -= S*mu * env.c

print("V", V)
dec = 5
x = np.arange(-env.S, env.D)
print("gamma*t: ", env.gamma*env.t)
print("x, LHS, RHS, V: \n", np.c_[np.arange(-env.S, env.D+1), LHS, RHS, V])
print("LHS==RHS? ", (np.around(LHS[x+S], dec) ==
                     np.around(RHS[x+S], dec)).all())
print("pi_0: ", pi_0)
print("tail_prob: ", tail_prob)
print("g: ", g)
# print("g*tau: ", g*env.tau)

# print(env.P_xy)
# V_single_queue = V
# V_single_queue[4:] - V[:,4]
