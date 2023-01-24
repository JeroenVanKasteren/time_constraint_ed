"""
Sandbox Value Iteration

# Env initialization
J = 2
S = 4
lambda_ = np.array([1, 1])
mu = np.array([0.5, 1.5])
print(sum(lambda_/mu)/S)  # Total system load < 1
print(lambda_/mu)  # Used to estimate s_star

t = np.array([1/5]*J)
c = np.array([2, 3, 1])

gamma = 30
D = 15
P = 1e2
e = 1e-5

@author: Jeroen

Policy , Essential to have the numbers in ascending order
Pi = -5, not evalated
Pi = -1, Servers full
Pi = 0, No one waiting
Pi = i, take queue i into serves, i={1,...,J}
Pi = J+1, Keep idle

Penalty is an incentive to take people into service, it is not needed when
there is no decision (servers full)
"""

# PATH = (r"D:\Programs\Surfdrive\Surfdrive\VU\Promovendus"
#         r"\Time constraints in emergency departments\Code")
PATH = (r"C:\Users\jkn354\surfdrive\VU\Promovendus"
        r"\Time constraints in emergency departments\Code")

import os
os.chdir(PATH+"\Other")
from init import Env
os.chdir(PATH)
from Plotting import plot_Pi, plot_V

import numpy as np
from numpy import array, arange, zeros
from numba import njit

np.set_printoptions(precision=3, threshold=2000, linewidth=150)  # TODO

# np.set_printoptions(precision=3, linewidth=150, suppress=True)
env = Env(J=1, S=1, mu=array([3]), lmbda=array([1]), t=array([4]),
          r=array([2]), c=array([2]), P=0,
          gamma=3, D=30, trace=True, print_modulo=20)
# env = Env(J=1, S=4, mu=array([1/2]), lmbda=array([1]), t=array([1]),
#           r=array([2]), c=array([1]), P=0,
#           gamma=5, D=20, trace=False, print_modulo=20)
# env = Env(J=2, S=2, lmbda=array([0.5,0.5]), mu=array([1,1]), t=array([1.]),
#           r=array([1,1]), c=array([1,1]), P=0,
#           gamma=5, D=5, trace=True)
# env = Env(J=2, S=3, mu=array([1,1]), lmbda=array([1,1]), t=array([3,3]),
#           r=array([1,1]), c=array([1,1]), P=0, e=1e-4,
#           gamma=2, D=20, max_iter=1e5, trace=False)  # g.2.22
Not_Evaluated = env.NOT_EVALUATED
Servers_Full = env.SERVERS_FULL
None_Waiting = env.NONE_WAITING
Keep_Idle = env.KEEP_IDLE

J=env.J; S=env.S; D=env.D; gamma=env.gamma; t=env.t; c=env.c; r=env.r; P=env.P
sizes=env.sizes; size=env.size; dim=env.dim
sizes_w=env.sizes_w; size_w=env.size_w; dim_w=env.dim_w
s_states=env.s_states; x_states=env.x_states
P_xy=env.P_xy

def W_init(env, V, W):
    w_states = np.repeat([slice(1,D+2), slice(None)], env.J)
    W[tuple(w_states)] = V[tuple([slice(None)]*(J*2))]
    for i in arange(env.J):
        w_states = np.repeat([slice(1,D+2), slice(None)], env.J)
        w_states[i] = 0  # x_i = -1
        next_states = [slice(None)]*(J*2)
        next_states[i] = 1  # x_i = 1
        W[tuple(w_states)] = V[tuple(next_states)]
        w_states[i] = D+1  # x_i = D
        W[tuple(w_states)] -= P
    return W

@njit
def W_f(V, W):
    """W."""
    V = V.reshape(size); W = W.reshape(size_w)
    for s in s_states:
        for x in x_states:
            state = np.sum(x*sizes_w[0:J] + s*sizes_w[J:J*2])
            if np.any(x == 0):  # directly admit arrival
                j = np.where(x==0)[0][0]
                next_x = x.copy() - 1
                next_x[j] = 0
                next_s = s.copy()
                next_s[j] += 1
                next_state = np.sum(next_x*sizes[0:J] + next_s*sizes[J:J*2])
                W[state] = array([r[j] + V[next_state], W[state]]).max()
            for i in arange(J):
                if(x[i] > 1):  # Class i waiting
                    w = r[i] - c[i] if x[i]-1 > gamma*t[i] else r[i]
                    next_x = x.copy() - 1
                    if np.any(x == 0):
                        next_x[np.where(x==0)[0][0]] = 1  # arrival not admitted
                    for y in arange(x[i]):  # (x[i]-1)+1
                        next_x[i] = y
                        next_s = s.copy()
                        next_s[i] += 1
                        next_state = np.sum(next_x*sizes[0:J] + \
                                            next_s*sizes[J:J*2])
                        w += P_xy[i, x[i]-1, y] * V[next_state]
                    W[state] = array([w, W[state]]).max()
    return W.reshape(dim_w)

def V_f(env, V, W):
    """V_t."""
    states = [slice(None)]*(env.J*2)
    states_w = np.repeat([slice(1, env.D+2),slice(None)], env.J)
    V_t = env.tau * V
    for i in arange(env.J):
        states_ = states.copy()
        next_states = states_w.copy()
        states_[i] = 0  # x_i = 0
        next_states[i] = 0  # x_i = -1
        V_t[tuple(states_)] += env.lmbda[i] * (W[tuple(next_states)] -
                                               V[tuple(states_)])
        states_ = states.copy()
        next_states = states_w.copy()
        states_[i] = slice(1, env.D)  # 0 < x_i < D
        next_states[i] = slice(3, env.D+2)  # 1 < x_i <= D
        V_t[tuple(states_)] += env.gamma * (W[tuple(next_states)] -
                                            V[tuple(states_)])
        states_ = states.copy()
        next_states = states_w.copy()
        states_[i] = env.D  # x_i = D
        next_states[i] = env.D+1  # x_i = D
        V_t[tuple(states_)] += env.gamma * (W[tuple(next_states)] -
                                            V[tuple(states_)])
        for s_i in arange(1, env.S+1):  # s_i
            states_ = states.copy()
            next_states = states_w.copy()
            states_[env.J+i] = s_i
            next_states[env.J+i] = s_i - 1
            V_t[tuple(states_)] += s_i * env.mu[i] * \
                (W[tuple(next_states)] - V[tuple(states_)])
    return V_t/env.tau

@njit
def policy_improvement(V, W, Pi):
    """Determine best action/policy per state by one-step lookahead."""
    V = V.reshape(size); W = W.reshape(size_w); Pi = Pi.reshape(size_w)
    unstable = False
    for s in s_states:
        for x in x_states:
            state = np.sum(x*sizes_w[0:J] + s*sizes_w[J:J*2])
            pi = Pi[state]
            Pi[state] = Keep_Idle
            w = W[state]
            if np.any(x == 0):  # directly admit arrival
                j = np.where(x==0)[0][0]
                next_x = x.copy() - 1
                next_x[j] = 0
                next_s = s.copy()
                next_s[j] += 1
                next_state = np.sum(next_x*sizes[0:J] + next_s*sizes[J:J*2])
                Pi[state] = j + 1 if r[j] + V[next_state] > w else Pi[state]
                w = array([r[j] + V[next_state], w]).max()
            for i in arange(J):
                if(x[i] > 1):  # Class i waiting
                    value = r[i] - c[i] if x[i]-1 > gamma*t[i] else r[i]
                    next_x = x.copy() - 1
                    if np.any(x == 0):
                        next_x[np.where(x==0)[0][0]] = 1  # arrival not admitted
                    for y in arange(x[i]):  # (x[i]-1)+1
                        next_x[i] = y
                        next_s = s.copy()
                        next_s[i] += 1
                        next_state = np.sum(next_x*sizes[0:J] + \
                                            next_s*sizes[J:J*2])
                        value += P_xy[i, x[i]-1, y] * V[next_state]
                    Pi[state] = i+1 if value > w else Pi[state]
                    w = array([value, w]).max()
            if pi != Pi[state]:
                unstable = True
    return Pi.reshape(dim_w), unstable

# Value Iteration
name = 'Value Iteration'
V = zeros(env.dim)  # V_{t-1}
W = zeros(env.dim_w)
Pi = env.initialize_pi()

V_t_ = zeros(env.dim)  # TODO
W_ = zeros(env.dim_w)

count = 0
env.timer(True, name, env.trace)
converged = False
while not converged:  # Update each state.
    W = W_init(env, V, W)

    W_[1:D+2,:] = V.copy()  # TODO
    W_[0,:] = V[1,:]
    if not np.allclose(W_, W):
        print("W_init incorrect")
        print("Iter: ", count)
        print("W_", W_)
        print("W", W)
        break

    W = W_f(V, W)

    W_[0,0] = max(r[0]+V[0,1],V[1,0])  # TODO
    W_[1,0] = 0
    for x in arange(2,D+2):
        W_[x,0] = r[0] - c[0] if x-1 > gamma*t[0] else r[0]
        for y in arange(x):  # (x-1)+1
            W_[x,0] += P_xy[0, x-1, y] * V[y,1]
        W_[x,0] = max(W_[x,0], V[x-1,0])
    W_[0,1] = V[1,1]
    W_[1:D+2,1] = V[:,1]
    if not np.allclose(W_, W):
        print("W incorrect")
        print("Iter: ", count)
        print("W_", W_)
        print("W", W)
        break

    V_t = V_f(env, V, W)

    V_t_[0,0] = env.lmbda*W[0,0] # TODO
    V_t_[0,1] = env.lmbda[0]*W[0,1]+(env.tau-env.lmbda[0]-env.mu[0])*V[0,1]
    V_t_[1:D,0] = gamma*W[3:D+2,0]+(env.tau-gamma)*V[1:D,0]
    V_t_[D,0] = gamma*W[D+1,0]+(env.tau-gamma)*V[D,0]
    V_t_[1:D,1] = gamma*W[3:D+2,1]+env.mu[0]*W[2:D+1,0]+(env.tau-gamma-env.mu[0])*V[1:D,1]
    V_t_[D,1] = gamma*W[D+1,1]+env.mu[0]*W[D+1,0]+(env.tau-gamma-env.mu[0])*V[D,1]
    if not np.allclose(V_t_, V_t*env.tau):
        # V_t_== V_t*env.tau
        print("V_t incorrect")
        print("Iter: ", count)
        print("V_t_", np.round(V_t_,5))
        print("V_t", V_t*env.tau)
        break

    converged, g = env.convergence(V_t, V, count, name)
    V = V_t - V_t[tuple([0]*(env.J*2))]  # Rescale and Save V_t
    count += 1
env.timer(False, name, env.trace)

# Determine policy via Policy Improvement.
W = W_init(env, V, W)
Pi, _ = policy_improvement(V, W, Pi)

print("V", V)
# print("Pi", Pi)
# print("g", g)

if env.J > 1:
    plot_Pi(env, env, Pi, zero_state=True)
    plot_Pi(env, env, Pi, zero_state=False)
for i in arange(env.J):
    plot_Pi(env, env, Pi, zero_state=True, i=i)
    plot_V(env, env, V, zero_state=True, i=i)
