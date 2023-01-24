"""
Created on Tue Mar 24 15:49:09 2020.

@author: Jeroen.
"""

import sys
sys.path.insert(1,r"D:\Documents\SURFdrive\VU\Promovendus"
                r"\Time constraints in emergency departments\Code\Other")
# sys.path.insert(1,r"C:\Users\jkn354\SURFdrive\VU\Promovendus"
#                 r"\Time constraints in emergency departments\Code\Other")
from init import env
from scipy.special import factorial as fac

env = env(J=1, S=1, mu=[0.5], lambda_=[1], t=[0],
          gamma=1, D=3, P=1e2, e=1e-5,
          max_iter=10, trace=True, print_modulo=5, time_check=True)

J = 3
lambda_V = np.array([1, 1.5, 0.5])
mu_V = np.array([0.5, 1, 0.5])
s_V = 10
print(sum(lambda_V/mu_V)/s_V)  # Total system load < 1
print(lambda_V/mu_V)  # Used to estimate s_star
s_star = np.array([4, 2, 2])
t_V = np.array([1/5]*J)
gamma = 50
tau_V = np.maximum(lambda_V, gamma) + s_star*mu_V
D = 100
weight_V = np.array([2, 3, 1])


def get_pi_0(_s, rho, **kwargs):
    """Calculate pi(0)."""
    Js = kwargs.get('i', range(J))  # Classes
    lambda_ = lambda_V[Js]  #; gamma = self.gamma
    if 'i' in kwargs:
        _s = np.array([_s]); rho = np.array([rho]); Js = [0]; pi_0 = [0]
    else:
        pi_0 = np.zeros(len(Js))
    for i in Js:  # For every class
        k = np.array(range(_s[i]))
        pi_0[i] += sum((_s[i]*rho[i])**k / fac(k))
    pi_0 += (_s*rho)**_s / fac(_s) * \
        (gamma + rho * lambda_)/gamma * (1 / (1 - rho))
    pi_0 = 1 / pi_0
    pi_0 *= (_s * rho)**_s / fac(_s)
    return pi_0


def get_tail_prob(_s, rho, pi_0, **kwargs):
    """P(W>t)."""
    Js = kwargs.get('i', range(J))
    lambda_ = lambda_V[Js]; mu = mu_V[Js]; t = t_V[Js]  #; gamma = self.gamma
    tail_prob = pi_0 / (1 - rho) * \
        (lambda_ + gamma) / (gamma + lambda_ * pi_0) * \
        (1 - (_s*mu - lambda_) / (_s*mu + gamma))**(t*gamma)
    return tail_prob


def V_to_memory(_s, i, s_star):  # _s Extra
    """
    Calculate V for a single queue for all x = -s, ..., 0, ..., D.

    Only call this function once per class for effeciency.
    Handles s > s_star (leaves V(x) =  0 for x = -s, ..., 0 )
    """
    lambda_ = lambda_V[i]; mu = mu_V[i]; t = t_V[i]; weight = weight_V[i]
    # gamma = self.gamma
    rho = lambda_ / (_s*mu)

    # Extra, calculate once, use pi_0[i], g[i], ...
    pi_0 = get_pi_0(_s, rho, i=i)
    tail_prob = get_tail_prob(_s, rho, pi_0, i=i)
    # Scale to get average reward
    g = tail_prob * (lambda_ + pi_0 * lambda_**2 / gamma)

    V = np.zeros(_s+1+D)  # V(-s) = 0, reference state
    if(_s <= s_star[i]):  # Extra
        # V(x), x<=0, Precalculate elements of double sum
        A = np.delete(np.indices((_s+1, _s+1)), 0, 1)  # Indices Matrix i,j
        with np.errstate(divide='ignore', invalid='ignore'):
            tmp = fac(A[0] - 1) / fac(A[0] - A[1] - 1) * (mu/lambda_)**A[1]
        # Double sum
        for k in range(1, _s+1):
            V[k] = V[k-1] + sum(tmp[k-1, 0:k-1+1])
        V = g/lambda_*V  # Solve with self

    x = np.array(range(1, D+1))
    tmp_frac = (_s*mu + gamma) / (lambda_ + gamma)
    V[x+_s] = V[_s] + g / (gamma*_s*mu*(1 - rho)**2) * \
        (lambda_*x*(rho - 1) + (lambda_+gamma) * tmp_frac**x -
         (lambda_ + gamma))
    k = np.array(range(_s-1+1))
    V[x+_s] += g / (lambda_*gamma*(rho-1)) * \
        (lambda_ - gamma - gamma/rho *
         sum(fac(_s-1) / fac(_s-k-1) * (mu / lambda_)**k)) * \
        (-rho + (lambda_ + gamma) / (_s*mu + gamma) * tmp_frac**x)
    # -1_{x > t*gamma}[...]
    alpha = np.floor(t*gamma+1).astype(int)
    x = np.array(range(alpha, D+1))
    V[x+_s] -= weight/(gamma * (1 - rho)**2) * \
        ((lambda_*(x - t*gamma - 1)*(rho - 1) - (lambda_ + gamma)) +
         (lambda_ + gamma) * tmp_frac**(x-t*gamma-1))
    return V


def trans_prob():
    """
    Precalculate P_xy.

    Jump from x>0 to y. For index convenience p_{0,y} = 0
    """
    P_xy = np.zeros((J, D+1, D+1))
    A = np.indices((D+1, D+1))  # x=A[0], x=A[1]
    for i in range(J):
        P_xy[i, 1:, 0] = (gamma / (lambda_V[i] + gamma))**A[0, 1:D+1, 0]
        P_xy[i, 1:, 1:] = (gamma / (lambda_V[i] + gamma)) ** \
            (A[0, 1:, 1:] - A[1, 1:, 1:]) * lambda_V[i] / (lambda_V[i]+gamma)
    P_xy = np.tril(P_xy, 0)
    return P_xy


P_xy = trans_prob()

i = 0
s = s_star[i]  # Debug
lambda_ = lambda_V[i]
mu = mu_V[i]
rho = lambda_ / (s*mu)
t = t_V[i]
tau = tau_V[i]
weight = weight_V[i]

V = V_to_memory(s_star[i], i, s_star)
x = np.append(np.array(range(-s_star[i], 0)), np.array(range(D+1)))
# np.stack((x, V), axis=1)

pi_0 = get_pi_0(s, rho, i=i)
tail_prob = get_tail_prob(s, rho, pi_0, i=i)
# Scale to get average reward
g = tail_prob * (lambda_ + pi_0 * lambda_**2 / gamma)

Equality = np.zeros([s+1+D, 3])
Equality[:, 0] = x
LHS = g + tau*V[x+s]
RHS = np.zeros(s+1+D)
# x == -s
RHS[0] = lambda_*V[1] + (tau-lambda_)*V[0]
# -s < x <= 0
y = x[np.logical_and(x > -s, x <= 0)]
RHS[y+s] = lambda_*V[y+s+1] + (y+s)*mu*V[y+s-1] + \
    (tau-lambda_-(y+s)*mu)*V[y+s]
# x>=1
RHS[s+1:s+1+D-1] = gamma*V[s+2:s+1+D] + \
    s*mu*np.sum(P_xy[i, 1:D+1-1, :1+D]*V[s:s+1+D], 1) + \
    (tau-gamma-s*mu)*V[s+1:s+1+D-1]
alpha = np.floor(t*gamma+1).astype(int)
x_alpha = np.array(range(s+alpha, s+1+D-1))
RHS[x_alpha] += weight*s*mu*np.ones(D+1-1-alpha)

Equality[:, 1] = LHS
Equality[:, 2] = RHS

dec = 7
print((np.around(Equality[:s+D-1, 1], dec) ==
       np.around(Equality[:s+D-1, 2], dec)).all())  # TODO

# x=20
# tmp_sum = 0
# for y in range(0,x+1):
#     if(y==0):
#         p_xy = (gamma / (lambda_ + gamma))**x
#     else:
#         p_xy = (gamma / (lambda_ + gamma))**(x-y) * \
#             lambda_ / (lambda_ + gamma)
#     tmp_sum += p_xy
#     print(p_xy)
# print(tmp_sum)

# 3.8189796030208895e-05
# 2.288400626172035e-05
# 2.2884023740430977e-05
# True
# V(-3) 0.0029806440921911347
# V(-2) 0.009631313491753888
# V(0) 0.15609478643644725
# V(1) 0.7553072270736971
# V(2) 0.8140540024637064
# V(20) 1.8250322080123087

###############################################################################
###############################################################################
