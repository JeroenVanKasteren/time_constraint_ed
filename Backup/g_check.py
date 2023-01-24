"""
Created on Tue Mar 24 15:49:09 2020.

@author: Jeroen.
"""

import numpy as np

lambda_ = 1
mu = 0.5
s = 4
t = 1/5
gamma = 50
tau = max(lambda_, gamma)+s*mu
rho = lambda_ / (s * mu)
assert rho < 1, "rho < 1 does not hold"
D = 100


def get_pi_0(_s, i):
    """Calculate pi(0)."""
    # lambda_ = self.lambda_[i]
    # mu = self.mu[i]
    # gamma = self.gamma
    rho = lambda_ / (_s*mu)
    pi_0 = 0
    for k in range(_s-1+1):  # Sum
        pi_0 += (_s*rho)**k / np.math.factorial(k)
    pi_0 += (_s*rho)**_s / np.math.factorial(_s) * \
        (gamma + rho * lambda_)/gamma * (1 / (1 - rho))
    pi_0 = 1 / pi_0
    pi_0 *= (_s * rho)**_s / np.math.factorial(_s)
    return pi_0


def get_tail_prob(_s, i, pi_0):
    """P(W>t)."""
    # lambda_ = self.lambda_[i]
    # mu = self.mu[i]
    # gamma = self.gamma
    rho = lambda_ / (_s*mu)
    tail_prob = pi_0 / (1 - rho) * \
        (lambda_ + gamma) / (gamma + lambda_ * pi_0) * \
        (1 - (_s*mu - lambda_) / (_s*mu + gamma))**(t*gamma)
    return tail_prob


def V(x, _s, i):
    """Calculate V for a single queue."""
    # lambda_ = self.lambda_[i]
    # mu = self.mu[i]
    # gamma = self.gamma
    rho = lambda_ / (_s*mu)
    # V(-s) = 0, reference state
    if(x == -_s):
        return 0
    # V(x) for x<=0 or x>0
    if(x <= 0):  # V(x) for x<0
        _V = 0
        for k in range(1, x+_s+1):
            for l in range(k-1+1):
                _V += np.math.factorial(k - 1) / \
                    np.math.factorial(k - l - 1) * \
                    (mu/lambda_)**l
        _V *= g / lambda_
    else:  # V(x) for x>0
        tmp_frac = (_s*mu + gamma) / (lambda_ + gamma)
        _V = V(0, s, 0)
        _V += g / (gamma*_s*mu*(1 - rho)**2) * \
            (lambda_*x*(rho - 1) + (lambda_+gamma) * tmp_frac**x -
             (lambda_ + gamma))
        tmp_sum = 0  # Calculate sum
        for k in range(_s-1+1):
            tmp_sum += np.math.factorial(_s-1) / np.math.factorial(_s-k-1) * \
                (mu / lambda_)**k
        _V += g / (lambda_*gamma*(rho-1)) * \
            (lambda_ - gamma - gamma/rho * tmp_sum) * \
            (-rho + (lambda_ + gamma) / (_s*mu + gamma) * tmp_frac**x)
        if(x > t*gamma):  # -1_{x > t*gamma}[...]
            _V -= 1/(gamma * (1 - rho)**2) * \
                ((lambda_*(x - t*gamma - 1)*(rho - 1) - (lambda_ + gamma)) +
                 (lambda_ + gamma) * tmp_frac**(x-t*gamma-1))
    return _V


pi_0 = get_pi_0(s, 0)
tail_prob = get_tail_prob(s, 0, pi_0)
g = tail_prob * (lambda_ + pi_0 * lambda_**2 / gamma)

Equality = np.zeros([D-(-3), 3])
for x in range(-3, D):
    Equality[x, 0] = x
    LHS = g+tau*V(x, s, 0)
    if(x <= 0):  # -s <= x <= 0
        RHS = lambda_*V(x+1, s, 0) + (x+s)*mu*V(x-1, s, 0) + \
            (tau-lambda_-(x+s)*mu)*V(x, s, 0)
    else:  # x>=1
        tmp = 0
        for y in range(0, x+1):
            if(y == 0):
                p_xy = (gamma / (lambda_ + gamma))**x
            else:
                p_xy = (gamma / (lambda_ + gamma))**(x-y) * \
                    lambda_ / (lambda_ + gamma)
            tmp += p_xy*V(y, s, 0)
        RHS = gamma*V(x+1, s, 0) + s*mu*tmp + (tau-gamma-s*mu)*V(x, s, 0)
        if(x > t*gamma):  # -1_{x > t*gamma}[...]
            RHS += s*mu*1
    Equality[x, 1] = LHS
    Equality[x, 2] = RHS

print(pi_0)
print(tail_prob)
print(g)
dec = 8
print((np.around(Equality[:, 1], dec) ==
       np.around(Equality[:, 2], dec)).all())
print("V(-3)", V(-3, s, 0))
print("V(-2)", V(-2, s, 0))
print("V(0)", V(0, s, 0))
print("V(1)", V(1, s, 0))
print("V(2)", V(2, s, 0))
print("V(20)", V(20, s, 0))

print(np.around(Equality[0:15], dec))

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


# def get_pi_0():
#     # s = self.s
#     # lambda_ = self.lambda_
#     # mu = self.mu
#     # gamma = self.gamma
#     # rho = self.rho
#     pi_0 = 0
#     for i in range(s-1+1):  # Sum
#         pi_0 += (s*rho)**i / np.math.factorial(i)
#     pi_0 += (s*rho)**s / np.math.factorial(s) * \
#         (gamma + rho * lambda_)/gamma * ( 1 / (1 - rho))
#     pi_0 = 1 / pi_0
#     pi_0 *= (s * rho)**s / np.math.factorial(s)
#     return pi_0

# def get_tail_prob():
#     """ P(W>t) """
#     # s = self.s
#     # lambda_ = self.lambda_
#     # mu = self.mu
#     # gamma = self.gamma
#     # rho = self.rho
#     # pi_0 = self.pi_0
#     tail_prob = pi_0 / (1 - rho) * \
#         (lambda_ + gamma) / (gamma + lambda_ * pi_0) * \
#             (1 - (s*mu - lambda_) / (s*mu + gamma))**(t*gamma)
#     return tail_prob

# def V(x):
#     # s = self.s
#     # lambda_ = self.lambda_
#     # mu = self.mu
#     # gamma = self.gamma
#     # rho = self.rho
#     # V(-s) = 0, reference state
#     if(x == -s):
#         return 0
#     # V(x) for x<=0 or x>0
#     if(x<=0):  # V(x) for x<0
#         _V = 0
#         for i in range(1,x+s+1):
#             for j in range(i-1+1):
#                 _V += np.math.factorial(i - 1) / \
#                 np.math.factorial(i - j - 1) * \
#                         (mu/lambda_)**j
#         _V *= g / lambda_
#     else:  # V(x) for x>0
#         tmp_frac = (s*mu + gamma) / (lambda_ + gamma)
#         _V = V(0)
#         _V += g / (gamma*s*mu*(1 - rho)**2) * \
#             (lambda_*x*(rho - 1) + (lambda_+gamma) * tmp_frac**x - \
#              (lambda_ + gamma))
#         tmp_sum = 0  # Calculate sum
#         for k in range(s-1+1):
#             tmp_sum += np.math.factorial(s-1) / np.math.factorial(s-k-1) * \
#                 (mu / lambda_)**k
#         _V += g / (lambda_*gamma*(rho-1)) * \
#             (lambda_ - gamma - gamma/rho * tmp_sum) * \
#                 (-rho + (lambda_ + gamma) / (s*mu + gamma) * tmp_frac**x)
#         if(x > t*gamma):  # -1_{x > t*gamma}[...]
#             _V -= 1/(gamma * (1 - rho)**2) * \
#             ((lambda_*(x - t*gamma - 1)*(rho - 1) - (lambda_ + gamma)) + \
#              (lambda_ + gamma) * tmp_frac**(x-t*gamma-1))
#     return _V
