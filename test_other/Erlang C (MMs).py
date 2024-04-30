"""
Edited Erlang C calculator (aka MMs)
https://github.com/AntonioGallego/pyErlang/blob/master/pyErlang.py
"""

from math import factorial, exp

def erlangC(s, a):
    """
    Returns the probability a call waits, P(W_q>0) = C(s, a)

    Parameters:
        s   (int): servers
        a (float): traffic intensity, lambda/mu
    """
    rho =
    p = u / m  ##  agent occupancy
    suma = 0
    for k in range(0, m):
        suma += PowerFact(u, k)
    erlang = b ** e / ((PowerFact(u, m)) + (1 - p) * suma)
    return erlang


def SLA(m, u, T, target):
    """
    Returns the average speed of answer

    Parameters:
        m        (int): agent count
        u      (float): traffic intensity
        T      (float): average call time
        target (float): target answer time
    """
    return (1 - erlangC(m, u) * exp(-(m - u) * (target / T)))


def ASA(m, u, T):
    """
    Returns the average speed of answer (ASA)

    Parameters:
        m   (int): agent count
        u (float): traffic intensity
        T (float): average call time
    """
    return erlangC(m, u) * (T / (m - u))


def agentsNeeded(u, T, targetSLA, target):
    """
    Returns the number of agents needed to reach given SLA

    Parameters:
        u         (float): traffic intensity
        T         (float): average call time
        target    (float): target answer time
        targetSLA (float): % representing calls answered under target time
    """
    level = 0
    m = 1
    while level < targetSLA:
        level = SLA(m, u, T, target)
        m += 1
    return m - 1


def showStats(calls, interval, T, m, target, level):
    """
    Prints Erlang related statistics

    Parameters:
        calls    (int): calls received in a given time interval
        interval (int): time interval in secs (i.e. 1800s == 30m)
        T        (int): average call time, in secs
        m        (int): number of agents

    Intermediate results:
        landa       calls/interval
        u=landa*T   traffic intensity
        p=u/m       agent occupancy
    """
    landa = calls / interval
    u = landa * T  # traffic intensity
    p = u / m  # agent occupancy
    print(
        'calls: {}   interval: {}   landa: {:.8f} (l = calls/interval)'.format(
            calls, interval, landa))
    print(
        'traffic intensity: {:.2f}   agents: {}    agent occupancy: {:.2f}'.format(
            u, m, p))
    print(
        'ErlangC, Probability of waiting: {:.2f}%'.format(erlangC(m, u) * 100))
    print('ASA, Average speed of answer: {:.1f} secs'.format(ASA(m, u, T)))
    print('Probability call is answered in less than {} secs: {:.2f}%'.format(
        target, SLA(m, u, T, target) * 100))
    print(
        'Agents needed to reach {:.2f}% calls answered in <{} secs: {}'.format(
            level * 100, target, agentsNeeded(u, T, level, target)))


def main():
    """
    Runs Erlang tests

    Parameters:
        s          (int): servers
        mu       (float): service rate
        load     (float): traffic intensity (rho)
        t        (float): target time
    """

    instances = [
        # s,  mu, load, t
        [5, 1/30, 0.85, 60],  # inst 9, unit: minutes
        [5, 1/60, 0.85, 60]   # inst 10, unit: minutes
    ]

    for s,  mu, load, t in instances:
        showStats(calls, interval, T, m, target, level)
        print("-" * 10)


if __name__ == "__main__":
    main()