#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Approximation code for the paper:

S. Lorenzini, A. Cinfrignini, D. Petturiti, and B. Vantaggi.
Quantile-constrained Choquet-Wasserstein p-box 
approximation of arbitrary belief functions.
In Proceedings of FUZZ-IEEE 2025.
"""

import numpy as np
from projection_KL import projection_KL

# Finite metric space
X = np.array([5, 10, 15]) / 10
print('X:', X)

# Focal elements of the belief function mu (subsets of {0, 1, ..., n-1})
F_mu = np.array([{0}, {1}, {2}, {0, 1, 2}])
M = len(F_mu)

# Marginal Mobius inverse of mu
w = np.array([3, 1, 2, 4])
m_mu = w / w.sum()
print('m_mu:', m_mu, 'with sum = ', sum(m_mu))


# Maximal set of focal elements of the belief function nu (subsets of {0, 1, ..., n-1})
# The focal elements must be intervals with separately non-decreasing endpoints
F_nu = np.array([{0}, {0, 1}, {0, 1, 2}, {1, 2}, {2}])
N = len(F_nu)

# Lower-upper quantile constraints
J1 = [0, 1]
J2 = [0, 1, 2, 3]
delta1 = 0.8
delta2 = 0.8


# Order of the Choquet-Wasserstein pseudo-distance
p = 1
print('Order of the Choquet-Wasserstein pseudo-distance p:', p)


# Compute the pessimistic and optimistc cost matrices
def c(x, y, p):
    return (np.abs(x - y))**p


c_min = np.zeros((M, N))

i = -1
for E in F_mu:
    i += 1
    j = -1
    for F in F_nu:
        j += 1
        min_EF = np.Infinity
        for h in E:
            for k in F:
                if c(X[h], X[k], p) <= min_EF:
                    min_EF = c(X[h], X[k], p)
        c_min[i, j] = min_EF


# Cost matrix under the 
c = c_min


def prox_G1(theta):
    m_gamma = np.zeros_like(theta)
    (m, n) = theta.shape
    for i in range(m):
        for j in range(n):
            m_gamma[i, j] = m_mu[i] * theta[i, j] / sum(theta[i, :])
    return m_gamma

def prox_G2(theta):
    m_gamma = np.zeros_like(theta)
    (m, n) = theta.shape
    old_m_nu = np.zeros(n)
    for j in range(n):
        old_m_nu[j] =  sum(theta[:, j])
    (m_nu, opt_val) = projection_KL(old_m_nu, J1, delta1, J2, delta2)
    for i in range(m):
        for j in range(n):
            m_gamma[i, j] = m_nu[j] * theta[i, j] / sum(theta[:, j])
    return m_gamma

# Entropic regularizaqtion parameter (positive but the closer to 0 the better)
lamb = 0.01

m_gamma0 = np.exp(- c / lamb)

z1 = np.ones(c.shape)
z2 = np.ones(c.shape)

print ('lambda:', round(lamb, 3))
m_gamma_old = m_gamma0
for n in range(1000):
    print()
    print('*** EPOCH:', n, '***')
    m_gamma = prox_G1(m_gamma_old * z1)
    z1 = z1 * (m_gamma_old / m_gamma)
    m_gamma_old = m_gamma
    m_gamma = prox_G2(m_gamma_old * z2)
    z2 = z2 * (m_gamma_old / m_gamma)
    if (np.sum(np.abs(m_gamma - m_gamma_old)) < 0.001):
        break
    m_gamma_old = m_gamma
    print('*** CURRENT MARGINALS ***')
    print('m_mu:', np.round(np.sum(m_gamma, axis=1), 3), 'with sum = ', round(np.sum(m_gamma), 3))
    print('m_nu:', np.round(np.sum(m_gamma, axis=0), 3), 'with sum = ', round(np.sum(m_gamma), 3))
    
    

m_mu = np.sum(m_gamma, axis=1)
m_nu = np.sum(m_gamma, axis=0)
print('m_gamma:\n', np.round(m_gamma, 3), 'with sum =', round(sum(sum(m_gamma)), 3), '\n')
print('m_mu:', np.round(m_mu, 3), 'with sum = ', round(np.sum(m_gamma), 3))
print('m_nu:', np.round(m_nu, 3), 'with sum = ', round(np.sum(m_gamma), 3))
print('F_l(quantile_l):', np.round(sum(m_nu[J1]), 3))
print('F_u(quantile_u):', np.round(sum(m_nu[J2]), 3))
(m, n) = c.shape
d = 0
for i in range(m):
    for j in range(n):
        d += c[i, j] * m_gamma[i ,j]
        
# p-order Choquet-Wasserstein pseudo-distance
d = d ** (1/ p)
print()
print('d = ', round(d ** (1/ p), 3), '\n')
    
    
    
    