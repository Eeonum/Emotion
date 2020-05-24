import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from qutip import *
from math import sqrt


def integrate(N, w_ph, f, w_at, a, b, g, psi0, tlist, gamma, solver):
    # construct the hamiltonian

    H = 0
    for j in range(N):
        H += w_ph * f.dag() * f + \
             0.5 * w_at[j] * (b.dag()[j] * b[j] - a.dag()[j] * a) + \
             g[j] / sqrt(N) * (f.dag() * a.dag()[j] * b[j] + b.dag()[j] * a[j] * f)


    c_op_list =[]

    # evolve and calculate expectation values
    if solver=="me":
        result=mesolve(H, psi0, tlist, c_op_list, sz_list)
