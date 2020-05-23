import matplotlib.pyplot as plt
import numpy as np
from qutip import *

q = Qobj([[1], [0]])
print(q)
print(q.dims)
print(q.shape)
print(q.data)
print(q.full())
print(q.isherm, q.type)

sy = Qobj([[0, -1j], [1j, 0]])
print(sy)

sz = Qobj([[1, 0], [0, -1]])
print(sz)

H = 1.0 * sz + 0.1 * sy
print("Qubit Hamiltonian = \n")
print(H)

print(sy.dag())
print(H.tr())
print(H.eigenstates())
print(H.eigenenergies())

# Fundamental basis states (Fock states of oscillator modes)
N = 2  # number of states in Hilbert space
n = 1  # the state that will be occupied

print(basis(N, n))
print(fock(N, n))
print(fock(4, 2))
print(coherent(N=10, alpha=1.0))


