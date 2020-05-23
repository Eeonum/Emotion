import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from math import sqrt

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

print(fock_dm(5, 2))
print(coherent_dm(N=8, alpha=1.0))
n = 1
print(thermal_dm(8, n))

print(sigmax())
print(sigmay())
print(sigmaz())

print(destroy(N=8))
print(create(N=8))
print(destroy(8).dag())


def commutator(op1, op2):
    return op1 * op2 - op2 * op1


a = destroy(5)
amir = commutator(a, a.dag())
print(amir)

x = (a + a.dag()) / sqrt(2)
p = -1j * (a - a.dag()) / sqrt(2)
anis = commutator(x, p)
print(anis)

pishi = commutator(sigmax(), sigmay()) - 2j * sigmaz()
print(pishi)
print(-1j * sigmax() * sigmay() * sigmaz())
print(qeye(2))
print(sigmax() ** 2 == sigmay() ** 2 == sigmaz() ** 2 == qeye(2))

sz1 = tensor(sigmaz(), qeye(2))
print(sz1)

psi1 = tensor(basis(N, 1), basis(N, 0))  # excited first qubit
psi2 = tensor(basis(N, 0), basis(N, 1))  # excited second qubit

print(sz1 * psi1 == psi1)  # this should not be true, because sz1 should flip the sign of the excited state of psi1
print(sz1 * psi2 == psi2)  # this should be true, because sz1 should leave psi2 unaffected

sz2 = tensor(qeye(2), sigmaz())

print(sz2)

print(tensor(sigmax(), sigmax()))

epsilon = [1.0, 1.0]
g = 0.1

sz1 = tensor(sigmaz(), qeye(2))
sz2 = tensor(qeye(2), sigmaz())

H = epsilon[0] * sz1 + epsilon[1] * sz2 + g * tensor(sigmax(), sigmax())
print(H)

wc = 1.0  # cavity frequency
wa = 1.0  # qubit/atom frenqency
g = 0.1  # coupling strength

# cavity mode operator
a = tensor(destroy(5), qeye(2))

# qubit/atom operators
sz = tensor(qeye(5), sigmaz())  # sigma-z operator
sm = tensor(qeye(5), destroy(2))  # sigma-minus operator

# the Jaynes-Cumming Hamiltonian
H = wc * a.dag() * a - 0.5 * wa * sz + g * (a * sm.dag() + a.dag() * sm)
print(H)

a = tensor(destroy(3), qeye(2))
sp = tensor(qeye(3), create(2))
print(a * sp)

print(tensor(destroy(3), create(2)))

H=sigmax()
psi0=basis(2,0)
tlist=np.linspace(0,10,100)
result = mesolve(H, psi0, tlist, [], [])

print(result)

print(len(result.states))
print(result.states[-1])

print(expect(sigmaz(), result.states[-1]))
print(expect(sigmaz(), result.states))

fig, axes = plt.subplots(1,1)

axes.plot(tlist, expect(sigmaz(), result.states))

axes.set_xlabel(r'$t$', fontsize=20)
axes.set_ylabel(r'$\left<\sigma_z\right>$', fontsize=20);
plt.show()

result = mesolve(H, psi0, tlist, [], [sigmax(), sigmay(), sigmaz()])

fig, axes = plt.subplots(1,1)

axes.plot(tlist, result.expect[2], label=r'$\left<\sigma_z\right>$')
axes.plot(tlist, result.expect[1], label=r'$\left<\sigma_y\right>$')
axes.plot(tlist, result.expect[0], label=r'$\left<\sigma_x\right>$')

axes.set_xlabel(r'$t$', fontsize=20)
axes.legend(loc=2);
plt.show()

w = 1.0               # oscillator frequency
kappa = 0.1           # relaxation rate
a = destroy(10)       # oscillator annihilation operator
rho0 = fock_dm(10, 5) # initial state, fock state with 5 photons
H = w * a.dag() * a   # Hamiltonian

# A list of collapse operators
c_ops = [sqrt(kappa) * a]

tlist = np.linspace(0, 50, 100)

# request that the solver return the expectation value of the photon number state operator a.dag() * a
result = mesolve(H, rho0, tlist, c_ops, [a.dag() * a])

fig, axes = plt.subplots(1,1)
axes.plot(tlist, result.expect[0])
axes.set_xlabel(r'$t$', fontsize=20)
axes.set_ylabel(r"Photon number", fontsize=16);
plt.show()
