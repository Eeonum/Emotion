import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

# make qutip available in the rest of the notebook
from qutip import *
from math import pi, sqrt

from IPython.display import Image

w0 = 1.0  * 2 * pi  # cavity frequency
wa = 1.0  * 2 * pi  # atom frequency
g  = 0.05 * 2 * pi  # coupling strength

kappa = 0.04        # cavity dissipation rate
gamma = 0.00        # atom dissipation rate
Gamma = 0.35        # atom pump rate

N = 50              # number of cavity fock states
n_th_a = 0.0        # avg number of thermal bath excitation

tlist = np.linspace(0, 150, 101)

# intial state
psi0 = tensor(basis(N,0), basis(2,0)) # start without excitations

# operators
a  = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2))
sx = tensor(qeye(N), sigmax())

# Hamiltonian
H = w0 * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() + a) * sx
print(H)

# collapse operators
c_ops = []

rate = kappa * (1 + n_th_a)
if rate > 0.0:
    c_ops.append(sqrt(rate) * a)

rate = kappa * n_th_a
if rate > 0.0:
    c_ops.append(sqrt(rate) * a.dag())

rate = gamma
if rate > 0.0:
    c_ops.append(sqrt(rate) * sm)

rate = Gamma
if rate > 0.0:
    c_ops.append(sqrt(rate) * sm.dag())

opt = Odeoptions(nsteps=2000) # allow extra time-steps
output = mesolve(H, psi0, tlist, c_ops, [a.dag() * a, sm.dag() * sm], options=opt)
print(output)

n_c = output.expect[0]
n_a = output.expect[1]

fig, axes = plt.subplots(1, 1, figsize=(8,6))

axes.plot(tlist, n_c, label="Cavity")
axes.plot(tlist, n_a, label="Atom excited state")
axes.set_xlim(0, 150)
axes.legend(loc=0)
axes.set_xlabel('Time')
axes.set_ylabel('Occupation probability');
plt.show()

rho_ss = steadystate(H, c_ops)

fig, axes = plt.subplots(1, 2, figsize=(12,6))

xvec = np.linspace(-5,5,200)

rho_cavity = ptrace(rho_ss, 0)
W = wigner(rho_cavity, xvec, xvec)
wlim = abs(W).max()

axes[1].contourf(xvec, xvec, W, 100, norm=colors.Normalize(-wlim,wlim), cmap=plt.get_cmap('RdBu'))
axes[1].set_xlabel(r'Im $\alpha$', fontsize=18)
axes[1].set_ylabel(r'Re $\alpha$', fontsize=18)

axes[0].bar(np.arange(0, N), np.real(rho_cavity.diag()), color="blue", alpha=0.6)
axes[0].set_ylim(0, 1)
axes[0].set_xlim(0, N)
axes[0].set_xlabel('Fock number', fontsize=18)
axes[0].set_ylabel('Occupation probability', fontsize=18);
plt.show()

tlist = np.linspace(0, 25, 5)
output = mesolve(H, psi0, tlist, c_ops, [], options=Odeoptions(nsteps=5000))

rho_ss_sublist = output.states

xvec = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(2, len(rho_ss_sublist), figsize=(3 * len(rho_ss_sublist), 6))

for idx, rho_ss in enumerate(rho_ss_sublist):
    # trace out the cavity density matrix
    rho_ss_cavity = ptrace(rho_ss, 0)

    # calculate its wigner function
    W = wigner(rho_ss_cavity, xvec, xvec)

    # plot its wigner function
    wlim = abs(W).max()
    axes[0, idx].contourf(xvec, xvec, W, 100, norm=colors.Normalize(-wlim, wlim), cmap=plt.get_cmap('RdBu'))
    axes[0, idx].set_title(r'$t = %.1f$' % tlist[idx])

    # plot its fock-state distribution
    axes[1, idx].bar(np.arange(0, N), np.real(rho_ss_cavity.diag()), color="blue", alpha=0.8)
    axes[1, idx].set_ylim(0, 1)
    axes[1, idx].set_xlim(0, 15)
    plt.show()


def calulcate_avg_photons(N, Gamma):
    # collapse operators
    c_ops = []

    rate = kappa * (1 + n_th_a)
    if rate > 0.0:
        c_ops.append(sqrt(rate) * a)

    rate = kappa * n_th_a
    if rate > 0.0:
        c_ops.append(sqrt(rate) * a.dag())

    rate = gamma
    if rate > 0.0:
        c_ops.append(sqrt(rate) * sm)

    rate = Gamma
    if rate > 0.0:
        c_ops.append(sqrt(rate) * sm.dag())

    # Ground state and steady state for the Hamiltonian: H = H0 + g * H1
    rho_ss = steadystate(H, c_ops)

    # cavity photon number
    n_cavity = expect(a.dag() * a, rho_ss)

    # cavity second order coherence function
    g2_cavity = expect(a.dag() * a.dag() * a * a, rho_ss) / (n_cavity ** 2)

    return n_cavity, g2_cavity

Gamma_max = 2 * (4*g**2) / kappa
Gamma_vec = np.linspace(0.1, Gamma_max, 50)

n_avg_vec = []
g2_vec = []

for Gamma in Gamma_vec:
    n_avg, g2 = calulcate_avg_photons(N, Gamma)
    n_avg_vec.append(n_avg)
    g2_vec.append(g2)

fig, axes = plt.subplots(1, 1, figsize=(12,6))

axes.plot(Gamma_vec * kappa / (4*g**2), n_avg_vec, color="blue", alpha=0.6, label="numerical")

axes.set_xlabel(r'$\Gamma\kappa/(4g^2)$', fontsize=18)
axes.set_ylabel(r'Occupation probability $\langle n \rangle$', fontsize=18)
axes.set_xlim(0, 2);
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(12,6))

axes.plot(Gamma_vec * kappa / (4*g**2), g2_vec, color="blue", alpha=0.6, label="numerical")

axes.set_xlabel(r'$\Gamma\kappa/(4g^2)$', fontsize=18)
axes.set_ylabel(r'$g^{(2)}(0)$', fontsize=18)
axes.set_xlim(0, 2)
axes.text(0.1, 1.1, "Lasing regime", fontsize=16)
axes.text(1.5, 1.8, "Thermal regime", fontsize=16);
plt.show()

Gamma = 0.5 * (4*g**2) / kappa

c_ops = [sqrt(kappa * (1 + n_th_a)) * a, sqrt(kappa * n_th_a) * a.dag(), sqrt(gamma) * sm, sqrt(Gamma) * sm.dag()]

rho_ss = steadystate(H, c_ops)

fig, axes = plt.subplots(1, 2, figsize=(16,6))

xvec = np.linspace(-10,10,200)

rho_cavity = ptrace(rho_ss, 0)
W = wigner(rho_cavity, xvec, xvec)
wlim = abs(W).max()
axes[1].contourf(xvec, xvec, W, 100, norm=colors.Normalize(-wlim,wlim), cmap=plt.get_cmap('RdBu'))
axes[1].set_xlabel(r'Im $\alpha$', fontsize=18)
axes[1].set_ylabel(r'Re $\alpha$', fontsize=18)

axes[0].bar(np.arange(0, N), np.real(rho_cavity.diag()), color="blue", alpha=0.6)
axes[0].set_xlabel(r'$n$', fontsize=18)
axes[0].set_ylabel(r'Occupation probability', fontsize=18)
axes[0].set_ylim(0, 1)
axes[0].set_xlim(0, N);
plt.show()

Gamma = 1.5 * (4*g**2) / kappa

c_ops = [sqrt(kappa * (1 + n_th_a)) * a, sqrt(kappa * n_th_a) * a.dag(), sqrt(gamma) * sm, sqrt(Gamma) * sm.dag()]

rho_ss = steadystate(H, c_ops)

fig, axes = plt.subplots(1, 2, figsize=(16,6))

xvec = np.linspace(-10,10,200)

rho_cavity = ptrace(rho_ss, 0)
W = wigner(rho_cavity, xvec, xvec)
wlim = abs(W).max()
axes[1].contourf(xvec, xvec, W, 100, norm=colors.Normalize(-wlim,wlim), cmap=plt.get_cmap('RdBu'))
axes[1].set_xlabel(r'Im $\alpha$', fontsize=18)
axes[1].set_ylabel(r'Re $\alpha$', fontsize=18)

axes[0].bar(np.arange(0, N), np.real(rho_cavity.diag()), color="blue", alpha=0.6)
axes[0].set_xlabel(r'$n$', fontsize=18)
axes[0].set_ylabel(r'Occupation probability', fontsize=18)
axes[0].set_ylim(0, 1)
axes[0].set_xlim(0, N);
plt.show()

from qutip.ipynbtools import version_table

print(version_table())