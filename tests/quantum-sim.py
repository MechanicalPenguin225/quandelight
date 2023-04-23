from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from quandelight.utils import pprint
from quandelight.conversions import *
from scipy.constants import c, hbar

# Computing the figures of merit from the quantum sim

d = 50*3.33e-30
omega_adim = 1.08
eps = 3
V = 1.59
Q = 10*81.2
tau = 1.1e-9

a = 1e-6 # length unit used in Meep
lamda = 1/omega_adim
print(lamda)

### PHYSICAL PARAMETERS

g = coupling_constant(d, omega_adim, eps, V) # coupling between atom and cavity. Expect hbar*g ~ 5 to 20 µeV
gamma = gamma(tau)*c/a # Emitter decay rate. Given at 1/gamma ~ 1.1 ns
kappa = cavity_decay(omega_adim, Q)*c/a # cavity decay rate. Expected hbar*kappa ~ 200 to 500 µeV. kappa = omega/2Q


delta_o = 0 # detuning for the atom. 0 for now
delta_c = 0 # detuning for the cavity. 0 for now
gamma_star = 0

### SIMULATION PARAMETERS

TIME_POINTS = 1001
CORR_TIME_POINTS = 301
CORR_TAU_POINTS = 301
end_time = 5/(np.sqrt(kappa*gamma))

pprint(f"{end_time:.2e}, {end_time*a/c:.2e}, {c/a:.2e}", "red")

### FUNCTIONS

def Omega(t, args):
    """Time-evolution of the Rabi pulse.
    ---------- INPUTS ----------
    t : float. Time
    args : dict {"width", "delay", "amp"} : shape factors of the pulse. "amp" is optional and defaults to pi."""
    sigma = args["width"]
    mu = args["delay"]
    if "amps" in args.keys() :
        A = args["amp"]
    else :
        A = pi
    return A/(sigma*np.sqrt(2*np.pi)) * np.exp(-((t-mu)/sigma)**2/2)

### PROGRAM FLOW

# some physical parameters from data

purcell = purcell_from_cavity_features(lamda, eps, Q, V)
pulse_width = 1/(10*purcell*gamma) # as proposed by Stephen

omega_args = {"width" : pulse_width, # args for the pulse shaping
              "delay": 0,}

# defining simulation params
times = np.linspace(0, end_time, TIME_POINTS) # times for the time-resolved evolution of the cavity.
corr_t = np.linspace(0, end_time, CORR_TIME_POINTS)
corr_tau = np.linspace(0, end_time, CORR_TAU_POINTS)


dt = times[1] - times[0]
dt_corr = corr_t[1] - corr_t[0]
dtau = corr_tau[1] - corr_tau[0]

# setting up hamiltoninans

sigma = tensor(destroy(2), qeye(2)) # atom annihilation operator
a = tensor(qeye(2), destroy(2)) # cavity annihilation operator

n_a = a.dag()*a # cavity number operator
n_sigma = sigma.dag()*sigma # atom number operator

H = delta_o*n_a + delta_c*n_sigma + g*(a.dag()*sigma + a*sigma.dag()) # Constant part of the Hamiltonian
H_t = 0.5*(sigma + sigma.dag()) # Time-varying part of the Hamiltonian
H_full = [H, [H_t, Omega]] # full Hamiltonian that we'll feed into QuTiP

c_ops = [np.sqrt(kappa)*a, 2*np.sqrt(gamma_star)*n_sigma, np.sqrt(gamma)*sigma] # Collapse operators

psi0 = tensor(fock(2, 0),fock(2, 0)) # initial state : |0, 0>. The pulse is what'll excite everything

### ---------- RUNNING SIMULATION ----------

# Regular time-evolution
result = mesolve(H_full, psi0, times, c_ops, [n_a, n_sigma], args=omega_args)

n_a_sims = result.expect[0] # list of <a^dag a> over time

# Getting correlation functions
g2_integrand = correlation_3op_2t(H_full, psi0, corr_t, corr_tau, c_ops, a.dag(), n_a, a, solver='me', args=omega_args)

# computing the actual values
mu = kappa*np.sum(n_a_sims)*dt
g2 = np.real(2*kappa**2/mu**2 * np.sum(g2_integrand)*dt_corr*dtau)

info_string = f"g = {g:.2e}, gamma = {gamma:.2e}, kappa = {kappa:.2e},\ndelta_o = {delta_o:.1e}, delta_c = {delta_c:.1e}, gamma_star = {gamma_star:.1e}"

pprint(f"mu = {mu:.2e}, purcell expre = {purcell/(1 + purcell):.2e}\n g² = {g2:.2e}", "yellow")
pprint(info_string, "purple")
### ---------- PLOTTING RESULTS ----------

fig, ax = plt.subplots(1, 2, constrained_layout = True)
ax_l, ax_r = ax

# LEFT AXIS : TIME EVOLUTION
ax_l_2 = ax_l.twinx()


ax_l.set_xlabel(r'$t$')

labels = [r"$\left\langle a^{\dagger}a\right\rangle$", r"$\left\langle \sigma^{\dagger}\sigma\right\rangle$"]
axes = [ax_l, ax_l_2]
colors = ["tab:red", "tab:blue"]
for i, out in enumerate(result.expect) :
    axes[i].grid(True, color = colors[i], alpha = 0.25)
    axes[i].plot(times, out, color = colors[i])
    axes[i].set_ylabel(labels[i], color = colors[i])
    axes[i].tick_params(axis='y', labelcolor = colors[i])



# RIGHT AXIS : g² integrand
ax_r.set_xlabel(r"$\tau$")
ax_r.set_ylabel(r"$t$")

g2_int_real = np.real(g2_integrand)

im = ax_r.imshow(g2_int_real, origin = "lower", aspect = "auto", extent = (corr_tau[0], corr_tau[-1], corr_t[0], corr_t[-1]), cmap = "cividis", vmin = 0)
fig.colorbar(im, label = "integrand for $g^{(2)}$")

ax_l.set_title(f"mu = {mu:.2e}, Fp/(1 + Fp) = {purcell/(1 + purcell):.2e}")
ax_r.set_title(f"g² = {g2:.2e}")

fig.suptitle(info_string)

plt.show()


