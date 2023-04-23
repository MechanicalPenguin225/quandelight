import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.constants import e, hbar, c, pi, eV, epsilon_0
from ..utils import pprint, dtstring
from ..conversions import *


sim_keys = ["time_points", "corr_time_points", "tau_points", "pulse_amplitude", "atol", "rtol","end_time", "plot"]

default_sim_kwargs = {"time_points" : 1001,
                      "corr_time_points": 301,
                      "tau_points": 301,
                      "pulse_amplitude" : pi,
                      "atol": 1e-8,
                      "rtol": 1e-6,
                      "end_time":"auto",
                      "plot": True}

def gaussian_pulse(t, args):
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


def sim_quantum(omega_adim, epsilon, V, Q, atom_tau, d=50*3.33e-30, delta_o=0, delta_c=0, gamma_star=0, a=1e-6, **sim_kwargs):
    """Simulates quantum evolution of the resonator-atom system, plots and gives figures of merit.
    ---------- INPUTS ----------
    - omega_adim : reduced frequency given by Meep (= in units of c/a)
    - epsilon_r : relative epsilon of the cavity. May need to be an averaged value for our DBR cavity.
    - V_meep : mode volume as given by meep (ie in units of a³).
    - Q : quality factor of the resonator
    - atom_tau : in s, the relaxation time of the bare atom
    - d : IN SI UNITS (C*m) : dipole moment of the transition we consider to emit the light.
    - delta_o : detuning of the atom wrt the excitation pulse.
    - delta_c : same, but for the cavity.
    - gamma_star : dephasing rate for the atom.
    - a : Meep unit of distance in case you decide to change it.
    - sim_parameters : simulation parameters to pass to QuTiP.

    ---------- RETURNS ----------


    - mu : avg number of emitted photons from simulation.
    - g2 : g2 factor for the emission of the source.
    """

    for key in sim_keys:
        if key not in sim_kwargs.keys():
            sim_kwargs[key] = default_sim_kwargs[key]


    ### PHYSICAL PARAMETERS

    lamda = 1/omega_adim

    g = coupling_constant(d, omega_adim, epsilon, V) # coupling between atom and cavity. Expect hbar*g ~ 5 to 20 µeV
    gamma_atom = gamma(atom_tau) # Emitter decay rate. Given at 1/gamma ~ 1.1 ns
    kappa = cavity_decay(omega_adim, Q) # cavity decay rate. Expected hbar*kappa ~ 200 to 500 µeV. kappa = omega/2Q

    kappa_ev = kappa_to_ev(kappa)*1e6
    g_ev = g_to_ev(g)*1e6

    purcell = purcell_from_cavity_features(lamda, epsilon, Q, V)
    purcell_gammas = purcell_from_gammas(g, gamma_atom, kappa)
    sanity_check_string = f"---------- SANITY CHECKS ----------\nℏk = {kappa_ev:.2f}µeV, exp. 200 to 500 µeV\nℏg = {g_ev:.2f} µeV, exp. 5 to 20\nFp={purcell:.2e}, Fp from gamma and kappa = {purcell_gammas:.2e}, deviation {(purcell_gammas/purcell)*100:.2f} %, exp. Purcell = 5"

    pprint(sanity_check_string, "cyan")

    gamma_atom = gamma_atom*c/a
    kappa = kappa*c/a

    pulse_width = 1/(10*purcell*gamma_atom) # as proposed by Stephen
    pulse_amp = sim_kwargs["pulse_amplitude"]

    omega_args = {"width" : pulse_width, # args for the pulse shaping
                  "delay": 0,
                  "amp" : pulse_amp,}
    ### SIMULATION PARAMETERS

    TIME_POINTS = sim_kwargs["time_points"]
    CORR_TIME_POINTS = sim_kwargs["corr_time_points"]
    CORR_TAU_POINTS = sim_kwargs["tau_points"]

    if sim_kwargs["end_time"] == "auto":
        end_time = 3*(np.max((1/kappa, 1/gamma_atom)))
    else :
        end_time = sim_kwargs["end_time"]

    mesolve_options = Options(atol = sim_kwargs["atol"], rtol = sim_kwargs["rtol"])


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
    H_full = [H, [H_t, gaussian_pulse]] # full Hamiltonian that we'll feed into QuTiP

    c_ops = [np.sqrt(kappa)*a, 2*np.sqrt(gamma_star)*n_sigma, np.sqrt(gamma_atom)*sigma] # Collapse operators

    psi0 = tensor(fock(2, 0),fock(2, 0)) # initial state : |0, 0>. The pulse is what'll excite everything

    ### ---------- RUNNING SIMULATION ----------

    # Regular time-evolution
    result = mesolve(H_full, psi0, times, c_ops, [n_a, n_sigma], args=omega_args, options = mesolve_options)

    n_a_sims = result.expect[0] # list of <a^dag a> over time

    # Getting correlation functions
    g2_integrand = correlation_3op_2t(H_full, psi0, corr_t, corr_tau, c_ops, a.dag(), n_a, a, solver='me', args=omega_args, options=mesolve_options)

    # computing the actual values
    mu = kappa*np.sum(n_a_sims)*dt
    g2 = np.real(2*kappa**2/mu**2 * np.sum(g2_integrand)*dt_corr*dtau)

    info_string = f"g = {g:.2e}, gamma = {gamma_atom:.2e}, kappa = {kappa:.2e},\ndelta_o = {delta_o:.1e}, delta_c = {delta_c:.1e}, gamma_star = {gamma_star:.1e}"

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

    ax_l.set_title(f"mu = {mu:.2e}, Fp/(1 + Fp) = {purcell/(1 + purcell):.2e}, Fp = {purcell:.2e}")
    ax_r.set_title(f"g² = {g2:.2e}")
    fig.suptitle(info_string)

    if sim_kwargs["plot"]:
        plt.show()

    return mu, g2
