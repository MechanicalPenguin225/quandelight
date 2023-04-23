import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.constants import e, hbar, c, pi, eV, epsilon_0
from ..utils import pprint, dtstring
from ..conversions import *

sim_param_names = []
default_sim_params = {}

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


def sim_quantum(d, omega_adim, epsilon, V, Q, atom_tau, delta_o=0, delta_c=0, gamma_star=0, **sim_parameters):
    """Simulates quantum evolution of the resonator-atom system, plots and gives figures of merit.
    ---------- INPUTS ----------
    - d : IN SI UNITS (C*m) : dipole moment of the transition we consider to emit the light.
    - omega_adim : reduced frequency given by Meep (= in units of c/a)
    - epsilon_r : relative epsilon of the cavity. May need to be an averaged value for our DBR cavity.
    - V_meep : mode volume as given by meep (ie in units of a³).
    - Q : quality factor of the resonator
    - atom_tau : in s, the relaxation time of the bare atom
    - delta_o : detuning of the atom wrt the excitation pulse.
    - delta_c : same, but for the cavity.
    - gamma_star : dephasing rate for the atom.
    - sim_parameters : simulation parameters to pass to QuTiP.

    ---------- RETURNS ----------


    """
    return mu, g2, 
