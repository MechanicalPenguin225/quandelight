import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.constants import e, hbar, c, pi, eV, epsilon_0

#### ---------- CAVITY DECAY RATE : KAPPA
def cavity_decay(omega, Q):
    """Compute cavity decay rate kappa from frequency and quality factor.
    ---------- Inputs ----------
    - omega : reduced frequency given by Meep sim, in units of c/a where a = 1 µm. So unit of freq is ~3e10 Hz
    - Q : quality factor given by Meep sim."""

    return omega/(2*Q)

def kappa_to_ev(kappa, a = 1e-6):
    """Converts kappa to SI and calculates hbar*kappa in eV to check that the figure makes sense.
    ---------- INPUTS ----------
    - kappa : cavity decay rate in Meep units (ie (c/a))
    - a : unit of diatnce in MEEP (µm by convention but might as well include this a parameter.)

    ---------- RETURNS ----------
    - hbar*k : hbar*kappa in eV.
    """
    kappa_SI = kappa*c/a
    return hbar*kappa_SI/eV

#### ---------- ATOM-CAVITY COUPLING : G

def coupling_constant(d, omega_adim, epsilon_r, V_meep, a=1e-6):
    """Computes cavity coupling constant from measured figures of merit.
    ---------- INPUTS ----------
    - d : IN SI UNITS (C*m) : dipole moment of the transition we consider to emit the light.
    - omega_adim : reduced frequency given by Meep (= in units of c/a)
    - epsilon_r : relative epsilon of the cavity. May need to be an averaged value for our DBR cavity.
    - V_meep : mode volume as given by meep (ie in units of a³).
    - a : Meep unit of length. Conventionally 1µm, not need to change it unless you chooe the unit to be different.
    ---------- RETURNS ----------
    - g : cavity coupling (unitless)."""
    return d*np.sqrt(c*omega_adim/(a**4 *hbar*2*epsilon_r*epsilon_0*V_meep))

def g_to_ev(g):
    """Converts g to energy units in eV to check that the figure makes sense.
    ---------- INPUTS ----------
    - g : cavity coupling (unitless)

    ---------- RETURNS ----------
    - hbar*g : hbar*g in eV.
    """
    return hbar*g/eV

#### ---------- ATOM DECAY RATE : GAMMA

def gamma(t_decay, a = 1e-6):
    """Computes gamma in relevant simulation units (units of c/a) from SI decay time of the atom.
    ---------- INPUTS ----------
    - t_decay : SI decay time of the atom.
    - a : Meep unit of length if you were to change it. Default : conventional 1µm

    ---------- RETURNS ----------
    - gamma : atom decay rate in relevant units (c/a)
    """
    return a/(c*t_decay)

#### ---------- PURCELL FACTOR Fp
def purcell_from_gammas(g, gamma, kappa, a=1e-6):
    """Compute Purcell factor from the expression using gamma, kappa, etc. NOTE : this is only for gamma_star = 0.
    --------- INPUTS ----------
    - g : coupling constant between cavity and atom.
    - gamma : emitter decay rate.
    - kappa : cavity decay rate.
    - a : Meep unit of length if you were to change it. Default : conventional 1µm

    ---------- RETURNS ----------
    - Fp : Purcell factor."""

    return 4*g**2/(kappa*gamma*(c/a)**2)

def purcell_from_cavity_features(lamda, eps,  Q, V):
    """Compute Purcell factor from the expression using ccavity figures : Q, V, lamda, etc. NOTE : this is only for gamma_star = 0.
    --------- INPUTS ----------
    - lamda : wavelength of light, in µm
    - eps : dielectric constant of cavity. (this may need to be an average dielectric constant).
    - Q : cavity qualiity factor.
    - V : cavity mode volume according to Meep (in µm³).

    ---------- RETURNS ----------
    - Fp : Purcell factor"""

    return (3/(4*pi**2))*(lamda/np.sqrt(eps))**3 * (Q/V)
