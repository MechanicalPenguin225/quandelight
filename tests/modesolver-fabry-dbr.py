import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, hbar
from meep.mpb import ModeSolver


eps1 = 1
eps2 = 3.45

N = 1

pml_thickness = 17.0


### FUNCTIONS

def gen_fabry_geometry(N, eps1, eps2):
    s1 = np.sqrt(eps1)
    s2 = np.sqrt(eps2)

    omega = (s1 + s2)/(4*s1*s2)

    lamda_0 = 2*np.pi/omega

    lamda_1 = 2*np.pi/(s1*omega)
    lamda_2 = 2*np.pi/(s2*omega)

    lay_th_1 = lamda_1/4
    lay_th_2 = lamda_2/4

    half_cavity_width = lamda_0/2

    grating_periodicity = lay_th_1 + lay_th_2

    sim_half_width = half_cavity_width + N*grating_periodicity + pml_thickness

    geometry = []

    for i in range(N):

        bilayer_center = half_cavity_width + i*grating_periodicity/2

        eps2_layer_center = mp.Vector3(0, 0, bilayer_center + 0.5*(lay_th_2 - grating_periodicity))
        eps1_layer_center = mp.Vector3(0, 0, bilayer_center + 0.5*(grating_periodicity - lay_th_1))

        geometry += [mp.Block(mp.Vector3(0, 0, lay_th_2), # right eps2 layer
                              center=eps2_layer_center,
                              material=mp.Medium(epsilon=eps2)),
                    mp.Block(mp.Vector3(0, 0, lay_th_2),  # left eps2 layer
                              center= mp.Vector3()-eps2_layer_center,
                              material=mp.Medium(epsilon=eps2)),
                    mp.Block(mp.Vector3(0, 0, lay_th_1),  # right eps1 layer
                              center=eps1_layer_center,
                              material=mp.Medium(epsilon=eps1)),
                    mp.Block(mp.Vector3(0, 0, lay_th_1),  # left eps1 layer
                              center= mp.Vector3()-eps1_layer_center,
                              material=mp.Medium(epsilon=eps1))]

    return geometry, sim_half_width, omega, half_cavity_width

### PROGRAM FLOW

geometry, sim_half_width, omega, half_cavity_width = gen_fabry_geometry(N, eps1, eps2)

resolution = 21

cell = mp.Vector3(2*sim_half_width)
#vol = mp.Volume(2*sim_half_width)


NUM_BANDS = 2
geometry_lattice = mp.Lattice(size=cell)

k_points = [mp.Vector3(0, 0,0), mp.Vector3(0, 0, 10*1/(2*omega))]
k_points = mp.interpolate(150, k_points)

omega_values = np.linspace(0.5*omega, 10*omega, 151)
k_bands = np.zeros((omega_values.size, NUM_BANDS))

ms = ModeSolver(num_bands=NUM_BANDS,
                k_points = k_points,
                geometry=geometry,
                geometry_lattice=geometry_lattice,
                resolution=resolution,
                dimensions = 1)

for i, current_omega in enumerate(omega_values):
    k_bands[i, :] = ms.find_k(
        mp.NO_PARITY,
        omega,
        1,
        NUM_BANDS,
        mp.Vector3(0, 0, 1),
        1e-3,
        current_omega * eps2,
        current_omega*0.1,
        current_omega*4)


fig, ax = plt.subplots(1, 2)
ax1, ax2 = ax
ax1.set_xlabel(r"$\omega$")
ax2.set_xlabel(r"$\omega$")

ax1.set_ylabel(r"$k$")
ax2.set_xlabel(r"$n$")

for band in range(NUM_BANDS):
    ax1.plot(omega_values, k_bands[:, band], label = f"band {band + 1}")
    ax2.plot(omega_values, k_bands[:, band]/omega_values, label = f"band {band + 1}")

ax1.legend()
ax2.legend()
plt.show()

