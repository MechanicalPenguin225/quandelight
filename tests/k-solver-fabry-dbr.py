import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, hbar
from meep.mpb import ModeSolver
from meep import mpb
import h5py


eps1 = 3
eps2 = 3.45
lamda = 15

FILL_CENTER = True

N = 16

cavity_transverse_extent = 1

## MPB VARIABLES

PADDING_RATIO = 1

RESOLUTION = 16
NUM_BANDS = 4*(N + 1) + 1
INTERP_POINTS = 1
OMEGA_POINTS = 11

PLOT_GUESSES = True

# COMPUTING SOME RELEVANT QUANTITIES

n1 = np.sqrt(eps1)
n2 = np.sqrt(eps2)



lamda_1 = lamda/n1
lamda_2 = lamda/n2

lay_th_1 = lamda_1/4
lay_th_2 = lamda_2/4

lamda_0 = lamda_1 if FILL_CENTER else lamda

half_cavity_width = lamda_0/2

grating_periodicity = lay_th_1 + lay_th_2

omega_adim = (n1 + n2)/(4*n1*n2*grating_periodicity) # careful, this is a reduced unit wrt the periodicity of the bragg reflector
band_width = 4/np.pi*np.arcsin(np.abs(n1 - n2)/(n1 + n2))

sim_half_width = half_cavity_width + N*grating_periodicity

geometry = []

if FILL_CENTER :
    geometry += [mp.Block(mp.Vector3(2*half_cavity_width, cavity_transverse_extent, 0),
                          center=mp.Vector3(),
                          material=mp.Medium(epsilon=eps1))]

for i in range(N):

    bilayer_edge = half_cavity_width + i*grating_periodicity

    eps2_layer_center = mp.Vector3(bilayer_edge + 0.5*lay_th_2, 0, 0)
    eps1_layer_center = mp.Vector3(bilayer_edge + lay_th_2 + 0.5*lay_th_1, 0, 0)

    geometry += [mp.Block(mp.Vector3(lay_th_2, cavity_transverse_extent, 0), # right eps2 layer
                          center=eps2_layer_center,
                          material=mp.Medium(epsilon=eps2)),
                 mp.Block(mp.Vector3(lay_th_2, cavity_transverse_extent, 0),  # left eps2 layer
                          center= mp.Vector3()-eps2_layer_center,
                          material=mp.Medium(epsilon=eps2)),
                 mp.Block(mp.Vector3(lay_th_1, cavity_transverse_extent, 0),  # right eps1 layer
                          center=eps1_layer_center,
                          material=mp.Medium(epsilon=eps1)),
                 mp.Block(mp.Vector3(lay_th_1, cavity_transverse_extent, 0),  # left eps1 layer
                          center= mp.Vector3()-eps1_layer_center,
                          material=mp.Medium(epsilon=eps1))]



### PROGRAM FLOW

cell = mp.Vector3(2*sim_half_width,
                  cavity_transverse_extent)
a_x = cell.x
a_y = cell.y

b_x = 1/a_x
b_y = 1/a_y

geometry_lattice = mp.Lattice(size=PADDING_RATIO*cell)

k_points = [mp.Vector3(),          # Gamma
            mp.Vector3(0.5)]          # Gamma

k_points = mp.interpolate(INTERP_POINTS, k_points)

omegas = np.linspace(omega_adim*(1 - band_width), omega_adim*(1 + band_width), OMEGA_POINTS)


ms = ModeSolver(num_bands=NUM_BANDS,
                k_points = k_points,
                geometry=geometry,
                geometry_lattice=geometry_lattice,
                resolution=RESOLUTION)

k_vals = np.zeros((len(omegas), NUM_BANDS + 1))
for i, omega in enumerate(omegas) :
    try :
        w = ms.find_k(mp.NO_PARITY, omega, 0, NUM_BANDS, mp.Vector3(1, 0), 1e-4, 1e-2, 0, 0.5)
        print(f"---------------------- ANS : {w}--------------------------")
        k_vals[i, :] = np.array(w)
    except :
        k_vals[i, :] = np.nan

fig, ax = plt.subplots()

for band in range(NUM_BANDS + 1):
    ax.plot(omegas, k_vals[:, band], marker = "+", label = f"{band}")
ax.legend()
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\left|k\right|$")
plt.show()
