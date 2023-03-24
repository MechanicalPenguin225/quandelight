### OBJECTIVE : find the confined levels within a DBR.

import meep as mp
from meep.mpb import ModeSolver
from meep import mpb
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, hbar
import h5py

eps1 = 3
eps2 = 3.45
lamda = 5

FILL_CENTER = True

N = 5

cavity_transverse_extent = 1

## MPB VARIABLES

X_PADDING_WIDTH = 25 # in lambdas
Y_PADDING_WIDTH = 0

RESOLUTION = 16
FOLDING_FACTOR = 2*(N + 2*X_PADDING_WIDTH + 1)
NUM_BANDS = FOLDING_FACTOR + 1
INTERP_POINTS = 1

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

cell = mp.Vector3(2*sim_half_width + 2*X_PADDING_WIDTH*lamda,
                  cavity_transverse_extent + 2*Y_PADDING_WIDTH*lamda)
a_x = cell.x
a_y = cell.y

b_x = 1/a_x
b_y = 1/a_y

geometry_lattice = mp.Lattice(size=cell)

# Making k point lists
k_points_x = [mp.Vector3(),
            mp.Vector3(0.5, 0),]          # Gamma

k_points_x = mp.interpolate(INTERP_POINTS, k_points_x)

# Modesolver for x direction
ms_x = ModeSolver(num_bands=NUM_BANDS,
                k_points = k_points_x,
                geometry=geometry,
                geometry_lattice=geometry_lattice,
                resolution=RESOLUTION)

ms_x.run_te()
omegas_x = ms_x.all_freqs.T
k_vals_x = np.array([v.dot(mp.Vector3(1, 0, 0)) for v in k_points_x])*b_x


# ---------- PLOTTING ----------

fig, ax = plt.subplots(1, 2, constrained_layout = True)
ax_x, ax_eps = ax

ax_x.set_xlabel(r"$k_x$")
ax_x.set_ylabel(r'$\omega$')

if PLOT_GUESSES :
    ax_x.axhspan(ymin = omega_adim*(1 - band_width / 2), ymax = omega_adim*(1 + band_width / 2), color = 'lightgray', alpha = 0.5)
    ax_x.axhline(y = omega_adim, color = 'lightgray', ls = '--')

# Plotting for x

for band_x, omega_points_x in enumerate(omegas_x):
    omega_plot = ax_x.plot(k_vals_x, omega_points_x, label = str(band_x))


# plotting eps

im = ax_eps.imshow(ms_x.get_epsilon(), aspect = 'auto', origin = 'lower', interpolation = 'none', cmap = 'binary')
fig.colorbar(im)

plt.show()


