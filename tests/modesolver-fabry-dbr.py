import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, hbar
from meep.mpb import ModeSolver
from meep import mpb
import h5py


eps1 = 3
eps2 = 3.45
lamda = 5

FILL_CENTER = True

N = 30

cavity_transverse_extent = 0.5

## MPB VARIABLES

RESOLUTION = 32
NUM_BANDS = 2
INTERP_POINTS = 128


### FUNCTIONS

def gen_fabry_geometry(N, eps1, eps2, lamda, fill_center = False):
    s1 = np.sqrt(eps1)
    s2 = np.sqrt(eps2)

    lamda_0 = lamda

    lamda_1 = lamda/s1
    lamda_2 = lamda/s2

    lay_th_1 = lamda_1/4
    lay_th_2 = lamda_2/4

    half_cavity_width = lamda_1/2 if fill_center else lamda_0/2

    grating_periodicity = lay_th_1 + lay_th_2

    sim_half_width = half_cavity_width + N*grating_periodicity

    geometry = []

    if fill_center :
        geometry += [mp.Block(mp.Vector3(2*half_cavity_width, cavity_transverse_extent, 0),  # right eps1 layer
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

    return geometry, sim_half_width, half_cavity_width

### PROGRAM FLOW

geometry, sim_half_width, half_cavity_width = gen_fabry_geometry(N, eps1, eps2, lamda, fill_center = FILL_CENTER)


cell = mp.Vector3(2*sim_half_width, cavity_transverse_extent)
a_x = cell.x
a_y = cell.y

b_x = 1/a_x
b_y = 1/a_y

geometry_lattice = mp.Lattice(size=cell)

# Making k point lists
k_points_x = [mp.Vector3(),
            mp.Vector3(0.5, 0),]          # Gamma

k_points_y = [mp.Vector3(),
            mp.Vector3(0, 0.5),]          # Gamma

k_points_x = mp.interpolate(INTERP_POINTS, k_points_x)
k_points_y = mp.interpolate(INTERP_POINTS, k_points_y)

# Modesolver for x direction
ms_x = ModeSolver(num_bands=NUM_BANDS,
                k_points = k_points_x,
                geometry=geometry,
                geometry_lattice=geometry_lattice,
                resolution=RESOLUTION)

ms_x.run_te()
omegas_x = ms_x.all_freqs.T
k_vals_x = np.array([v.norm() for v in k_points_x])*b_x

# Modesolver for y direction
ms_y = ModeSolver(num_bands=NUM_BANDS,
                k_points = k_points_y,
                geometry=geometry,
                geometry_lattice=geometry_lattice,
                resolution=RESOLUTION)

ms_y.run_te()
omegas_y = ms_y.all_freqs.T
k_vals_y = np.array([v.norm() for v in k_points_y])*b_y


# ---------- PLOTTING ----------

fig, ax = plt.subplots(1, 5, constrained_layout = True)
ax_x, ax_x_n, ax_y, ax_y_n, ax_eps = ax

ax_x.set_xlabel(r"$k_x$")
ax_x_n.set_xlabel(r"$k_x$")
ax_y_n.set_xlabel(r"$k_y$")
ax_y.set_xlabel(r"$k_y$")

ax_x_n.set_ylim(0, 3)
ax_y_n.set_ylim(0, 3)

for axis in [ax_x, ax_y]:
    axis.set_ylabel(r'$\omega$')

for axis in [ax_x_n, ax_y_n]:
    axis.set_ylabel(r"$n_eff$")
    axis.axhline(y = np.sqrt(eps2), color = 'lightgray', label = r"$n_2$")
    axis.axhline(y = np.sqrt(eps1), color = 'lightgray', ls = '--', label = r"$n_1$")


# Plotting for x

for band_x, omega_points_x in enumerate(omegas_x):
    omega_plot = ax_x.plot(k_vals_x, omega_points_x, label = str(band_x))
    ax_x_n.plot(k_vals_x, omega_points_x/k_vals_x, ls = '--', color = omega_plot[0].get_color())


# Plotting for y

for band_y, omega_points_y in enumerate(omegas_y):
    omega_plot = ax_y.plot(k_vals_y, omega_points_y, label = str(band_y))
    ax_y_n.plot(k_vals_y, omega_points_y/k_vals_y, ls = '--', color = omega_plot[0].get_color())



# plotting eps

im = ax_eps.imshow(ms_x.get_epsilon(), aspect = 'auto', origin = 'lower', interpolation = 'none', cmap = 'binary')
#im2 = ax_y.imshow(efield, aspect = 'auto', origin = 'lower', interpolation = 'none', cmap = "RdBu_r", alpha = 0.5)

fig.colorbar(im)

ax_x.legend()
ax_y.legend()
plt.show()
