import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, hbar
from meep.mpb import ModeSolver
from meep import mpb
import h5py

NUM_INTERP_EPS = 101

T = np.linspace(0, 1, NUM_INTERP_EPS)

eps1_i = 1
eps2_i = 5.45

eps1_f = 5.45
eps2_f = 1

lamda = 15

FILL_CENTER = True

N = 64

cavity_transverse_extent = 1/8

## MPB VARIABLES

PADDING_RATIO = 1

RESOLUTION = 16
NUM_BANDS = 2*(N +1) + 3

PLOT_GUESSES = True

# COMPUTING SOME RELEVANT QUANTITIES

eps_values = np.zeros((NUM_INTERP_EPS, 2))

eps_values[:, 0] = np.linspace(eps1_i, eps1_f, NUM_INTERP_EPS)
eps_values[:, 1] = np.linspace(eps2_i, eps2_f, NUM_INTERP_EPS)

levels_high_k = np.zeros((4, NUM_INTERP_EPS))
levels_low_k = np.zeros_like(levels_high_k)

omegas_eps = np.zeros((3, NUM_INTERP_EPS))

for index, eps_pair in enumerate(eps_values):
    eps1, eps2 = eps_pair
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

    omegas_eps[:, index] = [omega_adim*(1 - band_width/2), omega_adim, omega_adim*(1 + band_width/2)]

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

        # Making k point lists
        k_points_x = [mp.Vector3(), mp.Vector3(0.5, 0)]

        # Modesolver for x direction
        ms_x = ModeSolver(num_bands=NUM_BANDS,
                          k_points = k_points_x,
                          geometry=geometry,
                          geometry_lattice=geometry_lattice,
                          resolution=RESOLUTION)

        ms_x.run_te()
        omegas_x = ms_x.all_freqs

        N_min  = NUM_BANDS - 5
        N_max = NUM_BANDS - 1

        levels_high_k[:, index] = omegas_x[1, N_min:N_max]
        levels_low_k[:, index] = omegas_x[0, N_min:N_max]


# ---------- PLOTTING ----------

fig, ax = plt.subplots(3, 1, figsize = (15, 10), dpi=100, constrained_layout = True, sharex = True)
ax_eps, ax_levels_h, ax_levels_l = ax

for axis in ax :
    axis.grid(True)
    axis.axvline(x = 1/2.225)

ax_levels_l.set_xlabel(r"$t$")
ax_levels_l.set_ylabel(r'$\omega$ (low $k$)')
ax_levels_h.set_ylabel(r'$\omega$ (high $k$)')
ax_eps.set_ylabel(r"$\varepsilon$")

# plotting omega bands

for axis in ax[1:]:
    axis.fill_between(T, omegas_eps[0, :], omegas_eps[2, :], color = 'lightgray', alpha = 0.5, label = "Predicted band gap")
    axis.plot(T, omegas_eps[1, :], ls = '--', color = 'lightgray', label = "Predicted band gap center")

# plotting levels

for level_h in levels_high_k:
    ax_levels_h.plot(T, level_h)

for level_l in levels_low_k:
    ax_levels_l.plot(T, level_l)
# plotting epsilons

ax_eps.plot(T, eps_values[:, 0], label = r"$\varepsilon_1$")
ax_eps.plot(T, eps_values[:, 1], label = r"$\varepsilon_2$")

ax_eps.legend()
ax_levels_h.legend()
ax_levels_l.legend()

fig.suptitle(r"Dependance of the band on $\varepsilon_1$ and $\varepsilon_2$."+ f"\n{N} bilayers per mirror")
fig.savefig("eps_dep_v2.png")
plt.show()
