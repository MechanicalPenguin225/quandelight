import meep as mp
from meep.mpb import ModeSolver
from meep import mpb
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, hbar
import h5py
from quandelight.geometries import dbr_rectangular

eps1 = 3
eps2 = 3.45
lamda = 5

FILL_CENTER = True

N = 10

cavity_transverse_extent = lamda

## MPB VARIABLES

X_PADDING_WIDTH = np.linspace(0, 10, 3) # in lambdas
Y_PADDING_WIDTH = 0

RESOLUTION = 16
#FOLDING_FACTOR = 4*(N + 2*X_PADDING_WIDTH + 1)
NUM_BANDS = 4
INTERP_POINTS = 0

PLOT_GUESSES = True

geometry, dbr_dims, omega_adim, band_width = dbr_rectangular(N, eps1, eps2, lamda, cavity_transverse_extent, fill_center = FILL_CENTER, thickness = 0)
### PROGRAM FLOW

omegas_low_k = np.zeros((X_PADDING_WIDTH.size, NUM_BANDS)).T
omegas_high_k = np.zeros_like(omegas_low_k)

for i, x_pad in enumerate(X_PADDING_WIDTH):

    cell = dbr_dims + mp.Vector3(2*x_pad*lamda,2*Y_PADDING_WIDTH*lamda)
    a_x = cell.x
    a_y = cell.y

    b_x = 1/a_x
    b_y = 1/a_y

    geometry_lattice = mp.Lattice(size=cell)

    # Making k point lists
    k_points_x = [mp.Vector3(),
                  mp.Vector3(0.5, 0)]          # Gamma

    k_points_x = mp.interpolate(INTERP_POINTS, k_points_x)

    # Modesolver for x direction
    ms_x = ModeSolver(num_bands=NUM_BANDS,
                      k_points = k_points_x,
                      geometry=geometry,
                      geometry_lattice=geometry_lattice,
                      resolution=RESOLUTION,
                      target_freq = omega_adim,
                      tolerance = 1e-5)

    ms_x.run_te()
    omegas_low_k[:, i] = ms_x.all_freqs[0 ,:]
    omegas_high_k[:, i] = ms_x.all_freqs[1, :]

# ---------- PLOTTING ----------

fig, ax = plt.subplots(3, 1, constrained_layout = True, sharex = True)
ax_l, ax_h, ax_e = ax

for axis in ax[:1]:
    axis.set_xlabel(r"$N$")
    if PLOT_GUESSES :
        axis.axhspan(ymin = omega_adim*(1 - band_width / 2), ymax = omega_adim*(1 + band_width / 2), color = 'lightgray', alpha = 0.5)
        axis.axhline(y = omega_adim, color = 'lightgray', ls = '--')


for i, band_l in enumerate(omegas_low_k):
    band_h = omegas_high_k[i, :]

    ax_l.plot(X_PADDING_WIDTH, band_l)
    ax_h.plot(X_PADDING_WIDTH, band_h)



plt.show()

