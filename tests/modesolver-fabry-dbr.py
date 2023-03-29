import meep as mp
from meep.mpb import ModeSolver
from meep import mpb
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, hbar
import h5py
import math
from quandelight.geometries import dbr_rectangular

eps1 = 1
eps2 = 12
lamda = 5

FILL_CENTER = False

N = 128

cavity_transverse_extent = 0*lamda

## MPB VARIABLES

X_PADDING_WIDTH = 3 +  np.sqrt(2)-1 # in lambdas
Y_PADDING_WIDTH = 0

RESOLUTION = 8
FOLDING_FACTOR = 2*(N + 2*math.floor(X_PADDING_WIDTH) + 1)
NUM_BANDS = FOLDING_FACTOR + 3
INTERP_POINTS = 1

PLOT_GUESSES = True

geometry, dbr_dims, omega_adim, band_width = dbr_rectangular(N, eps1, eps2, lamda, cavity_transverse_extent, fill_center = FILL_CENTER, thickness = 0)
### PROGRAM FLOW



cell = dbr_dims + mp.Vector3(2*X_PADDING_WIDTH*lamda,2*Y_PADDING_WIDTH*lamda)
a_x = cell.x
a_y = cell.y

b_x = 1/a_x
#b_y = 1/a_y

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


# plotting eps / efield

if cavity_transverse_extent != 0:
    im = ax_eps.imshow(ms_x.get_epsilon(), aspect = 'auto', origin = 'lower', interpolation = 'none', cmap = 'binary')
    fig.colorbar(im)
else :
    N_min  = NUM_BANDS - 5
    N_max = NUM_BANDS - 1 + 1
    efield = [np.abs(ms_x.get_efield(n + 1, bloch_phase = False)[..., 0, 1]) for n in range(N_min, N_max)]

    for i, field in enumerate(efield) :
        ax_eps.plot(field, label = i)

    ax_eps.legend()
plt.show()

