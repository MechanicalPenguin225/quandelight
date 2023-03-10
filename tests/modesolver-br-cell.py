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

cavity_transverse_extent = 0.5

padding_x = 0
padding_y = 0
## MPB VARIABLES

RESOLUTION = 64
NUM_BANDS =2
INTERP_POINTS = 151


# DEPENDENT PARAMS


n1 = np.sqrt(eps1)
n2 = np.sqrt(eps2)

lamda_1 = lamda/n1
lamda_2 = lamda/n2

lay_th_1 = lamda_1/4
lay_th_2 = lamda_2/4


grating_periodicity = lay_th_1 + lay_th_2

omega_adim = (n1 + n2)/(4*n1*n2*grating_periodicity)
band_width = 4/np.pi*np.arcsin(np.abs(n1 - n2)/(n1 + n2))


### PROGRAM FLOW

geometry = [mp.Block(mp.Vector3(lay_th_2, cavity_transverse_extent, 0), # right eps2 layer
                     center=mp.Vector3(-grating_periodicity/2 + lay_th_2/2),
                     material=mp.Medium(epsilon=eps2)),
            mp.Block(mp.Vector3(lay_th_1, cavity_transverse_extent, 0), # right eps2 layer
                     center=mp.Vector3(grating_periodicity/2  - lay_th_1/2),
                     material=mp.Medium(epsilon=eps1))]



cell = mp.Vector3(grating_periodicity + 2*padding_x, cavity_transverse_extent + 2*padding_y)
a_x = cell.x
a_y = cell.y


geometry_lattice = mp.Lattice(size=cell)

# Making k point lists
k_points_te = [mp.Vector3(),
            mp.Vector3(0.5, 0),]          # Gamma

k_points_tm = [mp.Vector3(),
            mp.Vector3(0.5, 0),]          # Gamma

k_points_te = mp.interpolate(INTERP_POINTS, k_points_te)
k_points_tm = mp.interpolate(INTERP_POINTS, k_points_tm)

# Modesolver for x direction
ms_te = ModeSolver(num_bands=NUM_BANDS,
                k_points = k_points_te,
                geometry=geometry,
                geometry_lattice=geometry_lattice,
                resolution=RESOLUTION)

ms_te.run_te()
omegas_te = ms_te.all_freqs.T
k_vals_te = np.array([v.norm() for v in k_points_te])/a_x

# Modesolver for y direction
ms_tm = ModeSolver(num_bands=NUM_BANDS,
                k_points = k_points_tm,
                geometry=geometry,
                geometry_lattice=geometry_lattice,
                resolution=RESOLUTION)

ms_tm.run_tm()
omegas_tm = ms_tm.all_freqs.T
k_vals_tm = np.array([v.norm() for v in k_points_tm])/a_x


# ---------- PLOTTING ----------

fig, ax = plt.subplots(1, 5, constrained_layout = True)
ax_te, ax_te_n, ax_tm, ax_tm_n, ax_eps = ax

ax_te.set_xlabel(r"$k_te$")
ax_te_n.set_xlabel(r"$k_te$")
ax_tm_n.set_xlabel(r"$k_tm$")
ax_tm.set_xlabel(r"$k_tm$")

ax_te_n.set_ylim(0, 3)
ax_tm_n.set_ylim(0, 3)

for axis in [ax_te, ax_tm]:
    axis.set_ylabel(r'$\omega$')
    axis.axhspan(ymin = omega_adim*(1 - band_width / 2), ymax = omega_adim*(1 + band_width / 2), color = 'lightgray', alpha = 0.5)
    axis.axhline(y = omega_adim, color = 'lightgray', ls = '--')

for axis in [ax_te_n, ax_tm_n]:
    axis.set_ylabel(r"$n_eff$")
    axis.axhline(y = np.sqrt(eps2), color = 'lightgray', label = r"$n_2$")
    axis.axhline(y = np.sqrt(eps1), color = 'lightgray', ls = '--', label = r"$n_1$")


# Plotting for x

for band_te, omega_points_te in enumerate(omegas_te):
    omega_plot = ax_te.plot(k_vals_te, omega_points_te, label = str(band_te))
    ax_te_n.plot(k_vals_te, omega_points_te/k_vals_te, ls = '--', color = omega_plot[0].get_color())


# Plotting for y

for band_tm, omega_points_tm in enumerate(omegas_tm):
    omega_plot = ax_tm.plot(k_vals_tm, omega_points_tm, label = str(band_tm))
    ax_tm_n.plot(k_vals_tm, omega_points_tm/k_vals_tm, ls = '--', color = omega_plot[0].get_color())



# plotting eps

im = ax_eps.imshow(ms_te.get_epsilon(), aspect = 'auto', origin = 'lower', interpolation = 'none', cmap = 'binary')
#im2 = ax_tm.imshow(efield, aspect = 'auto', origin = 'lower', interpolation = 'none', cmap = "RdBu_r", alpha = 0.5)

fig.colorbar(im)

#ax_te.legend()
#ax_tm.legend()
plt.show()
