import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, hbar
from meep.mpb import ModeSolver
from meep import mpb
import h5py


eps1 = 1
eps2 = 3.45
lamda = 5

N = 12

cavity_transverse_extent = 0.5

## MPB VARIABLES

RESOLUTION = 32
NUM_BANDS = 2
INTERP_POINTS = 4


### FUNCTIONS

def gen_fabry_geometry(N, eps1, eps2, lamda, fill_center = False):
    s1 = np.sqrt(eps1)
    s2 = np.sqrt(eps2)

    lamda_0 = lamda

    lamda_1 = lamda/s1
    lamda_2 = lamda/s2

    lay_th_1 = lamda_1/4
    lay_th_2 = lamda_2/4

    half_cavity_width = lamda_0/2

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

geometry, sim_half_width, half_cavity_width = gen_fabry_geometry(N, eps1, eps2, lamda, fill_center = False)


cell = mp.Vector3(2*sim_half_width)

geometry_lattice = mp.Lattice(size=mp.Vector3(3*sim_half_width, 2*cavity_transverse_extent))

k_points = [mp.Vector3(),
            mp.Vector3(0.5, 0),
            mp.Vector3(0.5, 0.5),
            mp.Vector3(0, 0.5),
            mp.Vector3(),]          # Gamma

k_points = mp.interpolate(INTERP_POINTS, k_points)


ms = ModeSolver(num_bands=NUM_BANDS,
                k_points = k_points,
                geometry=geometry,
                geometry_lattice=geometry_lattice,
                resolution=RESOLUTION)

ms.run_te()

efield = mpb.output_efield(ms, 2)
ms.output_epsilon()
print("EFIELD OUTPUT", efield)

fig, ax = plt.subplots(1, 2, constrained_layout = True)
ax1, ax2 = ax
s = ms.all_freqs.T
for i in s :
    ax1.plot(i)
im = ax2.imshow(ms.get_epsilon(), aspect = 'auto', origin = 'lower', interpolation = 'none', cmap = 'binary')
#im2 = ax2.imshow(efield, aspect = 'auto', origin = 'lower', interpolation = 'none', cmap = "RdBu_r", alpha = 0.5)
fig.colorbar(im)
plt.show()
