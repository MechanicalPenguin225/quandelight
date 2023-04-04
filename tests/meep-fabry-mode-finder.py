import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, hbar
import h5py
import math
from quandelight.geometries import dbr_rectangular

eps1 = 1
eps2 = 12
lamda = 15

FILL_CENTER = True

N = 4

cavity_transverse_extent = 0
thickness = 0

## MEEP VARIABLES

X_PADDING_WIDTH = 0 # in lambdas, includes PML
Y_PADDING_WIDTH = 0
Z_PADDING_WIDTH = 1.1

PML_THICKNESS = 1*lamda

RESOLUTION = 16
geometry, dbr_dims, omega_adim, band_width = dbr_rectangular(N, eps1, eps2, lamda, cavity_transverse_extent, fill_center = FILL_CENTER, thickness = thickness, meep_1d = True)
### PROGRAM FLOW



cell = dbr_dims + mp.Vector3(2*X_PADDING_WIDTH*lamda,2*Y_PADDING_WIDTH*lamda, 2*Z_PADDING_WIDTH*lamda)

sources = [mp.Source(mp.ContinuousSource(omega_adim), mp.Ex, center = mp.Vector3(0, 0, 0.1))]

pml_layers = [mp.PML(PML_THICKNESS)]

sim = mp.Simulation(cell_size = cell,
                    boundary_layers = pml_layers,
                    geometry = geometry,
                    sources = sources,
                    resolution = RESOLUTION,
                    dimensions = 1,
                    force_complex_fields = True)

print(sim.cell_size)
sim.use_output_directory()

vals_e = []

def get_e_slice(sim):
    vals_e.append(sim.get_array(center=mp.Vector3(), size=mp.Vector3(0, 0, cell.z), component=mp.Ex))




pt = mp.Vector3(0, 0, 0.1*lamda)
harminv_instance = mp.Harminv(mp.Ex, pt, fcen = omega_adim, df = omega_adim*band_width)

sim.run(
    mp.at_beginning(mp.output_epsilon),  # only output epsilon at the first time step by wrapping the `mp.output_epsilon` call inside `mp.at_beginning`
    mp.to_appended("ex", mp.after_sources(mp.at_every(1.5, mp.output_efield_x))), # output Ez every 0.6 timestep in the same way
    mp.after_sources(harminv_instance),
    mp.at_every(1, get_e_slice),
        #until=100000)
    until_after_sources=mp.stop_when_fields_decayed(60, mp.Ex, pt, 1e-3))


eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
ex_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ex)


modes = harminv_instance.modes
print([mode.Q for mode in modes])

### GENERATING GIF

#### PLOTTING
fig = plt.figure()

ax_eps = plt.subplot2grid((2, 2), (0, 0), rowspan = 1, colspan = 1, fig = fig)
ax_e = plt.subplot2grid((2, 2), (1, 0), rowspan = 1, colspan = 1, fig = fig, sharex = ax_eps)
ax_im = plt.subplot2grid((2, 2), (0, 1), rowspan = 2, colspan = 1, fig = fig)

ax_eps.plot(eps_data)
ax_e.plot(ex_data)

im = ax_im.imshow(vals_e, interpolation='none', cmap='RdBu', aspect = "auto")
fig.colorbar(im)

plt.show()
