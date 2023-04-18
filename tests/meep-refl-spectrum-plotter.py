import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, hbar
import h5py
import math
from quandelight.geometries import dbr_rectangular
from quandelight.utils import pprint, dtstring

eps1 = 3
eps2 = 3.5
lamda = 5

FILL_CENTER = True

N = 100

cavity_transverse_extent = 0*lamda/np.sqrt(eps1)
thickness = 0

## MEEP VARIABLES

X_PADDING_WIDTH = 0 # in lambdas, includes PML
Y_PADDING_WIDTH = 0
Z_PADDING_WIDTH = 5

PML_THICKNESS = 2*lamda
FLUX_ZONE_THICKNESS = 0

RESOLUTION = 25
N_FREQ = 101

REMOVE_LEFT_SIDE = True

folder_name = "MEEP-REFL-SPEC-"
### PROGRAM FLOW

FILL_CENTER = FILL_CENTER and not(REMOVE_LEFT_SIDE)

date_and_time = dtstring()
prefix = f"{folder_name}{date_and_time}"


# determining whether the simulation is 1D. If that's the case, then the E-field will be along X because Meep only supports that, and the geometry will be along Z. Else, we can do whatever, but I usually orient the geometry along X and look for Ez.
is_1d = cavity_transverse_extent == 0 and thickness == 0
is_3d = cavity_transverse_extent != 0 and thickness != 0

if is_1d :
    efield_vector = mp.Ex
    source_pos = mp.Vector3(0, 0, 0.05*lamda/np.sqrt(eps1))
    dim = 1
    source_size = mp.Vector3()

elif is_3d :
    efield_vector = mp.Ez
    source_pos = mp.Vector3(0*.05*lamda/np.sqrt(eps1), 0, 0)
    dim = 3
    source_size = mp.Vector3(0, cavity_transverse_extent, thickness)
else :
    efield_vector = mp.Ez
    source_pos = mp.Vector3(0.05*lamda/np.sqrt(eps1), 0, 0)
    dim = 2
    source_size = mp.Vector3(0, cavity_transverse_extent, 0)



pprint(f"SIMULATION IS {dim}D", "purple")
# generating the geometry and the relevant quantities
geometry, dbr_dims, omega_adim, band_width = dbr_rectangular(N, eps1, eps2, lamda, cavity_transverse_extent, fill_center = FILL_CENTER, thickness = thickness, meep_1d = is_1d)

# setting up the simulation
cell = dbr_dims + mp.Vector3(2*X_PADDING_WIDTH*lamda,2*Y_PADDING_WIDTH*lamda, 2*Z_PADDING_WIDTH*lamda)

if REMOVE_LEFT_SIDE:
    if is_1d :
        geometry += [mp.Block(mp.Vector3(0, 0, cell.z/2), # right eps2 layer
                              center=mp.Vector3(0, 0, -cell.z/4),
                              material=mp.Medium(epsilon=1)),]
    else :
        geometry += [mp.Block(mp.Vector3(cell.x/2, cell.y, cell.z), # right eps2 layer
                              center=mp.Vector3(-cell.x/4, 0, 0),
                              material=mp.Medium(epsilon=1)),]


sources = [mp.Source(mp.GaussianSource(omega_adim, fwidth = 5*omega_adim),
                     efield_vector,
                     center = source_pos,
                     size = source_size)]

if is_1d:
    pml_layers = [mp.Absorber(PML_THICKNESS)]
    fr_center_t = dbr_dims/2
    fr_center_r = mp.Vector3()-dbr_dims/2
    fr_size = mp.Vector3()
else :
    pml_layers = [mp.PML(PML_THICKNESS)]
    fr_center_t = mp.Vector3(dbr_dims.x/2)# + (X_PADDING_WIDTH - PML_THICKNESS)/2, 0, 0)
    fr_center_r = mp.Vector3(2*source_pos.x, 0, 0)
    fr_size = mp.Vector3(0, cavity_transverse_extent, thickness)


### FIRST RUN WITHOUT GEOMETRY
sim = mp.Simulation(cell_size = cell,
                    boundary_layers = pml_layers,
                    sources = sources,
                    resolution = RESOLUTION,
                    dimensions = dim,
                    filename_prefix=prefix)

refl_fr = mp.FluxRegion(center = fr_center_r,
                        size = fr_size,)

refl = sim.add_flux(omega_adim, omega_adim*band_width, N_FREQ, refl_fr)

tran_fr = mp.FluxRegion(center = fr_center_t,
                        size = fr_size,)

tran = sim.add_flux(omega_adim, omega_adim*band_width, N_FREQ, tran_fr)

check_pt = fr_center_r


sim.run(until_after_sources = mp.stop_when_fields_decayed(50, efield_vector, check_pt, 1e-5))


empty_refl_data = sim.get_flux_data(refl)
empty_tran_flux = np.array(mp.get_fluxes(tran))

#REDOING WITH GEOMETRY.
sim.reset_meep()

sim = mp.Simulation(cell_size = cell,
                    boundary_layers = pml_layers,
                    geometry=geometry,
                    sources = sources,
                    resolution = RESOLUTION,
                    dimensions = dim,
                    filename_prefix=prefix)

refl = sim.add_flux(omega_adim, omega_adim*band_width, N_FREQ, refl_fr)
tran = sim.add_flux(omega_adim, omega_adim*band_width, N_FREQ, tran_fr)

if not(is_1d):
    sim.plot2D()
    plt.show()

sim.load_minus_flux_data(refl, empty_refl_data)

sim.run(until_after_sources = mp.stop_when_fields_decayed(50, efield_vector, check_pt, 1e-5))

res_refl_flux = np.array(mp.get_fluxes(refl))
res_tran_flux = np.array(mp.get_fluxes(tran))

flux_freqs = np.array(mp.get_flux_freqs(refl))


wl = flux_freqs
Rs = (-res_refl_flux / empty_tran_flux)[::-1]
Ts = (res_tran_flux/empty_tran_flux)[::-1]
Rs_from_Ts = 1 - Ts

if mp.am_master():
    plt.figure()
    plt.axvline(x = omega_adim, color = 'k', ls = '--')
    plt.axhline(y = 1, color = "lightgray")
    plt.grid(True)
    plt.axhline(color = 'k')
    plt.plot(wl, Rs, "bo-", alpha= 0.5, label="reflectance")
    plt.plot(wl, Rs_from_Ts, "yo-", alpha= 0.5, label="reflectance from transmittance")
    plt.plot(wl, Ts, "ro-", label="transmittance")
    #plt.plot(wl, 1 - Rs - Ts, "go-", label="loss")
    plt.xlabel(r"$\omega$")
    plt.legend(loc="upper right")
    plt.show()
