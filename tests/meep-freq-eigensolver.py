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
lamda = 0.925

FILL_CENTER = True

N = 10

cavity_transverse_extent = 5*lamda/np.sqrt(eps1)
thickness = 0

## MEEP VARIABLES

X_PADDING_WIDTH = 2 + np.sqrt(2) # in lambdas, includes PML
Y_PADDING_WIDTH = 2 + np.sqrt(2)
Z_PADDING_WIDTH = 0

PML_THICKNESS = 2*lamda

RESOLUTION_FACTOR = 10

SOLVER_TOL = 1e-7 # default 1e-7
SOLVER_CWTOL = SOLVER_TOL*1e-3 # default SOLVER_TOL*1e-3

SOLVER_MAXITER = 100 # default 100
SOLVER_CWMAXITER = 10_000 # default 10^4
SOLVER_L = 10 # default 10

folder_name = "MEEP-EIG-"
### PROGRAM FLOW

RESOLUTION = 4*RESOLUTION_FACTOR*np.sqrt(np.max((eps1, eps2)))/lamda
date_and_time = dtstring()
prefix = f"{folder_name}{date_and_time}"


# determining whether the simulation is 1D. If that's the case, then the E-field will be along X because Meep only supports that, and the geometry will be along Z. Else, we can do whatever, but I usually orient the geometry along X and look for Ez.
is_1d = cavity_transverse_extent == 0 and thickness == 0
is_3d = cavity_transverse_extent != 0 and thickness != 0

if is_1d :
    efield_vector = mp.Ex
    source_pos = mp.Vector3(0, 0, 0.05*lamda/np.sqrt(eps1))
    dim = 1
elif is_3d :
    efield_vector = mp.Ez
    source_pos = mp.Vector3(0*.05*lamda/np.sqrt(eps1), 0.1*cavity_transverse_extent, 0.1*thickness)
    dim = 3
else :
    efield_vector = mp.Ez
    source_pos = mp.Vector3(0.05*lamda/np.sqrt(eps1), 0.1*cavity_transverse_extent, 0)
    dim = 2


pprint(f"SIMULATION IS {dim}D", "purple")
# generating the geometry and the relevant quantities
geometry, dbr_dims, omega_adim, band_width = dbr_rectangular(N, eps1, eps2, lamda, cavity_transverse_extent, fill_center = FILL_CENTER, thickness = thickness, meep_1d = is_1d)

# setting up the simulation
cell = dbr_dims + mp.Vector3(2*X_PADDING_WIDTH*lamda,2*Y_PADDING_WIDTH*lamda, 2*Z_PADDING_WIDTH*lamda)

sources = [mp.Source(mp.ContinuousSource(omega_adim, fwidth = omega_adim*band_width), efield_vector, center = source_pos)]

if is_1d:
    pml_layers = [mp.Absorber(PML_THICKNESS)]
else :
    pml_layers = [mp.PML(PML_THICKNESS)]

sim = mp.Simulation(cell_size = cell,
                    boundary_layers = pml_layers,
                    geometry = geometry,
                    sources = sources,
                    resolution = RESOLUTION,
                    force_complex_fields = True,
                    dimensions = dim,
                    filename_prefix=prefix)


sim.use_output_directory()
sim.init_sim()

# running the eigensolver
eigfreq = sim.solve_eigfreq(tol=SOLVER_TOL,
                            maxiters=SOLVER_MAXITER,
                            #guessfreq=omega_adim,
                            cwtol=SOLVER_CWTOL,
                            cwmaxiters=SOLVER_CWMAXITER,
                            L=SOLVER_L)

#comuting some figures of merit.
omega_calc = eigfreq.real
Q = eigfreq.real / (-2*eigfreq.imag)
mode_volume = sim.modal_volume_in_box()

# outputting them
printy=f"computed:{eigfreq:.2e}, expected {omega_adim:.2e}, diff {100 - 100*omega_calc/omega_adim:.2f} %"
pprint(printy, color = "green")
pprint(f"Q = {Q:.2e}", "yellow")
pprint(f'Mode volume : {mode_volume:.2e}', color = "red")
mp.output_epsilon(sim)
mp.output_efield(sim)

info_string = "MEEP EIGENSOLVER.\n" + rf"$\epsilon = ({eps1:.2f}, {eps2:.2f}), \lambda = {lamda:.2f}, W={cavity_transverse_extent:.2e}, h={thickness:.2e}, N={N},$" +"\n" + rf" $pad=({X_PADDING_WIDTH:.2f}, {Y_PADDING_WIDTH:.2f}, {Z_PADDING_WIDTH:.2f}), res = {RESOLUTION}$" + "\n" + rf"Q = {Q:.2e}, V = {mode_volume:.2e}"+ "\n" + printy

if is_1d :
    fig, ax = plt.subplots(tight_layout = True)
    ax2 = ax.twinx()
    efield = sim.get_array(center = mp.Vector3(), size=cell, component = efield_vector)
    eps = sim.get_array(center = mp.Vector3(), size=cell, component = mp.Dielectric)
    ax.plot(np.real(efield))
    ax2.plot(eps, color = 'lightgray', alpha=0.5)

    ax.set_xlabel("x (node number)")
    ax.set_ylabel(r"$E_x$")
    ax2.set_ylabel(r"$\varepsilon$")

    fig.suptitle(info_string)
    fig.savefig(f"{prefix}-out/fig.png")
else :
    sim.plot2D(fields = efield_vector, plot_eps_flag=True, eps_parameters = {
        "interpolation":"none",
    })
    plt.suptitle(info_string)
    plt.tight_layout()
    plt.savefig(f"{prefix}-out/fig.png")


plt.show()
