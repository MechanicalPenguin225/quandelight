import meep as mp
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.constants import e, hbar
import h5py
import math
from quandelight.geometries import dbr_cylindrical
from quandelight.utils import pprint, dtstring

eps1 = 3
eps2 = 3
lamda = 1

FILL_CENTER = True

N = 10

R_max = 4*lamda/np.sqrt(eps1)
undercut_angle = 4

## MEEP VARIABLES

R_PADDING_WIDTH = 2 # in lambdas, includes PML
Z_PADDING_WIDTH = 2

PML_THICKNESS = lamda # in sim units

RESOLUTION = 1

SOLVER_TOL = 1e-7 # default 1e-7
SOLVER_CWTOL = SOLVER_TOL*1e-3 # default SOLVER_TOL*1e-3

SOLVER_MAXITER = 100 # default 100
SOLVER_CWMAXITER = 10_000 # default 10^4
SOLVER_L = 10 # default 10

PRE_PLOT = False
folder_name = "MEEP-CYL-EIG-"
### PROGRAM FLOW

excitation_vector = mp.Er
source_pos = (lamda/np.sqrt(eps1))*mp.Vector3(0.1, 0, 0.1)
dim = 2

date_and_time = dtstring()
prefix = f"{folder_name}{date_and_time}"
pprint(f"SIMULATION IS {dim}D", "purple")
# generating the geometry and the relevant quantities
geometry, dbr_dims, omega_adim, band_width = dbr_cylindrical(N, eps1, eps2, lamda, R_max, undercut_angle, fill_center = FILL_CENTER)

# setting up the simulation
cell = dbr_dims + mp.Vector3(2*R_PADDING_WIDTH*lamda,0, 2*Z_PADDING_WIDTH*lamda)

sources = [mp.Source(mp.ContinuousSource(omega_adim, fwidth = omega_adim*band_width), excitation_vector, center = source_pos)]

pml_layers = [mp.PML(PML_THICKNESS)]

sim = mp.Simulation(cell_size = cell,
                    boundary_layers = pml_layers,
                    geometry = geometry,
                    sources = sources,
                    resolution = RESOLUTION,
                    force_complex_fields = True,
                    dimensions = mp.CYLINDRICAL,
                    filename_prefix=prefix)
if PRE_PLOT:
    sim.plot2D(plot_eps_flag=True, eps_parameters = {
        "interpolation":"none",
    })
    plt.axis('equal')
    plt.show()


    sys.exit()

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
printy = f"computed:{eigfreq:.2e}, expected {omega_adim:.2e}, diff{100 - 100*omega_calc/omega_adim:.2f} %"
pprint(printy , color = "green")
pprint(f"Q = {Q:.2e}", "yellow")
pprint(f'Mode volume : {mode_volume:.2e}', color = "red")

info_string = "MEEP EIGENSOLVER.\n" + rf"$\epsilon = ({eps1:.2f}, {eps2:.2f}), \lambda = {lamda:.2f}, Rmax={R_max:.2e}, undercut ={undercut_angle:.2f}, N={N}$," + "\n"+ rf" pad=$({R_PADDING_WIDTH:.2f}, {Z_PADDING_WIDTH:.2f}), res = {RESOLUTION}$" + "\n" + rf"Q = {Q:.2e}, V = {mode_volume:.2e}"+ "\n" + printy

mp.output_epsilon(sim)
mp.output_efield(sim)

sim.plot2D(fields = excitation_vector, plot_eps_flag=True, eps_parameters = {"interpolation":"none",})
plt.suptitle(info_string)
plt.tight_layout()
plt.savefig(f"{prefix}-out/fig.png")
plt.show()
