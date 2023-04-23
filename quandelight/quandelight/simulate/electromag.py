import numpy as np
import matplotlib.pyplot as plt
import meep as mp
from ..geometries import *
from ..utils import pprint, dtstring

meep_keys = ["PML_thickness", "resolution_factor", "solver_tol", "solver_cwtol", "solver_maxiter", "solver_cwmaxiter", "solver_L", "plot_result"]

default_meep_kwargs = {"PML_thickness":0.5,
                       "resolution_factor":10,
                       "solver_tol":1e-7,
                       "solver_cwtol":1e-10,
                       "solver_maxiter":100,
                       "solver_cwmaxiter":10_000,
                       "solver_L":10,
                       "plot_result" : False,}

########## ---------- RECTANGULAR SIM ----------

def sim_rectangular_cavity(epsilons, lamda, N, width, thickness, fill_center, **meep_kwargs):
    if "pads" in meep_kwargs.keys():
        pads = meep_kwargs["pads"]
    else :
        pads = [2, 2, 2]

    for key in meep_keys:
        if key not in meep_kwargs.keys():
            meep_kwargs[key] = default_meep_kwargs[key]

    eps1, eps2 = epsilons

    PLOT_RESULT = meep_kwargs["plot_result"]

    ## MEEP VARIABLES

    X_PADDING_WIDTH, Y_PADDING_WIDTH, Z_PADDING_WIDTH = meep_kwargs["pads"]

    PML_THICKNESS = meep_kwargs["PML_thickness"]

    RESOLUTION_FACTOR = meep_kwargs["resolution_factor"]

    SOLVER_TOL = meep_kwargs["solver_tol"]
    SOLVER_CWTOL = meep_kwargs["solver_cwtol"]
    SOLVER_MAXITER = meep_kwargs["solver_maxiter"]
    SOLVER_CWMAXITER = meep_kwargs["solver_cwmaxiter"]
    SOLVER_L = meep_kwargs["solver_L"]

    folder_name = "MEEP-EIG-"
    ### PROGRAM FLOW

    RESOLUTION = 4*RESOLUTION_FACTOR*np.sqrt(np.max((eps1, eps2)))/lamda
    date_and_time = dtstring()
    prefix = f"{folder_name}{date_and_time}"


    # determining whether the simulation is 1D. If that's the case, then the E-field will be along X because Meep only supports that, and the geometry will be along Z. Else, we can do whatever, but I usually orient the geometry along X and look for Ez.
    is_1d = width == 0 and thickness == 0
    is_3d = width != 0 and thickness != 0

    if is_1d :
        efield_vector = mp.Ex
        source_pos = mp.Vector3(0, 0, 0.05*lamda/np.sqrt(eps1))
        dim = 1
    elif is_3d :
        efield_vector = mp.Ez
        source_pos = mp.Vector3(0*.05*lamda/np.sqrt(eps1), 0.1*width, 0.1*thickness)
        dim = 3
    else :
        efield_vector = mp.Ez
        source_pos = mp.Vector3(0.05*lamda/np.sqrt(eps1), 0.1*width, 0)
        dim = 2


    pprint(f"SIMULATION IS {dim}D", "purple")
    # generating the geometry and the relevant quantities
    geometry, dbr_dims, omega_adim, band_width = dbr_rectangular(N, eps1, eps2, lamda, width, fill_center = fill_center, thickness = thickness, meep_1d = is_1d)

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
                                guessfreq=omega_adim,
                                cwtol=SOLVER_CWTOL,
                                cwmaxiters=SOLVER_CWMAXITER,
                                L=SOLVER_L)

    #comuting some figures of merit.
    omega_calc = eigfreq.real
    Q = eigfreq.real / (-2*eigfreq.imag)
    mode_volume = sim.modal_volume_in_box()

    diff = 100 - 100*omega_calc/omega_adim # difference in %

    if np.abs(diff) <= 5:
        color = "green"
    elif np.abs(diff <= 20):
        color = "yellow"
    else :
        color = "red"

    # outputting them
    printy=f"computed:{eigfreq:.2e}, expected {omega_adim:.2e}, diff {diff:.2f} %"
    pprint(printy, color = color)
    pprint(f"Q = {Q:.2e}", "cyan")
    pprint(f'Mode volume : {mode_volume:.2e}', color = "cyan")
    mp.output_epsilon(sim)
    mp.output_efield(sim)

    info_string = "MEEP EIGENSOLVER.\n" + rf"$\epsilon = ({eps1:.2f}, {eps2:.2f}), \lambda = {lamda:.2f}, W={width:.2e}, h={thickness:.2e}, N={N},$" +"\n" + rf" $pad=({X_PADDING_WIDTH:.2f}, {Y_PADDING_WIDTH:.2f}, {Z_PADDING_WIDTH:.2f}), res = {RESOLUTION}$" + "\n" + rf"Q = {Q:.2e}, V = {mode_volume:.2e}"+ "\n" + printy

    with open(f"{prefix}-out/params.txt", "w") as file :
        file.write(info_string)

    if is_1d:
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

    if PLOT_RESULT:
        plt.show()

    sim.reset_meep()
    return omega_adim, omega_calc, Q, mode_volume


########## ---------- CYLINDRICAL SIM ----------


def sim_cylindrical_cavity(epsilons, lamda, N, R_max, undercut_angle, fill_center, **meep_kwargs):
    if "pads" in meep_kwargs.keys():
        pads = meep_kwargs["pads"]
    else :
        pads = [2, 2]

    for key in meep_keys:
        if key not in meep_kwargs.keys():
            meep_kwargs[key] = default_meep_kwargs[key]

    eps1, eps2 = epsilons

    PLOT_RESULT = meep_kwargs["plot_result"]

    ## MEEP VARIABLES

    R_PADDING_WIDTH, Z_PADDING_WIDTH = meep_kwargs["pads"]

    PML_THICKNESS = meep_kwargs["PML_thickness"]

    RESOLUTION_FACTOR = meep_kwargs["resolution_factor"]

    SOLVER_TOL = meep_kwargs["solver_tol"]
    SOLVER_CWTOL = meep_kwargs["solver_cwtol"]
    SOLVER_MAXITER = meep_kwargs["solver_maxiter"]
    SOLVER_CWMAXITER = meep_kwargs["solver_cwmaxiter"]
    SOLVER_L = meep_kwargs["solver_L"]

    folder_name = "MEEP-CYL-EIG-"
    ### PROGRAM FLOW

    RESOLUTION = 4*RESOLUTION_FACTOR*np.sqrt(np.max((eps1, eps2)))/lamda

    excitation_vector = mp.Ep
    source_pos = (lamda/np.sqrt(eps1))*mp.Vector3(0.3, 0, 0.1)
    dim = 2

    date_and_time = dtstring()
    prefix = f"{folder_name}{date_and_time}"
    pprint(f"SIMULATION IS {dim}D", "purple")
    # generating the geometry and the relevant quantities
    geometry, dbr_dims, omega_adim, band_width = dbr_cylindrical(N, eps1, eps2, lamda, R_max, undercut_angle, fill_center = fill_center)

    # setting up the simulation
    cell = dbr_dims + mp.Vector3(R_PADDING_WIDTH*lamda,0, 2*Z_PADDING_WIDTH*lamda)

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

    sim.use_output_directory()
    sim.init_sim()

    # running the eigensolver
    eigfreq = sim.solve_eigfreq(tol=SOLVER_TOL,
                                maxiters=SOLVER_MAXITER,
                                #guessfreq=omega_adim,
                                cwtol=SOLVER_CWTOL,
                                cwmaxiters=SOLVER_CWMAXITER,
                                L=SOLVER_L)

    #computing some figures of merit.
    omega_calc = eigfreq.real
    Q = eigfreq.real / (-2*eigfreq.imag)
    mode_volume = sim.modal_volume_in_box()

    diff = 100 - 100*omega_calc/omega_adim # difference in %

    if np.abs(diff) <= 5:
        color = "green"
    elif np.abs(diff <= 20):
        color = "yellow"
    else :
        color = "red"

    # outputting them
    printy = f"computed:{eigfreq:.2e}, expected {omega_adim:.2e}, diff{diff:.2f} %"
    pprint(printy , color)
    pprint(f"Q = {Q:.2e}", "cyan")
    pprint(f'Mode volume : {mode_volume:.2e}', color = "cyan")

    info_string = "MEEP EIGENSOLVER.\n" + rf"$\epsilon = ({eps1:.2f}, {eps2:.2f}), \lambda = {lamda:.2f}, Rmax={R_max:.2e}, undercut ={undercut_angle:.2f}, N={N}$," + "\n"+ rf" pad=$({R_PADDING_WIDTH:.2f}, {Z_PADDING_WIDTH:.2f}), res = {RESOLUTION}$" + "\n" + rf"Q = {Q:.2e}, V = {mode_volume:.2e}"+ "\n" + printy

    with open(f"{prefix}-out/params.txt", "w") as file :
        file.write(info_string)

    mp.output_epsilon(sim)
    mp.output_efield(sim)

    sim.plot2D(fields = excitation_vector, plot_eps_flag=True, eps_parameters = {"interpolation":"none",})
    plt.suptitle(info_string)
    plt.tight_layout()
    plt.savefig(f"{prefix}-out/fig.png")
    if PLOT_RESULT :
        plt.show()

    sim.reset_meep()
    return omega_adim, omega_calc, Q, mode_volume
