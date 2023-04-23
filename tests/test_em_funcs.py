import numpy as np
import matplotlib.pyplot as plt
import meep as mp
from quandelight.simulate.electromag import *

meep_config = {"PML_thickness":2,
               "resolution_factor":4,
               "solver_tol":1e-7,
               "solver_cwtol":1e-10,
               "solver_maxiter":100,
               "solver_cwmaxiter":10_000,
               "solver_L":10,
               "plot_result" : True,}

print(sim_cylindrical_cavity((3, 3), 0.925, 10, 4*0.925, 4, True, pads=[3, 3], **meep_config))
