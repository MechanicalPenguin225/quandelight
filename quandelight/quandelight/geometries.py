import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, hbar
from meep.mpb import ModeSolver
from meep import mpb


# DEFINING FUNCTIONS

def dbr_rectangular(N = 8, eps1=3, eps2=3.45, lamda=5, cavity_transverse_extent=1, fill_center = True, thickness = 0, meep_1d=False):
    """generates a rectangular DBR cavity geometry (confined direction along x if 2d or 3d, along z if 1d). DBR cavity is centered at origin.
    Inputs :
    - N : int or 2-tuple of ints, number of layers on each side of the cavity. if N is a tuple of 2 intz (e.g. (5, 8)), it is the number of layers on the left (neg. x) and right (pos. x) respectively
    - eps1 : float >= 1, permittivity of one layer in the pairs (and also of the cavity if fill_center is True).
    - eps2: float >= 1, permittivity of the other layer.
    - lambda : float > 0, wavelength that the DBR resonator confines.
    - cavity_transverse_extent : float >= 0, transverse width of the resonator. NOTE : this will also affect the allowed wavelengths in the cavity.
    - fill_center : bool, whether the cavity part is the same material as eps1 (True) or is air (False).
    - thickness : float >=0, z-thickness of the DBR. if 0, the DBR is 2D.
    - meep_1d : bool, whether to rotate the structure to fit 1D Meep's requirements.

    Outputs :
    - geometry : list of MEEP geometries.
    - dims : mp.Vector3 : dimensions of the DBR
    - omega : photonic band gap center's planned angular frequency in the appropriate reduced units.
    - band_width : photonic band gap width (as a fraction of omega_adim).
    """

    n1 = np.sqrt(eps1)
    n2 = np.sqrt(eps2)

    M = max(N) if type(N) is tuple else N

    if meep_1d :
        rotation_matrix = np.array([[0, 1, 0],
                                    [0, 0, 1],
                                    [1, 0, 0]]) # I'm not using the rotation function that Meep implements because it requires radians and thus had a little error when roatating by exactly 90°. Instead I'm switching around Z and x. I'm not switching Z and Y bc this allows me to use this setting for 2D setups in cylindrical coordinates.
    else :
        rotation_matrix = np.eye(3)


    lamda_1 = lamda/n1
    lamda_2 = lamda/n2

    lay_th_1 = lamda_1/4
    lay_th_2 = lamda_2/4

    lamda_0 = lamda_1 if fill_center else lamda

    half_cavity_width = lamda_0/2

    grating_periodicity = lay_th_1 + lay_th_2

    omega = (n1 + n2)/(4*n1*n2*grating_periodicity) # careful, this is a reduced unit wrt the periodicity of the bragg reflector
    band_width = 4/np.pi*np.arcsin(np.abs(n1 - n2)/(n1 + n2))

    sim_half_width = half_cavity_width + M*grating_periodicity

    geometry = []

    if fill_center :
        center_block_dims = rotation_matrix@np.array([2*half_cavity_width, cavity_transverse_extent, thickness])
        geometry += [mp.Block(mp.Vector3(*center_block_dims),
                              center=mp.Vector3(),
                              material=mp.Medium(epsilon=eps1))]

    if type(N) is int :
        for i in range(N):

            bilayer_edge = half_cavity_width + i*grating_periodicity

            eps2_layer_center = mp.Vector3(*(rotation_matrix@np.array([bilayer_edge + 0.5*lay_th_2, 0, 0])))
            eps1_layer_center = mp.Vector3(*(rotation_matrix@np.array([bilayer_edge + lay_th_2 + 0.5*lay_th_1, 0, 0])))

            dims_eps2_block = rotation_matrix@np.array([lay_th_2, cavity_transverse_extent, thickness])
            dims_eps1_block = rotation_matrix@np.array([lay_th_1, cavity_transverse_extent, thickness])

            geometry += [mp.Block(mp.Vector3(*dims_eps2_block), # right eps2 layer
                                  center=eps2_layer_center,
                                  material=mp.Medium(epsilon=eps2)),
                         mp.Block(mp.Vector3(*dims_eps2_block),  # left eps2 layer
                                  center= mp.Vector3()-eps2_layer_center,
                                  material=mp.Medium(epsilon=eps2)),
                         mp.Block(mp.Vector3(*dims_eps1_block),  # right eps1 layer
                                  center=eps1_layer_center,
                                  material=mp.Medium(epsilon=eps1)),
                         mp.Block(mp.Vector3(*dims_eps1_block),  # left eps1 layer
                                  center= mp.Vector3()-eps1_layer_center,
                                  material=mp.Medium(epsilon=eps1))]
    elif type(N) is tuple and len(N) == 2 :
        N_l, N_r = N
        for i in range(max(N_l, N_r)):

            bilayer_edge = half_cavity_width + i*grating_periodicity

            eps2_layer_center = mp.Vector3(*(rotation_matrix@np.array([bilayer_edge + 0.5*lay_th_2, 0, 0])))
            eps1_layer_center = mp.Vector3(*(rotation_matrix@np.array([bilayer_edge + lay_th_2 + 0.5*lay_th_1, 0, 0])))

            dims_eps2_block = rotation_matrix@np.array([lay_th_2, cavity_transverse_extent, thickness])
            dims_eps1_block = rotation_matrix@np.array([lay_th_1, cavity_transverse_extent, thickness])

            if i < N_r:
                geometry += [mp.Block(mp.Vector3(*dims_eps1_block),  # right eps1 layer
                                      center=eps1_layer_center,
                                      material=mp.Medium(epsilon=eps1)),
                             mp.Block(mp.Vector3(*dims_eps2_block), # right eps2 layer
                                      center=eps2_layer_center,
                                      material=mp.Medium(epsilon=eps2)),
                             ]
            if i < N_l:
                geometry += [mp.Block(mp.Vector3(*dims_eps1_block),  # left eps1 layer
                                      center= mp.Vector3()-eps1_layer_center,
                                      material=mp.Medium(epsilon=eps1)),
                             mp.Block(mp.Vector3(*dims_eps2_block),  # left eps2 layer
                                      center= mp.Vector3()-eps2_layer_center,
                                      material=mp.Medium(epsilon=eps2))]
    else :
        raise ValueError("Wrong type of argument for N.")

    dims = mp.Vector3(*(rotation_matrix@np.array([2*sim_half_width, cavity_transverse_extent, thickness])))

    return geometry, dims, omega, band_width



def dbr_cylindrical(N = 8, eps1=3, eps2=3.45, lamda=5, R_max = 2, undercut_angle=4, fill_center = True):
    """generates a cylindrical (possibly tapered) DBR cavity geometry. DBR cavity is centered at origin. This is meant to be used in a cylindrical sim.
    IMPORTANT NOTE : if you use two different DBR layer numbers, make sure they are in ascending order.
    Inputs :
    - N : int or 2-tuple of ints, number of layers on each side of the cavity. if N is a tuple of 2 intz (e.g. (5, 8)), it is the number of layers on the left (neg. x) and right (pos. x) respectively
    - eps1 : float >= 1, permittivity of one layer in the pairs (and also of the cavity if fill_center is True).
    - eps2: float >= 1, permittivity of the other layer.
    - lambda : float > 0, wavelength that the DBR resonator confines.
    - R_max : float >= 0, max radius of the resonator. NOTE : this will also affect the allowed wavelengths in the cavity.
    - undercut_angle : float, angle (in degrees) of the undercut making the pillar conical. NOTE : this will also affect the allowed wavelengths in the cavity.
    - fill_center : bool, whether the cavity part is the same material as eps1 (True) or is air (False).

    Outputs :
    - geometry : list of MEEP geometries.
    - dims : mp.Vector3 : dimensions of the DBR
    - omega : photonic band gap center's planned angular frequency in the appropriate reduced units.
    - band_width : photonic band gap width (as a fraction of omega_adim).
    """
    undercut_angle_radians = undercut_angle/180*np.pi
    undercut_direction = mp.Vector3(0, 0, 1).rotate(mp.Vector3(0, 1, 0), -undercut_angle_radians) # will be the skew axis of our setup
    # generate a 2D rectangular resonator along the Z axis
    geometry, dims, omega, band_width = dbr_rectangular(N, eps1, eps2, lamda, cavity_transverse_extent=2*R_max, fill_center = fill_center, thickness = 0, meep_1d=True)

    L = dims.z
    delta_R = np.tan(undercut_angle_radians)*L # radius at the other end of the *bounding box given by dims*.

    geometry += [mp.Block(mp.Vector3(dims),# this block erases the r<0 part,
                          center= mp.Vector3(-dims.x/2, 0, 0),
                          material=mp.Medium(epsilon=1)),

                 mp.Block(mp.Vector3(L + delta_R, delta_R, 0), # this block creates the wedge part by cutting out.
                          e1 = mp.Vector3(delta_R, 0, L), # makes the block skewed.
                          e2 = mp.Vector3(1, 0, 0),# second dimension of the block is gonna be the extent along r
                          e3 = mp.Vector3(0, 1, 0), # we don't talk about the third direction.
                          center= mp.Vector3(R_max, 0, 0),
                          material=mp.Medium(epsilon=1)),
                 ]

    return geometry, dims, omega, band_width
