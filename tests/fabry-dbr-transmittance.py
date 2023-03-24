# HEAVILY BASED ON MEEP EXAMPLE FILE :
# transmission around a 90-degree waveguide bend in 2d
import matplotlib.pyplot as plt
import numpy as np

import meep as mp
########## ---------------------------------------------------------------------- MY CODE
eps1 = 1
eps2 = 12
lamda = 15

FILL_CENTER = True

N = 32

cavity_transverse_extent = 1/8

## MPB VARIABLES

PADDING_WIDTH = 5*lamda
PML_THICKNESS = lamda

RESOLUTION = 16
NUM_BANDS = 2*(N + 1) + 1
INTERP_POINTS = 1

PLOT_GUESSES = True

# COMPUTING SOME RELEVANT QUANTITIES

n1 = np.sqrt(eps1)
n2 = np.sqrt(eps2)



lamda_1 = lamda/n1
lamda_2 = lamda/n2

lay_th_1 = lamda_1/4
lay_th_2 = lamda_2/4

lamda_0 = lamda_1 if FILL_CENTER else lamda

half_cavity_width = lamda_0/2

grating_periodicity = lay_th_1 + lay_th_2

omega_adim = (n1 + n2)/(4*n1*n2*grating_periodicity) # careful, this is a reduced unit wrt the periodicity of the bragg reflector
band_width = 4/np.pi*np.arcsin(np.abs(n1 - n2)/(n1 + n2))

sim_half_width = half_cavity_width + N*grating_periodicity

geometry = []

if FILL_CENTER :
    geometry += [mp.Block(mp.Vector3(2*half_cavity_width, cavity_transverse_extent, 0),
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



########## ----------------------------------------------------------------------

resolution = RESOLUTION  # pixels/um

sx = 2*(sim_half_width + PADDING_WIDTH)  # size of cell in X direction
sy = cavity_transverse_extent + 2*PADDING_WIDTH + 2*PML_THICKNESS # size of cell in Y direction
cell = mp.Vector3(sx, sy, 0)

w = cavity_transverse_extent
dpml = PML_THICKNESS
pml_layers = [mp.PML(PML_THICKNESS)]

fcen = omega_adim  # pulse center frequency
df = 4*band_width  # pulse width (in frequency)
sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(),
        size=mp.Vector3(0, cavity_transverse_extent, 0),
    )
]

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

nfreq = 100  # number of frequencies at which to compute flux

# transmitted flux
tran_fr = mp.FluxRegion(
    center=mp.Vector3(sim_half_width + (PADDING_WIDTH - dpml)/2,0, 0), size=mp.Vector3(0, 2 * w, 0)
)
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

pt = mp.Vector3(0.5 * sx - dpml - 0.5)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

# for normalization run, save flux fields data for reflection plane
straight_refl_data = sim.get_flux_data(refl) ################ TODO : SAVE THE TRANSMITTED THINGAMABOBBER

# save incident power for transmission plane
straight_tran_flux = mp.get_fluxes(tran)

sim.reset_meep()


sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

# reflected flux
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

tran_fr = mp.FluxRegion(
    center=mp.Vector3(wvg_xcen, 0.5 * sy - dpml - 0.5, 0), size=mp.Vector3(2 * w, 0, 0)
)
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

# for normal run, load negated fields to subtract incident from refl. fields
sim.load_minus_flux_data(refl, straight_refl_data)

pt = mp.Vector3(wvg_xcen, 0.5 * sy - dpml - 0.5)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

bend_refl_flux = mp.get_fluxes(refl)
bend_tran_flux = mp.get_fluxes(tran)

flux_freqs = mp.get_flux_freqs(refl)

wl = []
Rs = []
Ts = []
for i in range(nfreq):
    wl = np.append(wl, 1 / flux_freqs[i])
    Rs = np.append(Rs, -bend_refl_flux[i] / straight_tran_flux[i])
    Ts = np.append(Ts, bend_tran_flux[i] / straight_tran_flux[i])

if mp.am_master():
    plt.figure()
    plt.plot(wl, Rs, "bo-", label="reflectance")
    plt.plot(wl, Ts, "ro-", label="transmittance")
    plt.plot(wl, 1 - Rs - Ts, "go-", label="loss")
    plt.axis([5.0, 10.0, 0, 1])
    plt.xlabel("wavelength (μm)")
    plt.legend(loc="upper right")
    plt.show()
