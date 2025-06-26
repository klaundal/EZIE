import numpy as np
import matplotlib.pyplot as plt
from secsy import spherical, get_SECS_B_G_matrices, get_SECS_J_G_matrices, CSgrid, CSprojection
from apexpy import Apex
import ppigrf
import datetime as dt
d2r = np.pi / 180

## Regularization parameters - USE FOR TUNING
l1 = 1e3 # increase this to make solution smoother and magnitudes smaller
l2 = 1e5 # increase this to make solution align more in magnetic east-west dirction

# LOAD DATA
import fakedata
sc_lat   = fakedata.sc_lat  # array of SC latitudes during the pass
sc_lon   = fakedata.sc_lon  # array of SC longitudes during the pass
obs_lon  = fakedata.obs_lon # array of observation longitudes
obs_lat  = fakedata.obs_lat # array of observation latitudes
obs_d    = fakedata.obs_d   # array of observed total field values in nT
obs_err  = fakedata.obs_err # array of uncertainties

# grid parameters (adjust dimensions and resolution if necessary)
map_params = {'LRES':40., # grid resolution in L-direction [km]
              'WRES':20., # grid resolution in W-direction [km]
              'W': 3200, # along-track dimension of analysis grid (TODO: This should probably be automated)
              'L': 2000, # cross-track dimension of analysis grid (TODO: Same as above)
              'wshift':25, # shift the grid center wres km in cross-track direction # necessary?
              }

# Some constants
RE = 6371.2    # Earth radius [km]
RI = RE + 110  # Ionosphere radius [km]
OBSHEIGHT = 80 # Assumed observation height [km]
EPOCH = dt.datetime(2025, 6, 1) # IGRF magnetic field epoch (IGRF changes very slowly so this is not important to get right)

def get_grid(sc_lon0, sc_lat0, sc_ve0, sc_vn0, map_params): # set up grid
    position = (sc_lon0, sc_lat0)
    orientation = (-sc_vn0, sc_ve0) # align coordinate system such that xi axis points right wrt to satellite velocity vector, and eta along velocity
    projection = CSprojection(position, orientation)
    L, W, LRES, WRES, wshift = map_params['L'], map_params['W'], map_params['LRES'], map_params['WRES'], map_params['wshift']
    grid = CSgrid(projection, L, W, LRES, WRES, wshift = wshift, R = RI)
    return grid

def get_LL(grid, apx, hI): # set up matrix that produces gradients in the magnetic eastward direction, and use to construct regularization matrix LL:
    Le, Ln = grid.get_Le_Ln()
    f1, f2 = apx.basevectors_qd(grid.lat.flatten(), grid.lon.flatten(), hI, coords='geo')
    f1 = f1/np.linalg.norm(f1, axis = 0) # normalize
    L = Le * f1[0].reshape((-1, 1)) + Ln * f1[1].reshape((-1, 1))
    LL = L.T.dot(L)
    return LL


# grid center location
sc_lon0, sc_lat0 = sc_lon[len(sc_lon)//2], sc_lat[len(sc_lon)//2]

# calculate satellite velocity (unit is irrelevant, only direction)
ve, vn = spherical.tangent_vector(sc_lat[:-1], sc_lon[:-1],
                                  sc_lat[1 :], sc_lon[1: ])

ve = np.hstack((ve, np.nan))[len(sc_lon)//2]
vn = np.hstack((vn, np.nan))[len(sc_lon)//2]

grid = get_grid(sc_lon0, sc_lat0, ve, vn, map_params)

Be0, Bn0, Bu0 = map(np.ravel, ppigrf.igrf(obs_lon, obs_lat, OBSHEIGHT, EPOCH))
B0_vector = np.vstack((Be0, Bn0, Bu0))
B0 = np.linalg.norm(B0_vector, axis = 0)
b0 = B0_vector / B0
    
db_parallel = (obs_d**2 - B0**2) / (2 * B0)

Q = np.diag(1/obs_err)

GBe, GBn, GBu = get_SECS_B_G_matrices(obs_lat, obs_lon, OBSHEIGHT + RE, grid.lat, grid.lon, RI = RI)
G = GBe * b0[0].reshape((-1, 1)) + GBn * b0[1].reshape((-1, 1)) + GBu * b0[2].reshape((-1, 1))

apx = Apex(EPOCH.year, refh = 110)
LL = get_LL(grid, apx, RI - RE)
LL_mag = np.max(LL)

GTQG = G.T.dot(Q).dot(G)
GTQd = G.T.dot(Q).dot(db_parallel)
gtg_mag = np.median(np.diag(GTQG))

reg = l1*gtg_mag*np.eye(LL.shape[0]) + l2*gtg_mag/LL_mag*LL

# calculate model vector
m = np.linalg.lstsq(GTQG + reg, GTQd, rcond = 0)[0]

# matrices to evaluate the solution on a grid
GBegrid, GBngrid, GBugrid = get_SECS_B_G_matrices(grid.lat_mesh, grid.lon_mesh, OBSHEIGHT + RE, grid.lat, grid.lon, RI = RI)
Be_model = GBegrid.dot(m)
Bn_model = GBngrid.dot(m)
Bu_model = GBugrid.dot(m)

from secsy import CSplot
fig, ax = plt.subplots()
cax = CSplot(ax, grid)
cax.pcolormesh(Bu_model.reshape(grid.lat_mesh.shape), vmin = -200, vmax = 200, cmap = plt.cm.bwr)
cax.add_coastlines(resolution = '50m')
cax.scatter(obs_lon, obs_lat, marker = '.', c = 'green', s = 1)
cax.add_spherical_grid(color = 'grey')

plt.show()
