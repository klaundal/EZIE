import numpy as np
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices, CSgrid, CSprojection
import cases
import pandas as pd
from dipole import geo2mag
from secsy import spherical 
import datetime as dt
import matplotlib.pyplot as plt
from pysymmetry.visualization.grids import sdarngrid
from importlib import reload
reload(cases) 
d2r = np.pi / 180

RE = 6372.e3
RI = RE + 120e3
DLAT = .5


info = cases.cases['case_4']

OBSHEIGHT = info['observation_height']

d2r = np.pi / 180

LRES = 40. # spatial resolution of SECS grid along satellite track
WRES = 20. # spatial resolution perpendicular to satellite tarck
wshift = info['wshift'] # shift center of grid wshift km to the right of the satellite (rel to velocity)
DT  = info['DT'] # size of time window [min]
RI  = (6371.2 + 110) * 1e3 # SECS height (m)

data = pd.read_pickle(info['filename'])

# convert all geographic coordinates and vector components in data to geomagnetic:
for i in range(4):
    i = i + 1
    _, _, data['dbe_measured_' + str(i)], data['dbn_measured_' + str(i)] = geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_measured_' + str(i)].values, data['dbn_measured_' + str(i)].values, epoch = 2020)
    data['lat_' + str(i)], data['lon_' + str(i)], data['dbe_' + str(i)], data['dbn_' + str(i)] = geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_' + str(i)].values, data['dbn_' + str(i)].values, epoch = 2020)
data['sat_lat'], data['sat_lon'] = geo2mag(data['sat_lat'].values, data['sat_lon'].values, epoch = 2020)

# calculate SC velocity
te, tn = spherical.tangent_vector(data['sat_lat'][:-1].values, data['sat_lon'][:-1].values,
                                  data['sat_lat'][1 :].values, data['sat_lon'][1: ].values)

data['ve'] = np.hstack((te, np.nan))
data['vn'] = np.hstack((tn, np.nan))

# get index of central point of analysis interval:
tm = info['tm']

# limits of analysis interval:
t0 = data.index[data.index.get_loc(tm - dt.timedelta(seconds = DT//2 * 60), method = 'nearest')]
t1 = data.index[data.index.get_loc(tm + dt.timedelta(seconds = DT//2 * 60), method = 'nearest')]

# get unit vectors pointing at satellite (Cartesian vectors)
rs = []
for t in [t0, tm, t1]:
    rs.append(np.array([np.cos(data.loc[t, 'sat_lat'] * d2r) * np.cos(data.loc[t, 'sat_lon'] * d2r),
                        np.cos(data.loc[t, 'sat_lat'] * d2r) * np.sin(data.loc[t, 'sat_lon'] * d2r),
                        np.sin(data.loc[t, 'sat_lat'] * d2r)]))

# dimensions of analysis region/grid (in km)
W = 400 + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km
L = 1200
print(L, W)

# set up the cubed sphere projection
v  = np.array((data.loc[tm, 've'], data.loc[tm, 'vn']))
#angle = np.arctan2(v[1], v[0]) / d2r + np.pi/2 

orientation = np.array([v[1], -v[0]]) # align coordinate system such that xi axis points right wrt to satellite velocity vector, and eta along velocity

p = data.loc[tm, 'sat_lon'], data.loc[tm, 'sat_lat']
projection = CSprojection(p, orientation)
grid = CSgrid(projection, L, W, LRES, WRES, wshift = wshift)

# make an extended grid
xi_e   = np.r_[grid. xi_mesh.min() - 10 * grid.dxi : grid. xi_mesh.max() + 10 * grid. dxi:grid.dxi ] - grid.dxi  / 2 
eta_e  = np.r_[grid.eta_mesh.min() - 10 * grid.deta: grid.eta_mesh.max() + 10 * grid.deta:grid.deta] - grid.deta / 2 

extended_grid = CSgrid(CSprojection(grid.projection.position, grid.projection.orientation),
                       grid.L, grid.W, grid.Lres, grid.Wres, 
                       edges = (xi_e, eta_e), R = RI)


#extended_grid = CSgrid(projection, 2*L, 2*W, LRES, WRES, wshift = wshift)

# get maps of MHD magnetic fields:
mhdBu =  info['mhdfunc'](grid.lat.flatten(), grid.lon.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
mhdBe =  info['mhdfunc'](grid.lat.flatten(), grid.lon.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
mhdBn = -info['mhdfunc'](grid.lat.flatten(), grid.lon.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])


Ge, Gn, Gu = get_SECS_B_G_matrices(grid.lat, grid.lon, RE + OBSHEIGHT * 1e3, extended_grid.lat[::2, ::2], extended_grid.lon[::2, ::2], RI = RI)

m = np.linalg.lstsq(np.vstack((Ge, Gn, Gu)), np.hstack((mhdBe, mhdBn, mhdBu)), rcond = 1e-2)[0]
#m = np.linalg.lstsq(np.vstack((Gu)), np.hstack((mhdBu)), rcond = 1e-4)[0]

fig, axes = plt.subplots(nrows = 3, ncols = 2)
for ax, G, B in zip(axes, [Ge, Gn, Gu], [mhdBe, mhdBn, mhdBu]):
    cntrs = ax[0].contourf(grid.xi, grid.eta, G.dot(m).reshape(grid.shape), cmap = plt.cm.bwr, levels = np.linspace(-600, 600, 22))
    ax[1].contourf(        grid.xi, grid.eta,        B.reshape(grid.shape), cmap = plt.cm.bwr, levels = cntrs.levels)
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')


