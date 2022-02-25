""" find out how large the map window must be for the central portion not to change any more 
"""

import numpy as np
import datetime as dt
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices, CSgrid, CSprojection
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dipole import geo2mag
from secsy import spherical 
from simulation_utils import get_MHD_jeq, get_MHD_dB, get_MHD_dB_new
import pandas as pd
import os
import cases
from importlib import reload
reload(cases) 

timestep = 30 # time resolution of maps - also the length of the central portion



info = cases.cases['case_5']

OBSHEIGHT = info['observation_height']

d2r = np.pi / 180

LRES = 40. # spatial resolution of SECS grid along satellite track
WRES = 20. # spatial resolution perpendicular to satellite tarck
wshift = info['wshift'] # shift center of grid wshift km to the right of the satellite (rel to velocity)
timeres = info['timeres']
RI  = (6371.2 + 110) * 1e3 # SECS height (m)
RE  = 6371.2e3

data = pd.read_pickle(info['filename'])

# convert all geographic coordinates and vector components in data to geomagnetic:
for i in range(4):
    i = i + 1
    #_, _, data['dbe_measured_' + str(i)], data['dbn_measured_' + str(i)] = geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_measured_' + str(i)].values, data['dbn_measured_' + str(i)].values, epoch = 2020)
    #data['lat_' + str(i)], data['lon_' + str(i)], data['dbe_' + str(i)], data['dbn_' + str(i)] = geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_' + str(i)].values, data['dbn_' + str(i)].values, epoch = 2020)
    data['lat_' + str(i)], data['lon_' + str(i)], data['dbe_' + str(i)], data['dbn_' + str(i)] = geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_' + str(i)].values, data['dbn_' + str(i)].values, epoch = 2020)
data['sat_lat'], data['sat_lon'] = geo2mag(data['sat_lat'].values, data['sat_lon'].values, epoch = 2020)
#data['sat_lon']+=180

# calculate SC velocity
te, tn = spherical.tangent_vector(data['sat_lat'][:-1].values, data['sat_lon'][:-1].values,
                                  data['sat_lat'][1 :].values, data['sat_lon'][1: ].values)

data['ve'] = np.hstack((te, np.nan))
data['vn'] = np.hstack((tn, np.nan))

# get index of central point of analysis interval:
tm = info['tm']


# set up the cubed sphere projection
v  = np.array((data.loc[tm, 've'], data.loc[tm, 'vn']))
orientation = np.array([v[1], -v[0]]) # align coordinate system such that xi axis points right wrt to satellite velocity vector, and eta along velocity
p = data.loc[tm, 'sat_lon'], data.loc[tm, 'sat_lat']
projection = CSprojection(p, orientation)


# limits of analysis interval:
t0 = data.index[data.index.get_loc(tm - dt.timedelta(seconds = timestep//2), method = 'nearest')]
t1 = data.index[data.index.get_loc(tm + dt.timedelta(seconds = timestep//2), method = 'nearest')]

# get unit vectors pointing at satellite (Cartesian vectors)
rs = []
for t in [t0, tm, t1]:
    rs.append(np.array([np.cos(data.loc[t, 'sat_lat'] * d2r) * np.cos(data.loc[t, 'sat_lon'] * d2r),
                        np.cos(data.loc[t, 'sat_lat'] * d2r) * np.sin(data.loc[t, 'sat_lon'] * d2r),
                        np.sin(data.loc[t, 'sat_lat'] * d2r)]))

# dimensions of analysis region/grid (in km)
W = 200 + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km
L = 1500
print(L, W)

grid0 = CSgrid(projection, (L // LRES) * LRES, (W // WRES // 2 * 2) * WRES, LRES, WRES) # make sure that the width is an even multiple of WRES
Bu0 = np.zeros(grid0.size)

diffs = []
dts = []
DT = timestep
while True:
    DT += 10 # increase window size by 10 seconds
    Bu0_old = Bu0

    # limits of analysis interval:
    t0 = data.index[data.index.get_loc(tm - dt.timedelta(seconds = DT//2), method = 'nearest')]
    t1 = data.index[data.index.get_loc(tm + dt.timedelta(seconds = DT//2), method = 'nearest')]

    # get unit vectors pointing at satellite (Cartesian vectors)
    rs = []
    for t in [t0, tm, t1]:
        rs.append(np.array([np.cos(data.loc[t, 'sat_lat'] * d2r) * np.cos(data.loc[t, 'sat_lon'] * d2r),
                            np.cos(data.loc[t, 'sat_lat'] * d2r) * np.sin(data.loc[t, 'sat_lon'] * d2r),
                            np.sin(data.loc[t, 'sat_lat'] * d2r)]))

    # dimensions of analysis region/grid (in km)
    W = 200 + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km
    L = 1500


    grid = CSgrid(projection, (L // LRES) * LRES, (W // WRES // 2 * 2) * WRES, LRES, WRES) # make sure that the width is an even multiple of WRES
    Le, Ln = grid.get_Le_Ln()
    LL = Le.T.dot(Le) # matrix for calculation of eastward gradient - eastward in magnetic since all coords above have been converted to dipole coords

    obs = {'lat': [], 'lon': [], 'Be': [], 'Bn': [], 'Bu': [], 'cov_ee': [], 'cov_nn': [], 'cov_uu': [], 'cov_en': [], 'cov_eu': [], 'cov_nu': []}
    for i in range(4):
        obs['lat'] += list(data.loc[t0:t1, 'lat_' + str(i + 1)].values)
        obs['lon'] += list(data.loc[t0:t1, 'lon_' + str(i + 1)].values)
        obs['Be' ] += list(data.loc[t0:t1, 'dbe_measured_'  + str(i + 1)].values * info['signs'][0])
        obs['Bn' ] += list(data.loc[t0:t1, 'dbn_measured_'  + str(i + 1)].values * info['signs'][1])
        obs['Bu' ] += list(data.loc[t0:t1, 'dbu_measured_'  + str(i + 1)].values * info['signs'][2])
        obs['cov_ee'] += list(data.loc[t0:t1, 'cov_ee_' + str(i + 1)].values)
        obs['cov_nn'] += list(data.loc[t0:t1, 'cov_nn_' + str(i + 1)].values)
        obs['cov_uu'] += list(data.loc[t0:t1, 'cov_uu_' + str(i + 1)].values)
        obs['cov_en'] += list(data.loc[t0:t1, 'cov_en_' + str(i + 1)].values)
        obs['cov_eu'] += list(data.loc[t0:t1, 'cov_eu_' + str(i + 1)].values)
        obs['cov_nu'] += list(data.loc[t0:t1, 'cov_nu_' + str(i + 1)].values)

    # construct covariance matrix and invert it
    Wen = np.diagflat(obs['cov_en'])
    Weu = np.diagflat(obs['cov_eu'])
    Wnu = np.diagflat(obs['cov_nu'])
    Wee = np.diagflat(obs['cov_ee'])
    Wnn = np.diagflat(obs['cov_nn'])
    Wuu = np.diagflat(obs['cov_uu'])
    We = np.hstack((Wee, Wen, Weu))
    Wn = np.hstack((Wen, Wnn, Wnu))
    Wu = np.hstack((Weu, Wnu, Wuu))
    W  = np.vstack((We, Wn, Wu))
    Q  = np.linalg.inv(W)


    Ge, Gn, Gu = get_SECS_B_G_matrices(obs['lat'], obs['lon'], np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, 
                                       grid.lat.flatten(), grid.lon.flatten(), 
                                       current_type = 'divergence_free', RI = RI)
    G = np.vstack((Ge, Gn, Gu))
    d = np.hstack((obs['Be'], obs['Bn'], obs['Bu']))

    GTQG = G.T.dot(Q).dot(G)
    GTQd = G.T.dot(Q).dot(d)
    scale = np.max(GTQG)
    R = np.eye(GTQG.shape[0]) * scale*1e0 + LL / np.abs(LL).max() * scale * 1e3 

    SS = np.linalg.inv(GTQG + R).dot(G.T.dot(Q))
    m = SS.dot(d).flatten()
    m = np.ravel(m)

    # set up G matrices for the magnetic field evaluated on the inner grid
    Gde0, Gdn0, Gdu0 = get_SECS_B_G_matrices(grid0.lat_mesh[1:, 1:].flatten(), grid0.lon_mesh[1:, 1:].flatten(), (6371.2 + OBSHEIGHT) * 1e3, 
                                             grid.lat.flatten(), grid.lon.flatten(), 
                                             current_type = 'divergence_free', RI = RI)


    # evaluate the magnetic field in the inner grid
    Bu0 = Gdu0.dot(m).flatten()

    difference = np.linalg.norm(Bu0 - Bu0_old)
    dts.append(DT)
    diffs.append(difference)
    print(DT, difference)
    if difference < 5:
        break




