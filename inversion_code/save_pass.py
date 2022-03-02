""" run through a whole pass, and save the output in xarray format
"""

import numpy as np
import datetime as dt
import xarray as xr
from datetime import timedelta as td
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices, CSgrid, CSprojection
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dipole import geo2mag
from secsy import spherical 
from simulation_utils import get_MHD_jeq, get_MHD_dB, get_MHD_dB_new
import pandas as pd
import os
import cases
import new_cases
from importlib import reload
reload(cases) 

timestep = 30 # time resolution of maps - also the length of the central portion
window_size = 330 # window size in seconds

for case in new_cases.cases.keys():
    info = new_cases.cases[case]

    OBSHEIGHT = info['observation_height']
    wshift = info['wshift']

    d2r = np.pi / 180

    LRES = 40. # spatial resolution of SECS grid along satellite track
    WRES = 20. # spatial resolution perpendicular to satellite tarck
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

    # start and end:
    T0 = data.index[0] + td(seconds = window_size//2)
    T_end = data.index[-1] - td(seconds = window_size//2)


    datasets = []
    for counter in range( int((T_end - T0).total_seconds() // timestep)):
        tm = T0 + counter * td(seconds = timestep)

        # find the closest time which is in the index:
        tm = data.index[data.index.get_loc(tm, method = 'nearest')]

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
        W = 2*WRES + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km
        L = 1500

        grid0 = CSgrid(projection, (L // LRES) * LRES, (W // WRES // 2 * 2) * WRES, LRES, WRES, wshift = wshift) # make sure that the width is an even multiple of WRES
        Bu0 = np.zeros(grid0.size)

        print(tm, L, W, grid0.shape)


        diffs = []
        dts = []


        Bu0_old = Bu0

        # limits of analysis interval:
        t0 = data.index[data.index.get_loc(tm - dt.timedelta(seconds = window_size//2), method = 'nearest')]
        t1 = data.index[data.index.get_loc(tm + dt.timedelta(seconds = window_size//2), method = 'nearest')]

        # get unit vectors pointing at satellite (Cartesian vectors)
        rs = []
        for t in [t0, tm, t1]:
            rs.append(np.array([np.cos(data.loc[t, 'sat_lat'] * d2r) * np.cos(data.loc[t, 'sat_lon'] * d2r),
                                np.cos(data.loc[t, 'sat_lat'] * d2r) * np.sin(data.loc[t, 'sat_lon'] * d2r),
                                np.sin(data.loc[t, 'sat_lat'] * d2r)]))

        # dimensions of analysis region/grid (in km)
        W = 200 + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km
        L = 1500


        grid = CSgrid(projection, (L // LRES) * LRES, (W // WRES // 2 * 2) * WRES, LRES, WRES, wshift = wshift) # make sure that the width is an even multiple of WRES
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

        # set up G matrices for the magnetic field evaluated on the inner grid:
        Gde0, Gdn0, Gdu0 = get_SECS_B_G_matrices(grid0.lat_mesh.flatten(), grid0.lon_mesh.flatten(), (6371.2 + OBSHEIGHT) * 1e3, 
                                                 grid.lat.flatten(), grid.lon.flatten(), 
                                                 current_type = 'divergence_free', RI = RI)

        Be, Bn, Bu = Gde0.dot(m).reshape(grid0.lat_mesh.shape), Gdn0.dot(m).reshape(grid0.lat_mesh.shape), Gdu0.dot(m).reshape(grid0.lat_mesh.shape)

        # set up G matrices for the divergence-free currents:
        Gje0, Gjn0 = get_SECS_J_G_matrices(grid0.lat_mesh.flatten(), grid0.lon_mesh.flatten(),
                                                 grid.lat.flatten(), grid.lon.flatten(), 
                                                 current_type = 'divergence_free', RI = RI)

        je, jn = Gje0.dot(m).reshape(grid0.lat_mesh.shape) * 1e-6, Gjn0.dot(m).reshape(grid0.lat_mesh.shape) * 1e-6


        # Make xarray DataArrays:
        Be = xr.DataArray(data=Be, dims=['eta', 'xi'], coords={'xi': grid0.xi_mesh[0], 'eta': grid0.eta_mesh[:, 0], 'time':tm}, attrs={'summary': 'EZIE OSSE mapped magnetic field - eastward [nT]'})
        Bn = xr.DataArray(data=Bn, dims=['eta', 'xi'], coords={'xi': grid0.xi_mesh[0], 'eta': grid0.eta_mesh[:, 0], 'time':tm}, attrs={'summary': 'EZIE OSSE mapped magnetic field - northward [nT]'})
        Bu = xr.DataArray(data=Bu, dims=['eta', 'xi'], coords={'xi': grid0.xi_mesh[0], 'eta': grid0.eta_mesh[:, 0], 'time':tm}, attrs={'summary': 'EZIE OSSE mapped magnetic field - upward [nT]'})

        je = xr.DataArray(data=je, dims=['eta', 'xi'], coords={'xi': grid0.xi_mesh[0], 'eta': grid0.eta_mesh[:, 0], 'time':tm}, attrs={'summary': 'EZIE OSSE divergence-free current - eastward [mA/m]'})
        jn = xr.DataArray(data=jn, dims=['eta', 'xi'], coords={'xi': grid0.xi_mesh[0], 'eta': grid0.eta_mesh[:, 0], 'time':tm}, attrs={'summary': 'EZIE OSSE divergence-free current - northward [mA/m]'})

        lat = xr.DataArray(data=grid0.lat_mesh, dims=['eta', 'xi'], coords={'xi': grid0.xi_mesh[0], 'eta': grid0.eta_mesh[:, 0], 'time':tm}, attrs={'summary': 'latitude [degrees]'})
        lon = xr.DataArray(data=grid0.lon_mesh, dims=['eta', 'xi'], coords={'xi': grid0.xi_mesh[0], 'eta': grid0.eta_mesh[:, 0], 'time':tm}, attrs={'summary': 'longitude [degrees]'})

        # Make xarray Dataset:
        datasets.append( xr.Dataset({'Be':Be, 'Bn':Bn, 'Bu':Bu, 'je':je, 'jn':jn, 'lat':lat, 'lon':lon}) )

    # merge the datasets and save:
    ds = xr.concat(datasets, dim = 'time')
    ds.to_netcdf(info["outputfn"] + ".netcdf")

    print('saved ' + info["outputfn"] + ".netcdf")



    # plot like this:
    #pax.contourf( np.stack(ds.lat.values, axis=2), np.stack(ds.lon.values, axis=2) / 15 + 12, np.stack(ds.Bu.values, axis=2), cmap=plt.cm.bwr, levels=np.linspace(-600, 600, 22))



