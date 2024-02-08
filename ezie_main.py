#%% Import
import numpy as np
import pandas as pd
from importlib import reload
import matplotlib.ticker as ticker
import datetime as dt
import pickle
import copy
import matplotlib.pyplot as plt
import datetime as dt

from secsy import spherical 
from dipole import Dipole # https://github.com/klaundal/dipole
from apexpy import Apex
import pyamps

import os
os.chdir('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE')
import cases
import ezie_lib as ezl

#%% Defining some stuff

RE = 6371.2*1e3
d2r = np.pi / 180
RI = RE + 110e3 # radius of the ionosphere
dpl = Dipole(epoch = 2020) # initialize Dipole object

#%% Select case and load data from OSSE

reload(cases) 

# load parameters and data file names
#info = cases.cases['case_2']
info = cases.cases['case_2']
#info['tm'] = dt.datetime(2023, 7, 4, 12, 22, 56) # case 3
timeres = info['timeres'] # time resolution of the data [sec]
DT = info['DT'] # time window in minutes
OBSHEIGHT = info['observation_height'] * 1e3 # observation height in m

data = pd.read_pickle(info['filename'])

#%% convert all geographic coordinates and vector components in data to geomagnetic:
for i in range(4):
    i = i + 1
    data['lat_' + str(i)], data['lon_' + str(i)], data['dbe_' + str(i)], data['dbn_' + str(i)] = dpl.geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_' + str(i)].values, data['dbn_' + str(i)].values)
data['sat_lat'], data['sat_lon'] = dpl.geo2mag(data['sat_lat'].values, data['sat_lon'].values)

#%% Timespan and satellite velocity

# calculate SC velocity
te, tn = spherical.tangent_vector(data['sat_lat'][:-1].values, data['sat_lon'][:-1].values,
                                  data['sat_lat'][1 :].values, data['sat_lon'][1: ].values)

data['ve'] = np.hstack((te, np.nan))
data['vn'] = np.hstack((tn, np.nan))

# get index of central point of analysis interval:
tm = data.index[data.index.get_loc(info['tm'], method = 'nearest')]

# spacecraft velocity at central time:
v = np.array((data.loc[tm, 've'], data.loc[tm, 'vn']))

# spacecraft lat and lon at central time:
sc_lat0 = data.loc[tm, 'sat_lat'] 
sc_lon0 = data.loc[tm, 'sat_lon']

# limits of analysis interval:
t0 = data.index[data.index.get_loc(tm - dt.timedelta(seconds = DT/2*60), method = 'nearest')]
t1 = data.index[data.index.get_loc(tm + dt.timedelta(seconds = DT/2*60), method = 'nearest')]

# get unit vectors pointing at satellite (Cartesian vectors)
rs = []
for t in [t0, tm, t1]:
    rs.append(np.array([np.cos(data.loc[t, 'sat_lat'] * d2r) * np.cos(data.loc[t, 'sat_lon'] * d2r),
                        np.cos(data.loc[t, 'sat_lat'] * d2r) * np.sin(data.loc[t, 'sat_lon'] * d2r),
                        np.sin(data.loc[t, 'sat_lat'] * d2r)]))

#%% Define map paramters

# dimensions of analysis region/grid (in km)
W = 700 + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km

map_params = {'LRES':40.,
              'WRES':20.,
              'W': W, # along-track dimension of analysis grid (TODO: This is more a proxy than a precise description)
              'L': 2000, # cross-track dimension of analysis grid (TODO: Same as above)
              'wshift':25, # shift the grid center wres km in cross-track direction
              'total_time_window':6*60,
              'strip_time_window':30,
              'RI':RI, # height of the ionosphere [m]
              'RE':RE,
              'Rez':RE+OBSHEIGHT
              }

#%% Grab data from selected time

obs = {'lat': [], 'lon': [], 
       'Be': [], 'Bn': [], 'Bu': [], 
       'cov_ee': [], 'cov_nn': [], 'cov_uu': [], 'cov_en': [], 'cov_eu': [], 'cov_nu': [], 
       'times':[],
       'lat_1': [], 'lat_2': [], 'lat_3': [], 'lat_4': [], 
       'lon_1': [], 'lon_2': [], 'lon_3': [], 'lon_4': []}
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
    obs['times']  += list(data[t0:t1].index)
    
    # for plotting tracks
    obs['lat_' + str(i + 1)] = list(data.loc[t0:t1, 'lat_' + str(i + 1)].values)
    obs['lon_' + str(i + 1)] = list(data.loc[t0:t1, 'lon_' + str(i + 1)].values)

for key in obs.keys():
    obs[key] = np.array(obs[key])

#%%

"""
Make a function to calculate the magnetic eastward direction in the coordinate system of the other input data. In this case, since the input 
data is in dipole coordinates, the eastward direction is just (1, 0) for all points. In a realistic scenario, we would use input data in 
geographic coordinates and call for example apexpy's basevectors_qd function
"""
get_f1 = lambda lon, lat: np.vstack((np.ones(lon.size), np.zeros(lon.size)))

#%% Lambda relation - find the spot

# Tool for visualizing the location of a model parameter

grid = ezl.get_grid(sc_lon0, sc_lat0, v[0], v[1], map_params)


m_id = 3570 # case 3
m_id = 3570+27*48 # case 2

ezl.lambda_relation_plot_spot(obs, sc_lon0, sc_lat0, v[0], v[1], 
                              map_params, m_id=m_id,
                              plot_dir='/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure', plot_name='test')

print(grid.lat[m_id//grid.shape[1], m_id%grid.shape[1]])

#%% Lambda relation

# Stuff
gini_load = '/scratch//BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/gini_case_2.pkl'
gini_save = '/scratch//BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/gini_case_2.pkl'
plot_summary = True

# Select regularization parameter range
l1s = 10**np.linspace(-1.5, 1.5, 20)
l2s = 10**np.linspace(-1, 5, 20)

# Calculate the Hoyer index for selectec regularization parameters
if len(gini_load) != 0:
    gini = pickle.load(open(gini_load, 'rb'))
else:    
    gini = ezl.lambda_relation_get_gini(obs, t0, sc_lon0, sc_lat0, v[0], v[1], 
                                         map_params, get_f1, l1s, l2s)
    if len(gini_save) != 0:
        pickle.dump(gini, open(gini_save, 'wb'))

#%% Lambda relation - Plot summary

# Get gini for the parameter we are interested in.
#gini_id = gini[m_id, :, :] / np.sum(gini, axis=0)
gini_id = gini[m_id, :, :]

# Determine optimal l2 for every l1
ps = 3 # points selected
l2_opt_id = ezl.lambda_relation_find_ridge(copy.deepcopy(gini_id), ps=ps)
l2_opt = l2s[l2_opt_id]

# Compute a spline fit
steps = 200
xx = np.tile(l1s, (ps, 1)).T.flatten()
yy = l2_opt.flatten()
l2s_fit_l10, tck = ezl.make_spline_fit(np.log10(xx), np.log10(yy), s=5, steps=steps)
l2s_fit = 10**l2s_fit_l10
l1s_fit_l10 = np.linspace(np.log10(l1s[0]), np.log10(l1s[-1]), steps)
l1s_fit = 10**l1s_fit_l10

# Summary plot - Hoyer
if plot_summary:
    # Plot Hoyer
    vmin = np.min(gini_id)
    vmax = np.max(gini_id)
    plt.ioff()
    plt.figure(figsize=(10, 10))
    y, x = np.meshgrid(np.log10(l2s), np.log10(l1s))
    plt.tricontourf(x.flatten(), y.flatten(), gini_id.flatten(), levels=np.linspace(vmin, vmax, 40), cmap='magma')

    plt.xlabel('log10 l1')
    plt.ylabel('log10 l2')

    for i in range(gini_id.shape[0]):
        if np.any(gini_id[i, :] < 0):
            plt.plot(np.ones(l2_opt_id.shape[1])*np.log10(l1s)[i], np.log10(l2s)[l2_opt_id[i, :]], '.', color='tab:red', markersize=8)
        plt.plot(np.ones(l2_opt_id.shape[1])*np.log10(l1s)[i], np.log10(l2s)[l2_opt_id[i, :]], '.', color='tab:blue', markersize=8)

    plt.plot(l1s_fit_l10, l2s_fit_l10, color='k', linewidth=2)

    plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure/ezl_gini.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure/ezl_gini.png', bbox_inches='tight')
    plt.close('all')
    plt.ion()

#%% Lambda relation - Lcurve

# Iteratively solve inverse problem to retrieve residual and model norm.
inv_results = ezl.iterative_retrieval(obs, t0, sc_lon0, sc_lat0, v[0], v[1], 
                                      map_params, get_f1, 
                                      l1s=l1s_fit, l2s=l2s_fit, 
                                      full=False, Lcurve=True)

# Retrieve norms from inversion results
mnorm = np.zeros(len(inv_results))
dnorm = np.zeros(len(inv_results))
for i, inv_result in enumerate(inv_results):
    mnorm[i] = inv_result['mnorm_l1'] + inv_result['mnorm_l2']
    dnorm[i] = inv_result['dnorm']

# Find knee of L-curve
kn_id, skip_id = ezl.robust_Kneedle(np.log10(dnorm), np.log10(mnorm))

# Get optimal parameters
l1_opt = l1s_fit[kn_id]
l2_opt = l2s_fit[kn_id]

# Summary plot - L-curve
if plot_summary:
    plt.ioff()
    plt.figure(figsize=(10, 10))

    plt.loglog(dnorm, mnorm, '.-', color='k')
    plt.loglog(dnorm[kn_id], mnorm[kn_id], '*', markersize=15, color='tab:red')

    plt.xlabel('residual norm')
    plt.ylabel('model norm')

    plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure/ezl_lcurve.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure/ezl_lcurve.png', bbox_inches='tight')
    plt.close('all')
    plt.ion()

#%% Set reguarlizaition parameters

reg_params = {'lambda1': l1_opt,
              'lambda2': l2_opt}

#%% Extrapolation - Prediction Domain (PD)

# ID : Interpolation Domain
# SED : Safe Extrapolation Domain

ID, SED, grid = ezl.get_ID_and_SED(obs, t0, sc_lon0, sc_lat0, v[0], v[1], map_params, 
                                   get_f1, reg_params)

# Make the prediction quality flag
quality_flag = copy.deepcopy(SED)
quality_flag[ID == 1] = 2

# Determine the largest rectangle
PD_xi, PD_eta = ezl.get_largest_rectangle(SED, grid)

plt.ioff()
plt.figure(figsize=(10, 10))
ax = plt.gca()
cc = ezl.plot_map(ax, grid.xi.flatten(), grid.eta.flatten(), 
                  1/(quality_flag.flatten()+1), obs, grid, RI, 
                  'Pastel2', 3, 'Data quality flag')
plt.plot([PD_xi[0], PD_xi[0], PD_xi[1], PD_xi[1], PD_xi[0]], 
         [PD_eta[0], PD_eta[1], PD_eta[1], PD_eta[0], PD_eta[0]],
         color='w', linewidth=5)
plt.plot([PD_xi[0], PD_xi[0], PD_xi[1], PD_xi[1], PD_xi[0]], 
         [PD_eta[0], PD_eta[1], PD_eta[1], PD_eta[0], PD_eta[0]],
         color='k', linewidth=3)
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure/ezl_PD_and_flag.pdf', format='pdf', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure/ezl_PD_and_flag.png', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Run with model variance and resolution

inv_result, grid = ezl.standard_retrieval(obs, t0, sc_lon0, sc_lat0, v[0], v[1], 
                                          map_params, get_f1, reg_params,
                                          PD = [],
                                          plot_dir='/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure', 
                                          plot_name='ezl')

inv_result, grid = ezl.standard_retrieval(obs, t0, sc_lon0, sc_lat0, v[0], v[1], 
                                          map_params, get_f1, reg_params,
                                          PD = [PD_xi, PD_eta],
                                          plot_dir='/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure', 
                                          plot_name='ezl_PD')

#%% Data product - Solution (from which all else can be derived)

data_product = {}

# Model parameters (SECS amplitudes)
data_product['m'] = inv_result['m']

# Coordinates of the SECS pole locations - in geocentric coordinates (!)
data_product['geoclat_secs'] = grid.lat
data_product['geoclon_secs'] = grid.lon
data_product['radius_secs'] = np.ones(grid.shape)*map_params['RI']

# Posterior model covariance matrix
data_product['Cpm'] = inv_result['Cpm']

# Spatial resolution in cross-track and along-track directions
xi_FWHM, eta_FWHM, xi_flag, eta_flag = ezl.get_resolution(inv_result['R'], grid)
ef = (xi_flag==1) & (eta_flag==1)
data_product['xi_FWHM'] = xi_FWHM # cross-track resolution
data_product['eta_FWHM'] = eta_FWHM # along-track resolution

# Quality flags (0, 1, 2)
data_product['quality_flag'] = quality_flag

# Central time an size of time window
data_product['central_time'] = tm

# Config parameters: Regularization, ionosphere radius, code version
data_product['regularization'] = reg_params
data_product['map_parameters'] = map_params

#%% Data product - Gridded quantities (calculated at offset grid)

# Electrojet / equivalent current components [mA/m] in geocentric (!) east and north directions
Ge, Gn = ezl.get_SECS_J_G_matrices(grid.lat_mesh, grid.lon_mesh, 
                                   grid.lat.flatten(), grid.lon.flatten(), 
                                   current_type='divergence_free', RI=map_params['RI'])

data_product['Je'] = Ge.dot(inv_result['m'])
data_product['Jn'] = Gn.dot(inv_result['m'])
data_product['Je_std'] = np.sqrt(np.diag(Ge.dot(data_product['Cpm']).dot(Ge.T)))
data_product['Jn_std'] = np.sqrt(np.diag(Gn.dot(data_product['Cpm']).dot(Gn.T)))
data_product['geoclat_J'] = grid.lat_mesh.flatten() # geocentric latitude
data_product['lon_J'] = grid.lon_mesh.flatten() # geocentric longitude (the same as geodetic longitude)
data_product['r_J'] = np.ones(grid.xi_mesh.size)*map_params['RI'] # Radius of the ionospheric shell

# Be, Bn, Bu [nT] at h=80 km - geodetic coordinates
#       Convert from geodetic to geocentric 
#       We need the magnetic field at a constant height over the ellipsoid, not constant radius from the center
theta, r, _, _ = ezl.geod2geoc(grid.lat_mesh.flatten(), 
                               np.ones(grid.xi_mesh.size)*info['observation_height'], 
                               np.ones(grid.xi_mesh.size), 
                               np.ones(grid.xi_mesh.size))
geoclat_80 = 90 - theta # 
radius_80 = r*1e3

#       Calculate magnetic field in geocentric
Ge, Gn, Gu = ezl.get_SECS_B_G_matrices(geoclat_80, grid.lon_mesh.flatten(), radius_80, 
                                       grid.lat.flatten(), grid.lon.flatten(), current_type='divergence_free', RI=map_params['RI'])

Be_geoc_80 = Ge.dot(inv_result['m'])
Bn_geoc_80 = Gn.dot(inv_result['m'])
Btheta_geoc_80 = -Bn_geoc_80
Bu_geoc_80 = Gu.dot(inv_result['m'])

#       Convert from geocentric to geodetic
gdlat, height, Bn, Bu = ezl.geoc2geod(theta, r, Btheta_geoc_80, Bu_geoc_80)

#       Calculate magnetic field model variance in geocentric
Be_geoc_80_sig = np.sqrt(np.diag(Ge.dot(data_product['Cpm']).dot(Ge.T)))
Bn_geoc_80_sig = np.sqrt(np.diag(Gn.dot(data_product['Cpm']).dot(Gn.T)))
Bu_geoc_80_sig = np.sqrt(np.diag(Gu.dot(data_product['Cpm']).dot(Gu.T)))

#       Convert from geocentric to geodetic
#       Rotate individual 3D vector covariance matrices from and grab sqrt of the diagonal
Be_geod_80_sig = np.zeros(Ge.shape[0])
Bn_geod_80_sig = np.zeros(Ge.shape[0])
Bu_geod_80_sig = np.zeros(Ge.shape[0])
B_Cpm_geod_80 = np.zeros((3, 3, Ge.shape[0]))
for i in range(Ge.shape[0]):
    G_i = np.vstack((Ge[i, :], Gn[i, :], Gu[i, :]))
    B_Cpm_geoc_i = G_i.dot(data_product['Cpm']).dot(G_i.T)
    _, _, T = ezl.geoc2geod(theta[i], r[i], [], [], matrix=True)
    B_Cpm_geod_i = T.dot(B_Cpm_geoc_i).dot(T.T)
    B_Cpm_geod_80[:, :, i] = B_Cpm_geod_i
    Be_geod_80_sig[i] = np.sqrt(B_Cpm_geod_i[0, 0])
    Bn_geod_80_sig[i] = np.sqrt(B_Cpm_geod_i[1, 1])
    Bu_geod_80_sig[i] = np.sqrt(B_Cpm_geod_i[2, 2])

data_product['geodlat_80'] = gdlat
data_product['lon_80'] = grid.lon_mesh.flatten()
data_product['height_80'] = np.ones(gdlat.shape)*8e4 # meters above ellipsoide
data_product['Be_geod_80'] = Be_geoc_80
data_product['Bn_geod_80'] = Bn
data_product['Bu_geod_80'] = Bu
data_product['Be_geod_std_80'] = Be_geod_80_sig
data_product['Bn_geod_std_80'] = Bn_geod_80_sig
data_product['Bu_geod_std_80'] = Bu_geod_80_sig


# Be, Bn, Bu [nT] at h=0 km - geodetic coordinates
#       Convert from geodetic to geocentric 
theta, r, _, _ = ezl.geod2geoc(grid.lat_mesh.flatten(), 
                               np.ones(grid.xi_mesh.size)*0, 
                               np.ones(grid.xi_mesh.size), 
                               np.ones(grid.xi_mesh.size))
geoclat_0 = 90 - theta # 
radius_0 = r*1e3

#       Calculate magnetic field in geocentric
Ge, Gn, Gu = ezl.get_SECS_B_G_matrices(geoclat_0, grid.lon_mesh.flatten(), radius_0, 
                                       grid.lat.flatten(), grid.lon.flatten(), current_type='divergence_free', RI=map_params['RI'])

Be_geoc_0 = Ge.dot(inv_result['m'])
Bn_geoc_0 = Gn.dot(inv_result['m'])
Btheta_geoc_0 = -Bn_geoc_0
Bu_geoc_0 = Gu.dot(inv_result['m'])

#       Convert from geocentric to geodetic
gdlat, height, Bn, Bu = ezl.geoc2geod(theta, r, Btheta_geoc_0, Bu_geoc_0)

#       Calculate magnetic field model variance in geocentric
Be_geoc_0_sig = np.sqrt(np.diag(Ge.dot(data_product['Cpm']).dot(Ge.T)))
Bn_geoc_0_sig = np.sqrt(np.diag(Gn.dot(data_product['Cpm']).dot(Gn.T)))
Bu_geoc_0_sig = np.sqrt(np.diag(Gu.dot(data_product['Cpm']).dot(Gu.T)))

#       Convert from geocentric to geodetic
#       Rotate individual 3D vector covariance matrices from and grab sqrt of the diagonal
Be_geod_0_sig = np.zeros(Ge.shape[0])
Bn_geod_0_sig = np.zeros(Ge.shape[0])
Bu_geod_0_sig = np.zeros(Ge.shape[0])
B_Cpm_geod_0 = np.zeros((3, 3, Ge.shape[0]))
for i in range(Ge.shape[0]):
    G_i = np.vstack((Ge[i, :], Gn[i, :], Gu[i, :]))
    B_Cpm_geoc_i = G_i.dot(data_product['Cpm']).dot(G_i.T)
    _, _, T = ezl.geoc2geod(theta[i], r[i], [], [], matrix=True)
    B_Cpm_geod_i = T.dot(B_Cpm_geoc_i).dot(T.T)
    B_Cpm_geod_0[:, :, i] = B_Cpm_geod_i
    Be_geod_0_sig[i] = np.sqrt(B_Cpm_geod_i[0, 0])
    Bn_geod_0_sig[i] = np.sqrt(B_Cpm_geod_i[1, 1])
    Bu_geod_0_sig[i] = np.sqrt(B_Cpm_geod_i[2, 2])

data_product['geodlat_0'] = gdlat
data_product['lon_0'] = grid.lon_mesh.flatten()
data_product['height_0'] = np.ones(gdlat.shape)*0 # meters above ellipsoide
data_product['Be_geod_0'] = Be_geoc_0
data_product['Bn_geod_0'] = Bn
data_product['Bu_geod_0'] = Bu
data_product['Be_geod_std_0'] = Be_geod_0_sig
data_product['Bn_geod_std_0'] = Bn_geod_0_sig
data_product['Bu_geod_std_0'] = Bu_geod_0_sig


# The same parameters in QD coordinates @ 80 km
#       Initiate the apex object with the date at the center measurement
date = dt.datetime(tm.year, tm.month, tm.day, tm.hour, tm.minute)
apex = Apex(date)

#       Calculate the QD basevector for h=80 km
f1, f2 = apex.basevectors_qd(data_product['geodlat_80'], grid.lon_mesh.flatten(), 
                             np.ones(grid.xi_mesh.size)*info['observation_height'])

#       Columnwise cross-product
F = f1[0, :] * f2[1, :] - f1[1, :] * f2[0, :]

#       Transform B from geodetic to quasi-dipole (QD)
Be_qd_80 = (f1[0, :]*data_product['Be_geod_80'] + f1[1, :]*data_product['Bn_geod_80'])/F
Bn_qd_80 = (f2[0, :]*data_product['Be_geod_80'] + f2[1, :]*data_product['Bn_geod_80'])/F
Br_qd_80 = data_product['Bu_geod_80']/np.sqrt(F)
data_product['Be_QD_80'] = Be_qd_80
data_product['Bn_QD_80'] = Bn_qd_80
data_product['Bu_QD_80'] = Br_qd_80

#       Transform B sigma from geodetic to quasi-dipole (QD)
Be_qd_80_sig = np.zeros(Ge.shape[0])
Bn_qd_80_sig = np.zeros(Ge.shape[0])
Bu_qd_80_sig = np.zeros(Ge.shape[0])
for i in range(f1.shape[1]):
    T = np.array([[f1[0, i]/F[i], f1[1, i]/F[i], 0],
                  [f2[0, i]/F[i], f2[1, i]/F[i], 0],
                  [0, 0, 1/np.sqrt(F[i])]])
    B_Cpm_qd_i = T.dot(B_Cpm_geod_80[:, :, i]).dot(T.T)
    Be_qd_80_sig[i] = np.sqrt(B_Cpm_qd_i[0, 0])
    Bn_qd_80_sig[i] = np.sqrt(B_Cpm_qd_i[1, 1])
    Bu_qd_80_sig[i] = np.sqrt(B_Cpm_qd_i[2, 2])

data_product['Be_QD_std_80'] = Be_qd_80_sig
data_product['Bn_QD_std_80'] = Bn_qd_80_sig
data_product['Bu_QD_std_80'] = Bu_qd_80_sig

#       Transform coords from geodetic to quasi-dipole (QD)
QDlat_80, QDlon_80 = apex.geo2qd(data_product['geodlat_80'], grid.lon_mesh.flatten(), 
                                 np.ones(grid.xi_mesh.size)*info['observation_height'])
data_product['QD_lat_80'] = QDlat_80
data_product['QD_lon_80'] = QDlon_80

# The same parameters in QD coordinates @ 0 km
#       Calculate the QD basevector for h=0 km
f1, f2 = apex.basevectors_qd(data_product['geodlat_0'], grid.lon_mesh.flatten(), 
                             np.ones(grid.xi_mesh.size)*0)

#       Columnwise cross-product
F = f1[0, :] * f2[1, :] - f1[1, :] * f2[0, :]

#       Transform from geodetic to quasi-dipole (QD)
Be_qd_0 = (f1[0, :]*data_product['Be_geod_0'] + f1[1, :]*data_product['Bn_geod_0'])/F
Bn_qd_0 = (f2[0, :]*data_product['Be_geod_0'] + f2[1, :]*data_product['Bn_geod_0'])/F
Br_qd_0 = data_product['Bu_geod_0']/np.sqrt(F)
data_product['Be_QD_0'] = Be_qd_0
data_product['Bn_QD_0'] = Bn_qd_0
data_product['Bu_QD_0'] = Br_qd_0

#       Transform B sigma from geodetic to quasi-dipole (QD)
Be_qd_0_sig = np.zeros(Ge.shape[0])
Bn_qd_0_sig = np.zeros(Ge.shape[0])
Bu_qd_0_sig = np.zeros(Ge.shape[0])
for i in range(f1.shape[1]):
    T = np.array([[f1[0, i]/F[i], f1[1, i]/F[i], 0],
                  [f2[0, i]/F[i], f2[1, i]/F[i], 0],
                  [0, 0, 1/np.sqrt(F[i])]])
    B_Cpm_qd_i = T.dot(B_Cpm_geod_0[:, :, i]).dot(T.T)
    Be_qd_0_sig[i] = np.sqrt(B_Cpm_qd_i[0, 0])
    Bn_qd_0_sig[i] = np.sqrt(B_Cpm_qd_i[1, 1])
    Bu_qd_0_sig[i] = np.sqrt(B_Cpm_qd_i[2, 2])

data_product['Be_QD_std_0'] = Be_qd_0_sig
data_product['Bn_QD_std_0'] = Bn_qd_0_sig
data_product['Bu_QD_std_0'] = Bu_qd_0_sig

#       Transform coords from geodetic to quasi-dipole (QD)
QDlat_0, QDlon_0 = apex.geo2qd(data_product['geodlat_0'], grid.lon_mesh.flatten(), 
                               np.ones(grid.xi_mesh.size)*0)
data_product['QD_lat_0'] = QDlat_0
data_product['QD_lon_0'] = QDlon_0

# Calculate the magnetic local time (MLT) for each data set
data_product['QD_MLT_0'] = pyamps.mlon_to_mlt(data_product['QD_lon_0'], date, date.year)
data_product['QD_MLT_80'] = pyamps.mlon_to_mlt(data_product['QD_lon_80'], date, date.year)

#%% Magnetic field predictions at L2 data locations [nT], geodetic coordinates + input

# Input (satellite measurements)
data_product['B_input'] = inv_result['d']

# B predictions (Output from model)
data_product['B_predictions'] = inv_result['d_pred']

# coordinates
data_product['geodlat'] = obs['lat']
data_product['geodlon'] = obs['lon']
data_product['input_height'] = np.ones(len(obs['lat']))*info['observation_height']*1e3

#%% Plots

# Br, Btheta, Bphi (geocentric!) at r=RE + 80 km and equivalent current at r=RI 
    # Produced in using the standard_retrieval() function


# Br, Btheta, Bphi (geocentric!) at r=RE km and equivalent current at r=RI 
    # standard_retrieval() has been updated to also produce this plot


# Spatial resolution maps in cross-track and along-track directions
    # Produced in using the standard_retrieval() function


# Plot of model variance projected into data space
    # Produced in using the standard_retrieval() function


# Flag map
    # Produced in the 'Extrapolation - Prediction Domain (PD)' section of this script


# Time series of fitted magnetic field with confidence intervals at measurement location + input data
    # Produced in the following section 'Observations prediction comparison'
    # The dots are the measurements made by the satellite along with a 90% confidence interval (1.645*standard deviation)
    # The line along with the shaded area around it is the model fit and a 90% confidence interval.


#%% Observations prediction comparison

n = int(obs['lat'].size/4)

B_obs = inv_result['d'].reshape((3, 4, n)) # 3 components of B, 4 Beams, n observations
B_pred = inv_result['d_pred'].reshape((3, 4, n))

sB_pred = np.zeros(B_pred.shape)
sB_pred[0, :, :] = np.sqrt(np.diag(inv_result['G'][0].dot(inv_result['Cpm']).dot(inv_result['G'][0].T))).reshape((4, n))
sB_pred[1, :, :] = np.sqrt(np.diag(inv_result['G'][1].dot(inv_result['Cpm']).dot(inv_result['G'][1].T))).reshape((4, n))
sB_pred[2, :, :] = np.sqrt(np.diag(inv_result['G'][2].dot(inv_result['Cpm']).dot(inv_result['G'][2].T))).reshape((4, n))

sB_obs = np.zeros(B_obs.shape)
sB_obs[0, :, :] = np.sqrt(obs['cov_ee']).reshape(4, n)
sB_obs[1, :, :] = np.sqrt(obs['cov_nn']).reshape(4, n)
sB_obs[2, :, :] = np.sqrt(obs['cov_uu']).reshape(4, n)

plt.ioff()
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
for i, ax in enumerate(axs):
    for j, c in enumerate(['tab:blue', 'tab:orange', 'tab:green', 'tab:red']):
        ax.plot(B_obs[i, j, :], '.', markersize=5, color=c)
        ax.errorbar(x = range(n), y = B_obs[i, j, :], yerr=1.645*sB_obs[i, j, :], color=c, linestyle='None', linewidth=0.8)
        ax.fill_between(range(n), B_pred[i, j, :] - 1.645*sB_pred[i, j, :], B_pred[i, j, :] + 1.645*sB_pred[i, j, :], color=c, alpha=0.4)
        ax.plot(B_pred[i, j, :], linewidth=2, color=c, label='Beam {}'.format(j+1))
    ax.set_ylabel('nT')

axs[2].legend()
axs[0].text(1.05, 0.5, 'Be', va='center', ha='center', transform=axs[0].transAxes, fontsize=20)
axs[1].text(1.05, 0.5, 'Bn', va='center', ha='center', transform=axs[1].transAxes, fontsize=20)
axs[2].text(1.05, 0.5, 'Bu', va='center', ha='center', transform=axs[2].transAxes, fontsize=20)


plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure/data_prediction_comparison.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure/data_prediction_comparison.pdf', format='pdf', bbox_inches='tight')

axs[0].set_ylim([-500, 800])
axs[1].set_ylim([-750, 1000])

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure/data_prediction_comparison_zoom.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure/data_prediction_comparison_zoom.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()


