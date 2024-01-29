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
    # Bullshit
#t0 = data.index[data.index.get_loc(tm - dt.timedelta(seconds = DT//timeres * 60), method = 'nearest')]
#t1 = data.index[data.index.get_loc(tm + dt.timedelta(seconds = DT//timeres * 60), method = 'nearest')]
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
gini_load = ''
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

inv_result = ezl.standard_retrieval(obs, t0, sc_lon0, sc_lat0, v[0], v[1], 
                                    map_params, get_f1, reg_params,
                                    PD = [],
                                    plot_dir='/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure', 
                                    plot_name='ezl')

inv_result = ezl.standard_retrieval(obs, t0, sc_lon0, sc_lat0, v[0], v[1], 
                                    map_params, get_f1, reg_params,
                                    PD = [PD_xi, PD_eta],
                                    plot_dir='/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_code/EZIE/figure', 
                                    plot_name='ezl_PD')

