""" script to make fake data for use in ezie_Bscalar """
import cases
import numpy as np
import pandas as pd
from dipole import Dipole
from secsy import spherical 
import datetime as dt
import ppigrf
d2r = np.pi / 180
EPOCH = dt.datetime(2025, 6, 1) # IGRF magnetic field epoch (IGRF changes very slowly so this is not important to get right)

dpl = Dipole(epoch = 2025) # initialize Dipole object

info = cases.cases['case_2']
timeres = info['timeres'] # time resolution of the data [sec]
DT = info['DT'] # time window in minutes
OBSHEIGHT = 80 

data = pd.read_pickle(info['filename'])

#%% convert all geographic coordinates and vector components in data to geomagnetic:
for i in range(4):
    i = i + 1
    data['lat_' + str(i)], data['lon_' + str(i)], data['dbe_' + str(i)], data['dbn_' + str(i)] = dpl.geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_' + str(i)].values, data['dbn_' + str(i)].values)
data['sat_lat'], data['sat_lon'] = dpl.geo2mag(data['sat_lat'].values, data['sat_lon'].values)

#%% Timespan and satellite velocity

sc_lat, sc_lon = data['sat_lat'].values, data['sat_lon'].values
# calculate SC velocity
te, tn = spherical.tangent_vector(sc_lat[:-1], sc_lon[:-1],
                                  sc_lat[1 :], sc_lon[1: ])

data['ve'] = np.hstack((te, np.nan))
data['vn'] = np.hstack((tn, np.nan))

# get index of central point of analysis interval:
tm = data.index[data.index.get_loc(info['tm'])]

# spacecraft velocity at central time:
v = np.array((data.loc[tm, 've'], data.loc[tm, 'vn']))

# spacecraft lat and lon at central time:
sc_lat0 = data.loc[tm, 'sat_lat'] 
sc_lon0 = data.loc[tm, 'sat_lon']

# limits of analysis interval:
t0 = data.index[data.index.get_loc(tm - dt.timedelta(seconds = DT/2*60))]
t1 = data.index[data.index.get_loc(tm + dt.timedelta(seconds = DT/2*60))]

# get unit vectors pointing at satellite (Cartesian vectors)
rs = []
for t in [t0, tm, t1]:
    rs.append(np.array([np.cos(data.loc[t, 'sat_lat'] * d2r) * np.cos(data.loc[t, 'sat_lon'] * d2r),
                        np.cos(data.loc[t, 'sat_lat'] * d2r) * np.sin(data.loc[t, 'sat_lon'] * d2r),
                        np.sin(data.loc[t, 'sat_lat'] * d2r)]))

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

obs_lon = obs['lon']
obs_lat = obs['lat']


Be0, Bn0, Bu0 = map(np.ravel, ppigrf.igrf(obs_lon, obs_lat, OBSHEIGHT, EPOCH))
Be = Be0 + obs['Be']
Bn = Bn0 + obs['Bn']
Bu = Bu0 + obs['Bu']
obs_d  = np.sqrt(Be**2 + Bn**2 + Bu**2)
obs_err = np.sqrt(obs['cov_ee']**2 + obs['cov_nn']**2 + obs['cov_uu']**2)
