import numpy as np
import datetime as dt
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices, CSgrid, CSprojection
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dipole import geo2mag
from secsy import spherical 
from pysymmetry.visualization.polarsubplot import Polarsubplot
from simulation_utils import get_MHD_jeq, get_MHD_dB, get_MHD_dB_new
import pandas as pd
import cases
from importlib import reload
reload(cases) 
d2r = np.pi / 180


PLOT_ALL = True # False to use the time segment in case file, True to plot everything

info = cases.cases['case_1']


fig1, ax = plt.subplots(nrows = 1, ncols = 1)
pax = Polarsubplot(ax)


mlat, mlt = np.meshgrid(np.linspace(50, 90, 40*3), np.linspace(0, 24, 24*3))


Bu = info['mhdfunc'](mlat.flatten(), mlt.flatten() * 15 + 180, fn = info['mhd_B_fn'])
pax.contourf(mlat, mlt, Bu.reshape(mlat.shape), levels = np.linspace(-800, 800, 22), cmap = plt.cm.bwr)


data = pd.read_pickle(info['filename'])
data = data[data.lat_1 > 50]
#t0, t1 = data.index[0], data.index[-1]
tm = info['tm']

if PLOT_ALL:
    t0 = data.index[0]
    t1 = data.index[-1]
else:
    t0 = data.index[data.index.get_loc(tm - dt.timedelta(seconds = info['DT']//info['timeres'] * 60), method = 'nearest')]
    t1 = data.index[data.index.get_loc(tm + dt.timedelta(seconds = info['DT']//info['timeres'] * 60), method = 'nearest')]



# convert all geographic coordinates and vector components in data to geomagnetic:
for i in range(4):
    i = i + 1
    #_, _, data['dbe_measured_' + str(i)], data['dbn_measured_' + str(i)] = geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_measured_' + str(i)].values, data['dbn_measured_' + str(i)].values, epoch = 2020)
    data['lat_' + str(i)], data['lon_' + str(i)], data['dbe_' + str(i)], data['dbn_' + str(i)] = geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_' + str(i)].values, data['dbn_' + str(i)].values, epoch = 2020)
data['sat_lat'], data['sat_lon'] = geo2mag(data['sat_lat'].values, data['sat_lon'].values, epoch = 2020)


obs = {'lat': [], 'lon': [], 'Be': [], 'Bn': [], 'Bu': [], 'Be_true': [], 'Bn_true': [], 'Bu_true': []}
for i in range(4):
    obs['lat'] += list(data.loc[t0:t1, 'lat_' + str(i + 1)].values)
    obs['lon'] += list(data.loc[t0:t1, 'lon_' + str(i + 1)].values)
    obs['Be' ] += list(data.loc[t0:t1, 'dbe_measured_'  + str(i + 1)].values * info['signs'][0])
    obs['Bn' ] += list(data.loc[t0:t1, 'dbn_measured_'  + str(i + 1)].values * info['signs'][1])
    obs['Bu' ] += list(data.loc[t0:t1, 'dbu_measured_'  + str(i + 1)].values * info['signs'][2])
    obs['Be_true' ] += list(data.loc[t0:t1, 'dbe_'  + str(i + 1)].values * info['signs'][0])
    obs['Bn_true' ] += list(data.loc[t0:t1, 'dbn_'  + str(i + 1)].values * info['signs'][1])
    obs['Bu_true' ] += list(data.loc[t0:t1, 'dbu_'  + str(i + 1)].values * info['signs'][2])


lengths = [len(obs[key]) for key in obs.keys()]
assert np.allclose(lengths, lengths[0])

beam1, beam2, beam3, beam4 = {}, {}, {}, {}

coords = (np.array(obs['lat']), np.array(obs['lon']) + info['mapshift'])
mhdBu =  info['mhdfunc'](*coords, fn = info['mhd_B_fn'])
mhdBe =  info['mhdfunc'](*coords, component = 'Bphi [nT]', fn = info['mhd_B_fn'])
mhdBn = -info['mhdfunc'](*coords, component = 'Btheta [nT]', fn = info['mhd_B_fn'])

beam1['mhd_Be'], beam2['mhd_Be'], beam3['mhd_Be'], beam4['mhd_Be'] = np.split(mhdBe, 4)
beam1['mhd_Bn'], beam2['mhd_Bn'], beam3['mhd_Bn'], beam4['mhd_Bn'] = np.split(mhdBn, 4)
beam1['mhd_Bu'], beam2['mhd_Bu'], beam3['mhd_Bu'], beam4['mhd_Bu'] = np.split(mhdBu, 4)


for key in ['Be', 'Bn', 'Bu', 'Be_true', 'Bn_true', 'Bu_true', 'lat', 'lon']:
    beam1[key], beam2[key], beam3[key], beam4[key] = np.split(np.array(obs[key]), 4)


for beam in [beam1, beam2, beam3, beam4]:
    pax.plot(beam['lat'], (beam['lon'] + info['mapshift'] + 180)/15)

fig1.savefig('./figures/' + info['outputfn'] + '_measurement_tracks_.png', dpi = 250)


fig, axes = plt.subplots(nrows = 3, ncols = 4, figsize = (18, 9))
for i, beam in enumerate((beam1, beam2, beam3, beam4)):
    axes[0, i].plot(np.arange(beam['lat'].size), beam['Bn'], label = 'measured')
    axes[1, i].plot(np.arange(beam['lat'].size), beam['Bu'])
    axes[2, i].plot(np.arange(beam['lat'].size), beam['Be'])
    axes[0, i].plot(np.arange(beam['lat'].size), beam['mhd_Bn'], label = 'MHD')
    axes[1, i].plot(np.arange(beam['lat'].size), beam['mhd_Bu'])
    axes[2, i].plot(np.arange(beam['lat'].size), beam['mhd_Be'])
    axes[0, i].plot(np.arange(beam['lat'].size), beam['Bn_true'], label = 'True')
    axes[1, i].plot(np.arange(beam['lat'].size), beam['Bu_true'])
    axes[2, i].plot(np.arange(beam['lat'].size), beam['Be_true'])
    axes[0, i].set_title('Beam ' + str(i + 1) + ', Bn')
    axes[1, i].set_title('Beam ' + str(i + 1) + ', Bu')
    axes[2, i].set_title('Beam ' + str(i + 1) + ', Be')

    if i == 0:
        axes[0, i].legend(frameon = True)


fig.savefig('./figures/' + info['outputfn'] + '_mhd_osse_comparison_.png', dpi = 250)



"""
for beam in [beam1, beam2, beam3, beam4][-1:]:
    fig, axes = plt.subplots(nrows = 4)

    for shift in np.r_[0:360:15]:
        coords = (np.array(obs['lat']), np.array(obs['lon']) + shift)
        mhdBu =  info['mhdfunc'](*coords, fn = info['mhd_B_fn'])
        mhdBe =  info['mhdfunc'](*coords, component = 'Bphi [nT]', fn = info['mhd_B_fn'])
        mhdBn = -info['mhdfunc'](*coords, component = 'Btheta [nT]', fn = info['mhd_B_fn'])
    
        beam1['mhd_Be'], beam2['mhd_Be'], beam3['mhd_Be'], beam4['mhd_Be'] = np.split(mhdBe, 4)
        beam1['mhd_Bn'], beam2['mhd_Bn'], beam3['mhd_Bn'], beam4['mhd_Bn'] = np.split(mhdBn, 4)
        beam1['mhd_Bu'], beam2['mhd_Bu'], beam3['mhd_Bu'], beam4['mhd_Bu'] = np.split(mhdBu, 4)

        axes[0].plot(beam['mhd_Bn'])
        axes[1].plot(beam['mhd_Bu'])
        axes[2].plot(beam['mhd_Be'])


    axes[0].plot(beam['Bn_true'], color = 'black')
    axes[1].plot(beam['Bu_true'], color = 'black')
    axes[2].plot(beam['Be_true'], color = 'black')
"""

plt.show()