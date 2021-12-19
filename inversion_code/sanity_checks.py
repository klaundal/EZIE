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
import os
d2r = np.pi / 180

path = os.path.dirname(__file__)

TRIM = True
COMPONENT = 'U' # component to plot ('N', 'U', or 'E')

# PROPOSAL STAGE OSSE
info = {'filename':path + '/../data/proposal_stage_sam_data/EZIE_event_simulation_ezie_simulation_case_1_look_direction_case_2_retrieved_los_mag_fields.pd',
        'mhd_B_fn':path + '/../data/proposal_stage_mhd_data/gamera_dBs_Jfull_80km_2430',
        'mapshift':-210, # Sam has shifted the MHD output by this amount to get an orbit that crosses something interesting. The shift must be applied to my MHD readout functions
        'observation_height':80,
        'output_path':'final_figs/',
        'wshift':25,
        'tm':dt.datetime(2023, 7, 3, 2, 42, 22),
        'outputfn':'proposal_stage',
        'mhdfunc':get_MHD_dB,
        'clevels':np.linspace(-700, 700, 21)}

# OSSE CASE 1
"""
info = {'filename':path + '/../data/OSSE_new/case_1/EZIE_event_simulation_CASE1_standard_EZIE_retrieved_los_mag_fields2.pd',
        'mhd_B_fn':path + '/../data/OSSE_new/case_1/gamera_dBs_80km_2016-08-09T08_49_45.txt',
        'mapshift':0, # Sam has shifted the MHD output by this amount to get an orbit that crosses something interesting. The shift must be applied to my MHD readout functions
        'observation_height':80,
        'output_path':'final_figs/',
        'wshift':25,
        'tm':dt.datetime(2023, 7, 4, 5, 59, 51),
        'outputfn':'osse_case1',
        'mhdfunc':get_MHD_dB_new,
        'clevels':np.linspace(-300, 300, 12)}
"""


pax = Polarsubplot(plt.subplots(nrows = 1, ncols = 1)[1])


mlat, mlt = np.meshgrid(np.linspace(50, 90, 40*3), np.linspace(0, 24, 24*3))


Bu = info['mhdfunc'](mlat.flatten(), mlt.flatten() * 15 + 180, fn = info['mhd_B_fn'])
pax.contourf(mlat, mlt, Bu, levels = np.linspace(-800, 800, 22), cmap = plt.cm.bwr)



data = pd.read_pickle(info['filename'])
t0, t1 = data.index[0], data.index[-1]

obs = {'lat': [], 'lon': [], 'Be': [], 'Bn': [], 'Bu': [], 'cov_ee': [], 'cov_nn': [], 'cov_uu': [], 'cov_en': [], 'cov_eu': [], 'cov_nu': [], 'Be_true': [], 'Bn_true': [], 'Bu_true': []}
for i in range(4):
    obs['lat'] += list(data.loc[t0:t1, 'lat_' + str(i + 1)].values)
    obs['lon'] += list(data.loc[t0:t1, 'lon_' + str(i + 1)].values)
    obs['Be' ] += list(data.loc[t0:t1, 'dbe_measured_'  + str(i + 1)].values)
    obs['Bn' ] += list(data.loc[t0:t1, 'dbn_measured_'  + str(i + 1)].values)
    obs['Bu' ] += list(data.loc[t0:t1, 'dbu_measured_'  + str(i + 1)].values)
    obs['Be_true' ] += list(data.loc[t0:t1, 'dbe_'  + str(i + 1)].values)
    obs['Bn_true' ] += list(data.loc[t0:t1, 'dbn_'  + str(i + 1)].values)
    obs['Bu_true' ] += list(data.loc[t0:t1, 'dbu_'  + str(i + 1)].values)
    obs['cov_ee'] += list(data.loc[t0:t1, 'cov_ee_' + str(i + 1)].values)
    obs['cov_nn'] += list(data.loc[t0:t1, 'cov_nn_' + str(i + 1)].values)
    obs['cov_uu'] += list(data.loc[t0:t1, 'cov_uu_' + str(i + 1)].values)
    obs['cov_en'] += list(data.loc[t0:t1, 'cov_en_' + str(i + 1)].values)
    obs['cov_eu'] += list(data.loc[t0:t1, 'cov_eu_' + str(i + 1)].values)
    obs['cov_nu'] += list(data.loc[t0:t1, 'cov_nu_' + str(i + 1)].values)


beam1, beam2, beam3, beam4 = {}, {}, {}, {}

for key in ['Be', 'Bn', 'Bu', 'Be_true', 'Bn_true', 'Bu_true', 'lat', 'lon']:
    beam1[key], beam2[key], beam3[key], beam4[key] = np.split(np.array(obs[key]), 4)

coords = (np.array(obs['lat']), (np.array(obs['lon']) + info['mapshift']) % 360)
mhdBu =  info['mhdfunc'](*coords, fn = info['mhd_B_fn'])
mhdBe =  info['mhdfunc'](*coords, component = 'Bphi [nT]', fn = info['mhd_B_fn'])
mhdBn = -info['mhdfunc'](*coords, component = 'Btheta [nT]', fn = info['mhd_B_fn'])

beam1['mhd_Be'], beam2['mhd_Be'], beam3['mhd_Be'], beam4['mhd_Be'] = np.split(mhdBe, 4)
beam1['mhd_Bn'], beam2['mhd_Bn'], beam3['mhd_Bn'], beam4['mhd_Bn'] = np.split(mhdBn, 4)
beam1['mhd_Bu'], beam2['mhd_Bu'], beam3['mhd_Bu'], beam4['mhd_Bu'] = np.split(mhdBu, 4)

for beam in [beam1, beam2, beam3, beam4]:
    fig, axes = plt.subplots(nrows = 4)
    axes[0].plot(beam['Bn'])
    axes[1].plot(beam['Bu'])
    axes[2].plot(beam['Be'])
    axes[0].plot(beam['mhd_Bn'])
    axes[1].plot(beam['mhd_Bu'])
    axes[2].plot(beam['mhd_Be'])
    axes[0].plot(beam['Bn_true'])
    axes[1].plot(beam['Bu_true'])
    axes[2].plot(beam['Be_true'])

plt.show()