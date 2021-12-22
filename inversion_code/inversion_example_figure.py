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


path = os.path.dirname(__file__)

TRIM = True
COMPONENT = 'U' # component to plot ('N', 'U', or 'E')

# PROPOSAL STAGE OSSE
info = {'filename':path + '/../data/proposal_stage_sam_data/EZIE_event_simulation_ezie_simulation_case_1_look_direction_case_2_retrieved_los_mag_fields.pd',
        'mhd_B_fn':path + '/../data/proposal_stage_mhd_data/gamera_dBs_Jfull_80km_2430',
        'mapshift':-210, # Sam has shifted the MHD output by this amount to get an orbit that crosses something interesting. The shift must be applied to my MHD readout functions
        'observation_height':80,
        'output_path':'final_figs/',
        'wshift':-50,
        'tm':dt.datetime(2023, 7, 3, 2, 42, 22),
        'outputfn':'proposal_stage',
        'mhdfunc':get_MHD_dB,
        'clevels':np.linspace(-700, 700, 12)}

# OSSE CASE 1
"""
info = {'filename':path + '/../data/OSSE_new/case_1/EZIE_event_simulation_CASE1_standard_EZIE_retrieved_los_mag_fields2.pd',
        'mhd_B_fn':path + '/../data/OSSE_new/case_1/gamera_dBs_80km_2016-08-09T08_49_45.txt',
        'mapshift':180, # Sam has shifted the MHD output by this amount to get an orbit that crosses something interesting. The shift must be applied to my MHD readout functions
        'observation_height':80,
        'output_path':'final_figs/',
        'wshift':25,
        'tm':dt.datetime(2023, 7, 4, 5, 59, 51),
        'outputfn':'osse_case1',
        'mhdfunc':get_MHD_dB_new,
        'clevels':np.linspace(-300, 300, 12)}
"""



OBSHEIGHT = info['observation_height']

d2r = np.pi / 180

LRES = 40. # spatial resolution of SECS grid along satellite track
WRES = 20. # spatial resolution perpendicular to satellite tarck
wshift = info['wshift'] # shift center of grid wshift km to the right of the satellite (rel to velocity)
DT  = 4 # size of time window [min]
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
Le, Ln = grid.get_Le_Ln()
LL = Le.T.dot(Le) # matrix for calculation of eastward gradient - eastward in magnetic since all coords above have been converted to dipole coords

obs = {'lat': [], 'lon': [], 'Be': [], 'Bn': [], 'Bu': [], 'cov_ee': [], 'cov_nn': [], 'cov_uu': [], 'cov_en': [], 'cov_eu': [], 'cov_nu': []}
for i in range(4):
    obs['lat'] += list(data.loc[t0:t1, 'lat_' + str(i + 1)].values)
    obs['lon'] += list(data.loc[t0:t1, 'lon_' + str(i + 1)].values)
    obs['Be' ] += list(data.loc[t0:t1, 'dbe_measured_'  + str(i + 1)].values)
    obs['Bn' ] += list(data.loc[t0:t1, 'dbn_measured_'  + str(i + 1)].values)
    obs['Bu' ] += list(data.loc[t0:t1, 'dbu_measured_'  + str(i + 1)].values)
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
R = np.eye(GTQG.shape[0]) * scale*1e0 + LL / np.abs(LL).max() * scale * 1e1 

SS = np.linalg.inv(GTQG + R).dot(G.T.dot(Q))
m = SS.dot(d).flatten()
m = np.ravel(m)
#V_m = SS.dot(W).dot(SS.T) # model covariance
#RR = np.linalg.inv(GTQG + R).dot(GTQG + R) # model resolution matrix


fig = plt.figure(figsize = (6, 17))
axe_true      = plt.subplot2grid((16, 2), (0 , 0), rowspan = 5)
axe_secs      = plt.subplot2grid((16, 2), (0 , 1), rowspan = 5)
axn_true      = plt.subplot2grid((16, 2), (5 , 0), rowspan = 5)
axn_secs      = plt.subplot2grid((16, 2), (5 , 1), rowspan = 5)
axr_true      = plt.subplot2grid((16, 2), (10, 0), rowspan = 5)
axr_secs      = plt.subplot2grid((16, 2), (10, 1), rowspan = 5)

ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)


# plot the data tracks:
ximin, ximax, etamin, etamax = 0, 0, 0, 0 # plot limits
for i in range(4):

    lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
    xi, eta = projection.geo2cube(lon, lat)
    for ax in [axe_secs, axn_secs, axr_secs]:
        ax.plot(xi, eta, color = 'C' + str(i), linewidth = 3)
    else:
        if eta.min() < etamin:
            etamin = eta.min()
        if eta.max() > etamax:
            etamax = eta.max()
        if xi.min() < ximin:
            ximin = xi.min()
        if xi.max() > ximax:
            ximax = xi.max()


# set up G matrices for the magnetic field evaluated on a grid - for plotting maps
Gde, Gdn, Gdu = get_SECS_B_G_matrices(grid.lat.flatten()+.1, grid.lon.flatten(), np.ones(grid.lon.size) * (6371.2 + OBSHEIGHT) * 1e3, 
                                      grid.lat.flatten(), grid.lon.flatten(), 
                                      current_type = 'divergence_free', RI = RI)

# get maps of MHD magnetic fields:
mhdBu =  info['mhdfunc'](grid.lat.flatten(), grid.lon.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
mhdBe =  info['mhdfunc'](grid.lat.flatten(), grid.lon.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
mhdBn = -info['mhdfunc'](grid.lat.flatten(), grid.lon.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])

# plot magnetic field in upward direction (MHD and retrieved)
cntrs = axr_secs.contourf(grid.xi, grid.eta, Gdu.dot(m).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
axr_true.contourf(grid.xi, grid.eta, mhdBu.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

# plot magnetic field in eastward direction (MHD and retrieved)
axe_secs.contourf(grid.xi, grid.eta, Gde.dot(m).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
axe_true.contourf(grid.xi, grid.eta, mhdBe.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

# plot magnetic field in northward direction (MHD and retrieved)
axn_secs.contourf(grid.xi, grid.eta, Gdn.dot(m).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
axn_true.contourf(grid.xi, grid.eta, mhdBn.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

# plot colorbar:
ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
ax_cbar.set_xlabel('nT')
ax_cbar.set_yticks([])


# calculate the equivalent current of retrieved magnetic field:
jlat = grid.lat_mesh[::2, ::2].flatten()
jlon = grid.lon_mesh[::2, ::2].flatten()    
Gje, Gjn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
je, jn = Gje.dot(m).flatten(), Gjn.dot(m).flatten()
xi, eta, jxi, jeta = projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)

# plot the equivalent current in the SECS panels:
for ax in [axe_secs, axn_secs, axr_secs]:
    ax.quiver(xi, eta, jxi, jeta, linewidth = 2, scale = 1e10, zorder = 40, color = 'black')#, scale = 1e10)

# get the MHD equivalent current field:
mhd_je, mhd_jn = get_MHD_jeq(jlat, jlon + info['mapshift'])
xi, eta, mhd_jxi, mhd_jeta = projection.vector_cube_projection(mhd_je, mhd_jn, jlon, jlat)

# plot the MHD equivalent current in eaach panel
for ax in [axe_true, axn_true, axr_true , axe_secs, axn_secs, axr_secs]:
    ax.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 2, scale = 10, color = 'grey', zorder = 38)#, scale = 1e10)


# plot coordinate grids, fix aspect ratio and axes in each panel
for ax in [axe_secs, axe_true, axn_secs, axn_true, axr_secs, axr_true]:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for l in np.r_[60:90:5]:
        xi, eta = projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
        ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

    for l in np.r_[0:360:15]:
        xi, eta = projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
        ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

    ax.axis('off')

    ax.set_adjustable('datalim') 
    ax.set_aspect('equal')


# Write labels:
for ax, label in zip([axe_secs, axe_true, axn_secs, axn_true, axr_secs, axr_true],
                     ['Be SECS', 'Be MHD', 'Bn SECS', 'Bn MHD', 'Br SECS', 'Br MHD']):
    
    ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)


# plot grid in top left panel to show spatial dimensions:
xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                              np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
for i in range(xigrid.shape[0]):
    axe_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
for j in range(xigrid.shape[1]):
    axe_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)

# set plot limits:
for ax in [axe_secs, axe_true, axn_secs, axn_true, axr_secs, axr_true]:
    ax.set_xlim(ximin - 25/(RI * 1e-3), ximax + 25/(RI * 1e-3))
    ax.set_ylim(etamin + 55/(RI * 1e-3), etamax - 55/(RI * 1e-3))
    ax.set_adjustable('datalim') 
    ax.set_aspect('equal')

# remove whitespace
plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

# save plots:
plt.savefig('./figures/' + info['outputfn'] + 'inversion_example.png', dpi = 250)
plt.savefig('./figures/' + info['outputfn'] + 'inversion_example.pdf')


# save the relevant parts of the datafile for publication
columns = ['lat_1', 'lon_1', 'dbe_1', 'dbn_1', 'dbu_1', 'lat_2', 'lon_2', 'dbe_2', 'dbn_2', 'dbu_2', 'lat_3', 'lon_3', 'dbe_3', 'dbn_3', 'dbu_3', 'lat_4', 'lon_4', 'dbe_4', 'dbn_4', 'dbu_4', 'sat_lat', 'sat_lon', 'dbe_measured_1', 'dbn_measured_1', 'dbu_measured_1', 'cov_ee_1', 'cov_nn_1', 'cov_uu_1', 'cov_en_1', 'cov_eu_1', 'cov_nu_1', 'dbe_measured_2', 'dbn_measured_2', 'dbu_measured_2', 'cov_ee_2', 'cov_nn_2', 'cov_uu_2', 'cov_en_2', 'cov_eu_2', 'cov_nu_2', 'dbe_measured_3', 'dbn_measured_3', 'dbu_measured_3', 'cov_ee_3', 'cov_nn_3', 'cov_uu_3', 'cov_en_3', 'cov_eu_3', 'cov_nu_3', 'dbe_measured_4', 'dbn_measured_4', 'dbu_measured_4', 'cov_ee_4', 'cov_nn_4', 'cov_uu_4', 'cov_en_4', 'cov_eu_4', 'cov_nu_4']
savedata = data[t0:t1][columns]
savedata.index = dt = (savedata.index-savedata.index[0]).seconds
savedata.index.name = 'seconds'
savedata.to_csv(info['outputfn'] + 'electrojet_inversion_data.csv')


plt.show()
