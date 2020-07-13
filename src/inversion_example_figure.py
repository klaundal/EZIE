import numpy as np
import datetime as dt
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices, CSgrid, CSprojection
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dipole import geo2mag
from secsy import spherical 
from simulation_utils import get_MHD_jeq, get_MHD_dB
import pandas as pd


TRIM = True
COMPONENT = 'U' # component to plot ('N', 'U', or 'E')

#info = {'filename':'sam_data/ezie_simulation_background_information_for_kalle.sav',
#        'mapshift':-30,
#        'observation_height':80,
#        'output_path':'figs/',
#        'wshift':120}

info = {'filename':'sam_data/EZIE_event_simulation_ezie_simulation_case_1_look_direction_case_2_retrieved_los_mag_fields.pd',
        'mapshift':-210, # Sam has shifted the MHD output by this amount to get an orbit that crosses something interesting. The shift must be applied to my MHD readout functions
        'observation_height':80,
        'output_path':'final_figs/',
        'wshift':25}



OBSHEIGHT = info['observation_height']

d2r = np.pi / 180

LRES = 20. # spatial resolution of SECS grid along satellite track
WRES = 40. # spatial resolution perpendicular to satellite tarck
wshift = info['wshift'] # shift center of grid wshift km to the right of the satellite (rel to velocity)
DT  = 4 # size of time window [min]
RI  = (6371.2 + 110) * 1e3 # SECS height (m)

data = pd.read_pickle(info['filename'])

# convert all geographic coordinates and vector components in data to geomagnetic:
for i in range(4):
    i = i + 1
    data['lat_' + str(i)], data['lon_' + str(i)], data['dbe_' + str(i)], data['dbn_' + str(i)] = geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_' + str(i)].values, data['dbn_' + str(i)].values, epoch = 2020)
    _, _, data['dbe_measured_' + str(i)], data['dbn_measured_' + str(i)] = geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_measured_' + str(i)].values, data['dbn_measured_' + str(i)].values, epoch = 2020)
data['sat_lat'], data['sat_lon'] = geo2mag(data['sat_lat'].values, data['sat_lon'].values, epoch = 2020)

# calculate SC velocity
te, tn = spherical.tangent_vector(data['sat_lat'][:-1].values, data['sat_lon'][:-1].values,
                                  data['sat_lat'][1 :].values, data['sat_lon'][1: ].values)

data.loc[:-1, 've'] = te
data.loc[:-1, 'vn'] = tn


# get index of central point of analysis interval:
tm = data.index[1*len(data.index)//5:4*len(data.index)//5:2][39]

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
L = 400 + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km
W = 1200
print(L, W)

# set up the cubed sphere projection
v  = (data.loc[tm, 've'], data.loc[tm, 'vn'])
p = data.loc[tm, 'sat_lon'], data.loc[tm, 'sat_lat']
projection = CSprojection(p, v)
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
R = np.eye(GTQG.shape[0]) * scale*1e-1 + LL / np.abs(LL).max() * scale * 1e0 

SS = np.linalg.inv(GTQG + R).dot(G.T.dot(Q))
m = SS.dot(d)

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



for i in range(4):
    obs['lat'] += list(data.loc[t0:t1, 'lat_' + str(i + 1)].values)
    obs['lon'] += list(data.loc[t0:t1, 'lon_' + str(i + 1)].values)
    obs['Be' ] += list(data.loc[t0:t1, 'dbe_measured_'  + str(i + 1)].values)
    obs['Bn' ] += list(data.loc[t0:t1, 'dbn_measured_'  + str(i + 1)].values)
    obs['Bu' ] += list(data.loc[t0:t1, 'dbu_measured_'  + str(i + 1)].values)

    lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
    xi, eta = projection.geo2cube(lon, lat)
    for ax in [axe_secs, axn_secs, axr_secs]:
        ax.plot(xi, eta, color = 'C' + str(i), linewidth = 3)
    if i == 0:
        etamin, etamax = eta.min(), eta.max()
        ximin, ximax = xi.min(), xi.max()
    else:
        if eta.min() < etamin:
            etamin = eta.min()
        if eta.max() > etamax:
            etamax = eta.max()
        if xi.min() < ximin:
            ximin = xi.min()
        if xi.max() > ximax:
            ximax = xi.max()






Gde, Gdn, Gdu = get_SECS_B_G_matrices(grid.lat.flatten()+.1, grid.lon.flatten(), np.ones(grid.lon.size) * (6371.2 + OBSHEIGHT) * 1e3, 
                                      grid.lat.flatten(), grid.lon.flatten(), 
                                      current_type = 'divergence_free', RI = RI)

mhdBu = get_MHD_dB(grid.lat.flatten(), grid.lon.flatten() + info['mapshift'])
mhdBe = get_MHD_dB(grid.lat.flatten(), grid.lon.flatten() + info['mapshift'], component = 'Bphi [nT]')
mhdBn = -get_MHD_dB(grid.lat.flatten(), grid.lon.flatten() + info['mapshift'], component = 'Btheta [nT]')
mhdB = np.sqrt(mhdBe**2 + mhdBn**2 + mhdBu**2).reshape(grid.lat.shape)



B = np.sqrt(Gde.dot(m).reshape(grid.eta.shape)**2 + Gdn.dot(m).reshape(grid.eta.shape)**2 + Gdu.dot(m).reshape(grid.eta.shape)**2)



cntrs = axr_secs.contourf(grid.xi, grid.eta, Gdu.dot(m).reshape(grid.eta.shape), levels = np.linspace(-700, 700, 12), cmap = plt.cm.bwr, zorder = 0, extend = 'both')
axr_true.contourf(grid.xi, grid.eta, mhdBu.reshape(grid.eta.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

axe_secs.contourf(grid.xi, grid.eta, Gde.dot(m).reshape(grid.eta.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
axe_true.contourf(grid.xi, grid.eta, mhdBe.reshape(grid.eta.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

axn_secs.contourf(grid.xi, grid.eta, Gdn.dot(m).reshape(grid.eta.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
axn_true.contourf(grid.xi, grid.eta, mhdBn.reshape(grid.eta.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
ax_cbar.set_xlabel('nT')
ax_cbar.set_yticks([])

jlat = grid.lat_mesh[::5, ::5].flatten()
jlon = grid.lon_mesh[::5, ::5].flatten()
Gje, Gjn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
je, jn = Gje.dot(m).flatten(), Gjn.dot(m).flatten()
xi, eta, jxi, jeta = projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)

for ax in [axe_secs, axn_secs, axr_secs]:
    ax.quiver(xi, eta, jxi, jeta, linewidth = 2, scale = 1e10, zorder = 40, color = 'black')#, scale = 1e10)



mhd_je, mhd_jn = get_MHD_jeq(jlat, jlon + info['mapshift'])
xi, eta, mhd_jxi, mhd_jeta = projection.vector_cube_projection(mhd_je, mhd_jn, jlon, jlat)
#axmap.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 1, scale = 10, color = 'grey')#, scale = 1e10)
for ax in [axe_true, axn_true, axr_true]:#, axe_secs, axn_secs, axr_secs]:
    ax.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 2, scale = 10, color = 'grey', zorder = 38)#, scale = 1e10)



for ax in [axe_secs, axe_true, axn_secs, axn_true, axr_secs, axr_true]:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for l in np.r_[60:90:5]:
        xi, eta = projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
        ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

    for l in np.r_[0:360:15]:
        xi, eta = projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
        ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

    #ax.set_xlim(ximin - 25/(RI * 1e-3), ximax + 25/(RI * 1e-3))
    #ax.set_ylim(etamin, etamax)
    ax.axis('off')

    ax.set_adjustable('datalim') 
    ax.set_aspect('equal')



#axe_true.set_title('MHD simulation output')
#axe_secs.set_title('SECS inversion results')

for ax, label in zip([axe_secs, axe_true, axn_secs, axn_true, axr_secs, axr_true],
                     ['Be SECS', 'Be MHD', 'Bn SECS', 'Bn MHD', 'Br SECS', 'Br MHD']):
    
    ax.text(ximin- 25/(RI * 1e-3), etamax- 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)

for ax in [axe_secs, axe_true, axn_secs, axn_true, axr_secs, axr_true]:
    ax.set_xlim(ximin - 25/(RI * 1e-3), ximax + 25/(RI * 1e-3))
    ax.set_ylim(etamin + 55/(RI * 1e-3), etamax - 55/(RI * 1e-3))
    ax.set_adjustable('datalim') 
    ax.set_aspect('equal')
    #ax.set_xlim(*xlim)
    #ax.set_ylim(*ylim)

plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

#plt.savefig('./figures/inveresion_example_proposal.png', dpi = 250)
#plt.savefig('./figures/inveresion_example_proposal.pdf')


plt.show()
