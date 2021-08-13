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

obs = {'lat': [], 'lon': [], 'Be': [], 'Bn': [], 'Bu': [], 'cov_ee': [], 'cov_nn': [], 'cov_uu': [], 'cov_en': [], 'cov_eu': [], 'cov_nu': [], 'Bu2': [], 'lat2': [], 'lon2': [], 'cov_uu2': []}
for i in range(4):
    obs['lat'] += list(data.loc[t0:t1, 'lat_' + str(i + 1)].values)
    obs['lon'] += list(data.loc[t0:t1, 'lon_' + str(i + 1)].values)
    #obs['Be' ] += list(data.loc[t0:t1, 'dbe_measured_'  + str(i + 1)].values)
    #obs['Bn' ] += list(data.loc[t0:t1, 'dbn_measured_'  + str(i + 1)].values)
    obs['Bu' ] += list(data.loc[t0:t1, 'dbu_measured_'  + str(i + 1)].values)
    #obs['cov_ee'] += list(data.loc[t0:t1, 'cov_ee_' + str(i + 1)].values)
    #obs['cov_nn'] += list(data.loc[t0:t1, 'cov_nn_' + str(i + 1)].values)
    obs['cov_uu'] += list(data.loc[t0:t1, 'cov_uu_' + str(i + 1)].values)
    #obs['cov_en'] += list(data.loc[t0:t1, 'cov_en_' + str(i + 1)].values)
    #obs['cov_eu'] += list(data.loc[t0:t1, 'cov_eu_' + str(i + 1)].values)
    #obs['cov_nu'] += list(data.loc[t0:t1, 'cov_nu_' + str(i + 1)].values)

    obs['Bu2'    ] += list(data.loc[t0:t1, 'dbu_measured_'  + str(i + 1)].resample('10S').mean().values)
    obs['lat2'   ] += list(data.loc[t0:t1, 'lat_' + str(i + 1)].resample('10S').median().values)
    obs['lon2'   ] += list(data.loc[t0:t1, 'lon_' + str(i + 1)].resample('10S').median().values)
    obs['cov_uu2'] += list(data.loc[t0:t1, 'cov_uu_' + str(i + 1)].resample('10S').mean().values/
                           data.loc[t0:t1, 'cov_uu_' + str(i + 1)].resample('10S').count() )



# construct covariance matrix and invert it
#Wen = np.diagflat(obs['cov_en'])
#Weu = np.diagflat(obs['cov_eu'])
#Wnu = np.diagflat(obs['cov_nu'])
#Wee = np.diagflat(obs['cov_ee'])
#Wnn = np.diagflat(obs['cov_nn'])
Wuu = np.diagflat(obs['cov_uu'])
Wuu2 = np.diagflat(obs['cov_uu2'])
#We = np.hstack((Wee, Wen, Weu))
#Wn = np.hstack((Wen, Wnn, Wnu))
#Wu = np.hstack((Weu, Wnu, Wuu))
#W  = np.vstack((We, Wn, Wu))
#Q  = np.linalg.inv(W)
Q  = np.linalg.inv(Wuu)
Q2  = np.linalg.inv(Wuu2)

Ge, Gn, Gu = get_SECS_B_G_matrices(obs['lat'], obs['lon'], np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, 
                                   grid.lat.flatten(), grid.lon.flatten(), 
                                   current_type = 'divergence_free', RI = RI)

Ge2, Gn2, Gu2 = get_SECS_B_G_matrices(obs['lat2'], obs['lon2'], np.ones_like(obs['lat2']) * (6371.2 + OBSHEIGHT) * 1e3, 
                                      grid.lat.flatten(), grid.lon.flatten(), 
                                      current_type = 'divergence_free', RI = RI)


#G = np.vstack((Ge, Gn, Gu))
#d = np.hstack((obs['Be'], obs['Bn'], obs['Bu']))
G = Gu
d = obs['Bu']


GTQG = G.T.dot(Q).dot(G)
GTQd = G.T.dot(Q).dot(d)
scale = np.max(GTQG)
R = np.eye(GTQG.shape[0]) * scale*1e-1 + LL / np.abs(LL).max() * scale * 1e0 

SS = np.linalg.inv(GTQG + R).dot(G.T.dot(Q))
m = SS.dot(d)


G2 = Gu2
d2 = obs['Bu2']


GTQG = G2.T.dot(Q2).dot(G2)
GTQd = G2.T.dot(Q2).dot(d2)
scale = np.max(GTQG)
R = np.eye(GTQG.shape[0]) * scale*1e-1 / 5 + LL / np.abs(LL).max() * scale * 1e0  / 5

SS = np.linalg.inv(GTQG + R).dot(G2.T.dot(Q2))
m2 = SS.dot(d2)



#V_m = SS.dot(W).dot(SS.T) # model covariance
#RR = np.linalg.inv(GTQG + R).dot(GTQG + R) # model resolution matrix


fig = plt.figure(figsize = (14, 10))
axr_true      = plt.subplot2grid((2, 16), (0, 1 ), colspan = 5 )
axr_secs1     = plt.subplot2grid((2, 16), (0, 6 ), colspan = 5 )
axr_secs2     = plt.subplot2grid((2, 16), (0, 11), colspan = 5 )
axts          = plt.subplot2grid((2, 16), (1, 0) , colspan = 16)

ax_cbar = plt.subplot2grid((2, 16), (0, 0))



t0_long = data.index[data.index.get_loc(tm - dt.timedelta(seconds = DT//2 * 60), method = 'nearest')]
t1_long = data.index[data.index.get_loc(tm + dt.timedelta(seconds = DT//2 * 60), method = 'nearest')]


for i in range(4):
    obs['lat'] += list(data.loc[t0:t1, 'lat_' + str(i + 1)].values)
    obs['lon'] += list(data.loc[t0:t1, 'lon_' + str(i + 1)].values)
    obs['Be' ] += list(data.loc[t0:t1, 'dbe_measured_'  + str(i + 1)].values)
    obs['Bn' ] += list(data.loc[t0:t1, 'dbn_measured_'  + str(i + 1)].values)
    obs['Bu' ] += list(data.loc[t0:t1, 'dbu_measured_'  + str(i + 1)].values)


    mhdBu = get_MHD_dB(data.loc[:, 'lat_' + str(i + 1)].values, data.loc[:, 'lon_' + str(i + 1)].values + info['mapshift'])
    mhdBu = pd.Series(mhdBu, index = data.index)[t0_long:t1_long]

    d = mhdBu
    x = (d.index - d.index[0]).total_seconds().values/60
    axts.plot(x, d.values, color = 'C' + str(i), lw = 5)

    d = data.loc[t0_long:t1_long, 'dbu_measured_'  + str(i + 1)]
    x = (d.index - d.index[0]).total_seconds().values/60
    axts.plot(x, d.values, color = 'C' + str(i), lw = .4)
    d = data.loc[t0_long:t1_long, 'dbu_measured_'  + str(i + 1)].resample('10S').mean()
    x = (d.index - d.index[0]).total_seconds().values/60
    axts.plot(x, d.values, color = 'C' + str(i), lw = 2)



    lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
    xi, eta = projection.geo2cube(lon, lat)
    axr_secs1.scatter(xi, eta, c = 'C' + str(i), marker = 'o', s = 5)

    lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].resample('10S').median().values, data.loc[t0:t1, 'lat_' + str(i + 1)].resample('10S').median().values
    xi, eta = projection.geo2cube(lon, lat)
    axr_secs2.scatter(xi, eta, c = 'C' + str(i), marker = 'o', s = 5)
    

    #for ax in [axr_secs1, axr_secs2]:
    #    ax.plot(xi, eta, color = 'C' + str(i), linewidth = 3)
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



cntrs = axr_secs1.contourf(grid.xi, grid.eta, Gdu.dot(m).reshape(grid.eta.shape), levels = np.linspace(-700, 700, 12), cmap = plt.cm.bwr, zorder = 0, extend = 'both')
axr_secs2.contourf(grid.xi, grid.eta, Gdu.dot(m2).reshape(grid.eta.shape), levels = np.linspace(-700, 700, 12), cmap = plt.cm.bwr, zorder = 0, extend = 'both')
axr_true.contourf(grid.xi, grid.eta, mhdBu.reshape(grid.eta.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

ax_cbar.contourf(np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
ax_cbar.set_ylabel('nT')
ax_cbar.set_xticks([])

jlat = grid.lat_mesh[::3, ::3].flatten()
jlon = grid.lon_mesh[::3, ::3].flatten()    
Gje, Gjn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)

je, jn = Gje.dot(m).flatten(), Gjn.dot(m).flatten()
xi, eta, jxi, jeta = projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)
axr_secs1.quiver(xi, eta, jxi, jeta, linewidth = 2, scale = 1e10, zorder = 40, color = 'black')#, scale = 1e10)

je, jn = Gje.dot(m2).flatten(), Gjn.dot(m2).flatten()
xi, eta, jxi, jeta = projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)
axr_secs2.quiver(xi, eta, jxi, jeta, linewidth = 2, scale = 1e10, zorder = 40, color = 'black')#, scale = 1e10)


mhd_je, mhd_jn = get_MHD_jeq(jlat, jlon + info['mapshift'])
xi, eta, mhd_jxi, mhd_jeta = projection.vector_cube_projection(mhd_je, mhd_jn, jlon, jlat)
#axmap.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 1, scale = 10, color = 'grey')#, scale = 1e10)
for ax in [axr_true, axr_secs1, axr_secs2]:
    ax.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 2, scale = 10, color = 'grey', zorder = 38)#, scale = 1e10)



for ax in [axr_secs1, axr_secs2, axr_true]:
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


# plot grid in top left panel to show spatial dimensions:
xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                              np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
for i in range(xigrid.shape[0]):
    axr_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
for j in range(xigrid.shape[1]):
    axr_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)


axts.spines['right'].set_visible(False)
axts.spines['top'].set_visible(False)
axts.yaxis.set_ticks_position('left')
axts.xaxis.set_ticks_position('bottom')

axts.set_ylabel('nT')
axts.set_xlabel('Minutes since start of interval')

#axe_true.set_title('MHD simulation output')
#axe_secs.set_title('SECS inversion results')

for ax, label in zip([axr_secs1, axr_secs2, axr_true],
                     ['Br SECS 2sec', 'Br SECS 10sec', 'Br MHD']):
    
    ax.text(ximin- 25/(RI * 1e-3), etamax- 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)



for ax in [axr_secs1, axr_secs2, axr_true]:
    ax.set_xlim(ximin - 25/(RI * 1e-3), ximax + 25/(RI * 1e-3))
    ax.set_ylim(etamin + 55/(RI * 1e-3), etamax - 55/(RI * 1e-3))
    ax.set_adjustable('datalim') 
    ax.set_aspect('equal')
    #ax.set_xlim(*xlim)
    #ax.set_ylim(*ylim)

plt.subplots_adjust(top=0.965, bottom=0.05, left=0.06, right=0.99, hspace=0.2, wspace=0.2)

plt.savefig('./figures/inveresion_example_noise_analysis.png', dpi = 250)
plt.savefig('./figures/inveresion_example_noise_analysis.pdf')




plt.show()
