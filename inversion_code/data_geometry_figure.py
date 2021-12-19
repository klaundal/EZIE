import numpy as np
import datetime as dt
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices, CSgrid, CSprojection
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dipole import geo2mag
from secsy import spherical 
from simulation_utils import get_MHD_jeq, get_MHD_dB
import pandas as pd

mpl.rcParams['text.usetex'] = False


COMPONENT = 'U' # component to plot ('N', 'U', or 'E')

info = {'filename':'../data/proposal_stage_sam_data/ezie_simulation_background_information_for_kalle.sav',
        'mapshift':-30,
        'observation_height':80,
        'output_path':'figs/',
        'wshift':120}

info = {'filename':'../data/proposal_stage_sam_data/EZIE_event_simulation_ezie_simulation_case_1_look_direction_case_2_retrieved_los_mag_fields.pd',
        'mapshift':-210,
        'observation_height':80,
        'output_path':'final_figs/',
        'wshift':25}



SAMSHIFT = info['mapshift']
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

# calculate SC velocity and add 
te, tn = spherical.tangent_vector(data['sat_lat'][:-1].values, data['sat_lon'][:-1].values,
                                  data['sat_lat'][1 :].values, data['sat_lon'][1: ].values)

data.loc[:-1, 've'] = te
data.loc[:-1, 'vn'] = tn

# choose a time:
tm  = data.index[1*len(data.index)//5:4*len(data.index)//5:2][39]


t0 = data.index[data.index.get_loc(tm - dt.timedelta(seconds = DT//2 * 60), method = 'nearest')]
t1 = data.index[data.index.get_loc(tm + dt.timedelta(seconds = DT//2 * 60), method = 'nearest')]

rs = []
for t in [t0, tm, t1]:
    rs.append(np.array([np.cos(data.loc[t, 'sat_lat'] * d2r) * np.cos(data.loc[t, 'sat_lon'] * d2r),
                        np.cos(data.loc[t, 'sat_lat'] * d2r) * np.sin(data.loc[t, 'sat_lon'] * d2r),
                        np.sin(data.loc[t, 'sat_lat'] * d2r)]))

L = 400 + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km
W = 1200
print(L, W)

v  = (data.loc[tm, 've'], data.loc[tm, 'vn'])
p = data.loc[tm, 'sat_lon'], data.loc[tm, 'sat_lat']

v  = (data.loc[tm, 've'], data.loc[tm, 'vn'])
angle = np.arctan2(v[1], v[0]) / d2r 
p = data.loc[tm, 'sat_lon'], data.loc[tm, 'sat_lat']
projection = CSprojection(p, angle)

#projection = CSprojection(p, v)
grid = CSgrid(projection, L, W, LRES, WRES, wshift = wshift)


fig = plt.figure(figsize = (5, 20))
axgrid      = plt.subplot2grid((10, 1), (0, 0), rowspan = 4)#fig.add_subplot(411)
axe         = plt.subplot2grid((10, 1), (4, 0), rowspan = 2, sharex = axgrid)#fig.add_subplot(412, sharex = axgrid)
axn         = plt.subplot2grid((10, 1), (6, 0), rowspan = 2, sharex = axgrid)#fig.add_subplot(413, sharex = axgrid)
axu         = plt.subplot2grid((10, 1), (8, 0), rowspan = 2, sharex = axgrid)#fig.add_subplot(414, sharex = axgrid)


for b in grid.get_grid_boundaries(geocentric = False):
    axgrid.plot(b[0], b[1], color = 'black', linewidth = .4)


obs = {'lat': [], 'lon': [], 'Be': [], 'Bn': [], 'Bu': [], 'cov_ee': [], 'cov_nn': [], 'cov_uu': [], 'cov_en': [], 'cov_eu': [], 'cov_nu': []}
for i in range(4):
    obs['lat'] += list(data.loc[t0:t1, 'lat_' + str(i + 1)].values)
    obs['lon'] += list(data.loc[t0:t1, 'lon_' + str(i + 1)].values)
    obs['Be' ] += list(data.loc[t0:t1, 'dbe_measured_'  + str(i + 1)].values)
    obs['Bn' ] += list(data.loc[t0:t1, 'dbn_measured_'  + str(i + 1)].values)
    obs['Bu' ] += list(data.loc[t0:t1, 'dbu_measured_'  + str(i + 1)].values)

    lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
    xi, eta = projection.geo2cube(lon, lat)
    axgrid.plot(xi, eta, color = 'C' + str(i), linewidth = 5)
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

    axe.plot(xi, data.loc[t0:t1, 'dbe_' + str(i + 1)], color = 'C' + str(i), linewidth = 2)
    axn.plot(xi, data.loc[t0:t1, 'dbn_' + str(i + 1)], color = 'C' + str(i), linewidth = 2)
    axu.plot(xi, data.loc[t0:t1, 'dbu_' + str(i + 1)], color = 'C' + str(i), linewidth = 2)
    axe.scatter(xi, data.loc[t0:t1, 'dbe_measured_' + str(i + 1)], c = 'C' + str(i), marker = 'o', s = 3)
    axn.scatter(xi, data.loc[t0:t1, 'dbn_measured_' + str(i + 1)], c = 'C' + str(i), marker = 'o', s = 3)
    axu.scatter(xi, data.loc[t0:t1, 'dbu_measured_' + str(i + 1)], c = 'C' + str(i), marker = 'o', s = 3)

axe.hlines(0, ximin, ximax, color = 'black', linewidth = 1.5, linestyle = '-')
axn.hlines(0, ximin, ximax, color = 'black', linewidth = 1.5, linestyle = '-')
axu.hlines(0, ximin, ximax, color = 'black', linewidth = 1.5, linestyle = '-')



axe.plot([ximin, ximin], [0, 1000], color = 'black', linewidth = 1.5)
axn.plot([ximin, ximin], [0, 1000], color = 'black', linewidth = 1.5)
axu.plot([ximin, ximin], [0, 1000], color = 'black', linewidth = 1.5)
axe.plot([ximin, ximin - (ximax - ximin)/100], [1000, 1000], color = 'black', linewidth = 1.5)
axn.plot([ximin, ximin - (ximax - ximin)/100], [1000, 1000], color = 'black', linewidth = 1.5)
axu.plot([ximin, ximin - (ximax - ximin)/100], [1000, 1000], color = 'black', linewidth = 1.5)
axe.text(ximin - (ximax - ximin)/100, 1000, '1000 nT', ha = 'right', va = 'center')
axn.text(ximin - (ximax - ximin)/100, 1000, '1000 nT', ha = 'right', va = 'center')
axu.text(ximin - (ximax - ximin)/100, 1000, '1000 nT', ha = 'right', va = 'center')

axe.plot([ximin,   ximin - (ximax - ximin)/100], [0, 0], color = 'black', linewidth = 1.5)
axn.plot([ximin,   ximin - (ximax - ximin)/100], [0, 0], color = 'black', linewidth = 1.5)
axu.plot([ximin,   ximin - (ximax - ximin)/100], [0, 0], color = 'black', linewidth = 1.5)
axe.text( ximin - (ximax -  ximin)/100, 0, '0 nT', ha = 'right', va = 'center')
axn.text( ximin - (ximax -  ximin)/100, 0, '0 nT', ha = 'right', va = 'center')
axu.text( ximin - (ximax -  ximin)/100, 0, '0 nT', ha = 'right', va = 'center')

axe.text(ximin + (ximax - ximin)/20, 1000, '$B_{e}$', ha = 'left', va = 'center', size = 18, bbox = dict(facecolor='white', alpha=0.5))
axn.text(ximin + (ximax - ximin)/20, 1000, '$B_{n}$', ha = 'left', va = 'center', size = 18, bbox = dict(facecolor='white', alpha=0.5))
axu.text(ximin + (ximax - ximin)/20, 1000, '$B_{r}$', ha = 'left', va = 'center', size = 18, bbox = dict(facecolor='white', alpha=0.5))


axe.set_ylim(-1500, 1500)
axn.set_ylim(-1500, 1500)

# plot contours of contant geomagnetic latitude
xlim = axgrid.get_xlim()
ylim = axgrid.get_ylim()
for l in np.r_[60:90:5]:
    #glat, glon = geo2mag(np.ones(360)*l, np.linspace(0, 360, 360), epoch = 2020, inverse = True)
    #xi, eta = projection.geo2cube(glon, glat)
    xi, eta = projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
    axgrid.plot(xi, eta, 'k-', linewidth = .5)
for l in np.r_[0:360:15]:
    #glat, glon = geo2mag(np.ones(360)*l, np.linspace(0, 360, 360), epoch = 2020, inverse = True)
    #xi, eta = projection.geo2cube(glon, glat)
    xi, eta = projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
    axgrid.plot(xi, eta, 'k-', linewidth = .5)



axgrid.set_xlim(*xlim)
axgrid.set_ylim(*ylim)


axgrid.set_adjustable('datalim') 
axgrid.set_aspect('equal')
axgrid.set_xlabel('km')
axgrid.set_ylabel('km')

for ax in [axe, axn, axu, axgrid]:
    ax.axis('off')



km_min = -100/(RI * 1e-3)
km_max =  100/(RI * 1e-3)
axgrid.plot([grid.xi_mesh.min(), grid.xi_mesh.min()], [km_min, km_max], linewidth = 2, color = 'black')
axgrid.plot([grid.xi_mesh.min() + (ximax - ximin)/100, grid.xi_mesh.min() - (ximax - ximin)/100], [km_min, km_min], linewidth = 2, color = 'black')
axgrid.plot([grid.xi_mesh.min() + (ximax - ximin)/100, grid.xi_mesh.min() - (ximax - ximin)/100], [km_max, km_max], linewidth = 2, color = 'black')
axgrid.text(grid.xi_mesh.min() - (ximax - ximin)/100, 0, '200 km', va = 'center', ha = 'right')

xsteps = np.r_[ximin:ximax:200/RI * 1e3]
#axgrid.plot(xsteps, [grid.xi_mesh.min()]*len(xsteps), linewidth = 2, color = 'black')
#for x in xsteps:
#    axgrid.plot([x, x], [grid.xi_mesh.min() - (etamax - etamin)/100, grid.xi_mesh.min() + (etamax - etamin)/100], linewidth = 2, color = 'black')

axgrid.text(xsteps[0] + km_max, 0 - 30/RI*1e3, '200 km', va = 'top', ha = 'center', bbox = dict(facecolor='white', alpha=0.4, linewidth = 0))

for ax in [axe, axn, axu, axgrid]:
    ax.scatter(xsteps, [0]*len(xsteps), marker = '|', c = 'black', linewidth = 2, zorder = 500)


satxi, sateta = projection.geo2cube(data.loc[t0:t1, 'sat_lon'], data.loc[t0:t1, 'sat_lat'])
axgrid.plot(satxi, sateta, 'k-', zorder = 500)  
axgrid.scatter(satxi[-1], sateta[-1], marker = '>', color = 'black', s = 50, zorder = 100)
axgrid.text(satxi[-1], sateta[-1] - 50/(RI * 1e-3), 'Satellite direction', ha = 'center', va = 'top', bbox = dict(facecolor='white', alpha=0.9), size = 14)

axe.scatter(ximax, 0, marker = '>', c = 'black', s = 50, zorder = 50)
axn.scatter(ximax, 0, marker = '>', c = 'black', s = 50, zorder = 50)
axu.scatter(ximax, 0, marker = '>', c = 'black', s = 50, zorder = 50)



plt.tight_layout()
#plt.savefig('./figures/geometry.png', dpi = 250)
#plt.savefig('./figures/geometry.pdf')


plt.show()
