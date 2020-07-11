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

info = {'filename':'sam_data/ezie_simulation_background_information_for_kalle.sav',
        'mapshift':-30,
        'observation_height':80,
        'output_path':'figs/',
        'wshift':120}

info = {'filename':'sam_data/EZIE_event_simulation_ezie_simulation_case_1_look_direction_case_2_retrieved_los_mag_fields.pd',
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


# get index of central point + limits:

counter = 0
for tm in data.index[1*len(data.index)//5:4*len(data.index)//5:2][39:]:


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
    projection = CSprojection(p, v)
    grid = CSgrid(projection, L, W, LRES, WRES, wshift = wshift)
    Le, Ln = grid.get_Le_Ln()
    LL = Le.T.dot(Le)

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
    Q  = Q / np.abs(Q).max()


    Ge, Gn, Gu = get_SECS_B_G_matrices(obs['lat'], obs['lon'], np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, 
                                       grid.lat.flatten(), grid.lon.flatten(), 
                                       current_type = 'divergence_free', RI = RI)
    G = np.vstack((Ge, Gn, Gu))
    d = np.hstack((obs['Be'], obs['Bn'], obs['Bu']))


    GTQG = G.T.dot(Q).dot(G)
    GTQd = G.T.dot(Q).dot(d)
    S = np.max(GTQG)
    #R = np.eye(GTQG.shape[0]) * S*1e-1 + LL / np.abs(LL).max() * S * 1e1
    R = np.eye(GTQG.shape[0]) * S*1e-1 + LL / np.abs(LL).max() * S * 1e0
    m = np.linalg.lstsq(GTQG + R, GTQd, rcond = 0)[0]


    fig = plt.figure(figsize = (14, 8))
    axmap       = fig.add_subplot(132)
    axmap_true  = fig.add_subplot(131, sharey = axmap)
    axB         = fig.add_subplot(133, sharey = axmap)
    axB.yaxis.tick_right()
    if not TRIM:
        for b in grid.get_grid_boundaries(geocentric = False):
            axmap.plot(b[0], b[1], color = 'lightgrey', linewidth = .2)


    BSTEP = 1000
    #data['num'] = np.arange(len(data))
    for i in range(4):

        lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
        xi, eta = projection.geo2cube(lon, lat)
        axmap.plot(xi, eta, color = 'C' + str(i), linewidth = 2)
        axmap_true.plot(xi, eta, color = 'C' + str(i), linewidth = 2)
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


        Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, lat * 0 + (6371.2 + OBSHEIGHT) * 1e3, 
                                           grid.lat.flatten(), grid.lon.flatten(), 
                                           current_type = 'divergence_free', RI = RI)

        #axB.plot(Gu.dot(m).flatten() + i * BSTEP, eta, color = 'C' + str(i), linestyle = '--')    
        if COMPONENT in ['E', 'N', 'U']:
            axB.plot(data.loc[t0:t1, 'db' + COMPONENT.lower() + '_' + str(i + 1)] + i * BSTEP, eta, color = 'C' + str(i))
            axB.plot(data.loc[t0:t1, 'db' + COMPONENT.lower() + '_measured_' + str(i + 1)] + i * BSTEP, eta, color = 'C' + str(i), linewidth = .1, marker = 'o', markersize = 2)
        else:            
            axB.plot(data.loc[t0:t1, 'dbu_' + str(i + 1)] + i * BSTEP, eta, color = 'C' + str(i))
            axB.plot(data.loc[t0:t1, 'dbu_m0easured_' + str(i + 1)] + i * BSTEP, eta, color = 'C' + str(i), linewidth = .1, marker = 'o', markersize = 2)
        axB.plot([i * BSTEP, i * BSTEP], eta[[0, -1]], color = 'C' + str(i), linewidth = .4)

        #mhdB = get_MHD_dB(lat, lon + SAMSHIFT)
        #axB.plot(mhdB + i * BSTEP, eta, color = 'C' + str(i), linestyle = ':')

    axB.plot([0, BSTEP], eta[[0, 0]] - 1e5/RI, 'k-')
    axB.text(BSTEP/2., eta[0] - 1.2e5/RI, str(BSTEP) + ' nT', ha = 'center', va = 'top')

    axB.spines['left'].set_visible(False)
    axB.spines['top'].set_visible(False)
    axB.spines['bottom'].set_visible(False)
    axB.yaxis.set_ticks_position('right')
    axB.xaxis.set_ticks_position('none')
    axB.get_xaxis().set_visible(False)


    #xi, eta = projection.geo2cube(obs['lon'], obs['lat'])
    #axmap.scatter(xi, eta, c = obs['Bu'], vmin = -500, vmax = 500, cmap = plt.cm.bwr)


    Gde, Gdn, Gdu = get_SECS_B_G_matrices(grid.lat.flatten()+.1, grid.lon.flatten(), np.ones(grid.lon.size) * (6371.2 + OBSHEIGHT) * 1e3, 
                                          grid.lat.flatten(), grid.lon.flatten(), 
                                          current_type = 'divergence_free', RI = RI)

    mhdBu = get_MHD_dB(grid.lat.flatten(), grid.lon.flatten() + SAMSHIFT)
    mhdBe = get_MHD_dB(grid.lat.flatten(), grid.lon.flatten() + SAMSHIFT, component = 'Bphi [nT]')
    mhdBn = -get_MHD_dB(grid.lat.flatten(), grid.lon.flatten() + SAMSHIFT, component = 'Btheta [nT]')
    mhdB = np.sqrt(mhdBe**2 + mhdBn**2 + mhdBu**2).reshape(grid.lat.shape)
    d_perfect = np.hstack((mhdBe.flatten(), mhdBn.flatten(), mhdBu.flatten()))


    G_perfect = np.vstack((Gde, Gdn, Gdu))

    m_perfect = np.linalg.lstsq(G_perfect, d_perfect, rcond = 1e-3)[0]

    B = np.sqrt(Gde.dot(m).reshape(grid.eta.shape)**2 + Gdn.dot(m).reshape(grid.eta.shape)**2 + Gdu.dot(m).reshape(grid.eta.shape)**2)

    if COMPONENT == 'U':
        cntrs = axmap.contourf(grid.xi, grid.eta, Gdu.dot(m).reshape(grid.eta.shape), levels = np.linspace(-500, 500, 12), cmap = plt.cm.bwr, zorder = 0, extend = 'both')
        axmap_true.contourf(grid.xi, grid.eta, mhdBu.reshape(grid.eta.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    if COMPONENT == 'E':
        cntrs = axmap.contourf(grid.xi, grid.eta, Gde.dot(m).reshape(grid.eta.shape), levels = np.linspace(-500, 500, 12), cmap = plt.cm.bwr, zorder = 0, extend = 'both')
        axmap_true.contourf(grid.xi, grid.eta, mhdBe.reshape(grid.eta.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    if COMPONENT == 'N':
        cntrs = axmap.contourf(grid.xi, grid.eta, Gdn.dot(m).reshape(grid.eta.shape), levels = np.linspace(-500, 500, 12), cmap = plt.cm.bwr, zorder = 0, extend = 'both')
        axmap_true.contourf(grid.xi, grid.eta, mhdBn.reshape(grid.eta.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    if COMPONENT not in ['E', 'N', 'U']:
        cntrs = axmap.contourf(grid.xi, grid.eta, B, levels = np.linspace(0, 700, 12), zorder = 0, extend = 'both')
        axmap_true.contourf(grid.xi, grid.eta, mhdB, levels = cntrs.levels, zorder = 0, extend = 'both')



    jlat = grid.lat_mesh[::3, ::3].flatten()
    jlon = grid.lon_mesh[::3, ::3].flatten()
    Gje, Gjn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
    je, jn = Gje.dot(m).flatten(), Gjn.dot(m).flatten()
    jxi, jeta = projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)
    xi, eta = projection.geo2cube(jlon, jlat)
    axmap.quiver(xi, eta, jxi, jeta, linewidth = 2, scale = 1e10)#, scale = 1e10)


    #je, jn = Gje.dot(m_perfect).flatten(), Gjn.dot(m_perfect).flatten()
    #jxi, jeta = projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)
    #axmap_true.quiver(xi, eta, jxi, jeta, linewidth = 1, scale = 1e10, color = 'black')#, scale = 1e10)



    mhd_je, mhd_jn = get_MHD_jeq(jlat, jlon + SAMSHIFT)
    mhd_jxi, mhd_jeta = projection.vector_cube_projection(mhd_je, mhd_jn, jlon, jlat)
    #axmap.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 1, scale = 10, color = 'grey')#, scale = 1e10)
    axmap_true.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 2, scale = 10, color = 'black')#, scale = 1e10)


    # plot contours of contant geomagnetic latitude
    xlim = axmap.get_xlim()
    ylim = axmap.get_ylim()
    for l in np.r_[60:90:5]:
        #glat, glon = geo2mag(np.ones(360)*l, np.linspace(0, 360, 360), epoch = 2020, inverse = True)
        #xi, eta = projection.geo2cube(glon, glat)
        xi, eta = projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
        axmap.plot(xi, eta, 'k-')
        axmap_true.plot(xi, eta, 'k-')

    rad_to_km = lambda x, pos = None: '%.1f' % (RI * x * 1e-3)
    for ax in [axmap, axmap_true]:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # use km on axes, not radians: 
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(rad_to_km))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(rad_to_km))

        ax.set_aspect('equal')
        ax.set_xlabel('km')
        ax.set_ylabel('km')

    if TRIM:
        axmap.set_xlim(ximin - 25/(RI * 1e-3), ximax + 25/(RI * 1e-3))
        axmap.set_ylim(etamin, etamax)
        axmap_true.set_xlim(ximin - 25/(RI * 1e-3), ximax + 25/(RI * 1e-3))
        axmap_true.set_ylim(etamin, etamax)

    axmap_true.set_title('Truth')
    axmap.set_title('Estimate')
    counter += 1
    break
    plt.savefig(info['output_path'] + str(counter).zfill(4) + '.png')


plt.show()
