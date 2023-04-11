import numpy as np
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices, CSgrid, CSprojection
from datetime import timedelta

RE  =  6371.2e3

map_params = {'LRES':40,
              'WRES':20,
              'W': 2200, # along-track dimension of analysis grid (TODO: This is more a proxy than a precise description)
              'L': 1400, # cross-track dimension of analysis grid (TODO: Same as above)
              'wshift':25, # shift the grid center wres km in cross-track direction
              'total_time_window':4*60,
              'strip_time_window':30,
              'RI':RE + 110e3, # height of the ionosphere [m]
              }

inversion_params = {'lambda1': 1e0,
                    'lambda2': 1e3}

def get_AE_map(data_lat, data_lon, data_r, # data coordinates 
               Be, Bn, Bu,  # data points
               cov_ee, cov_nn, cov_uu, cov_en, cov_eu, cov_nu, # elements of data covariance matrix
               times, # time stamps for each data point
               t0, sc_lat0, sc_lon0, sc_ve0, sc_vn0, # central time and spacecraft position and velocity at this time
               get_f1, # function to calculate eastward direction in QD coordinates
               map_params, inversion_params # man parameters and inversion parameters
               ):
    """
    Calculate maps of electrojet current and associated magnetic field disturbances in a strip
    centered 


    Parameters
    ----------
    data_lat: array
        N-element array of data point latitude [deg]
    data_lon: array
        N-element array of data point longitude [deg]
    data_r: array
        N-element array of data point radius [m]
    Be: array
        N-element array of measured eastward magnetic field component [nT]
    Bn: array
        N-element array of measured northward magnetic field component [nT]
    Br: array
        N-element array of measured upward magnetic field component [nT]
    cov_ee: array
        N-element array of diagnol elements in data covariance matrix for eastward component
    cov_nn: array
        N-element array of diagnol elements in data covariance matrix for northward component
    cov_uu: array
        N-element array of diagnol elements in data covariance matrix for upward component
    cov_en: array
        N-element array of data covariance between eastward and northward component
    cov_eu: array
        N-element array of data covariance between eastward and upward component
    cov_nu: array
        N-element array of data covariance between northward and upward component
    times: array
        N-element array with timestamp for each data point
    t0: datetime
        central timestamp for the map
    sc_lat0: float
        spacecraft latitude [lat] at time = t0
    sc_lon0: float
        spacecraft longitude [deg] at time = t0
    sc_ve0: float
        spacecraft eastward velocity in Earth-fixed frame (arb. units, only direction is important)
    sc_vn0: float
        spacecraft northward velocity in Earth-fixed frame (arb. units, only direction is important)
    get_f1: function
        a function of lon and lat that calculates vector in the magnetic eastward direction 
        (such as Richmond1995's f1 vector)
    map_params: dict, optional
        dictionary of parameters that determine grid resolution and extent. 
    inversion_params: dict, optional
        dictionary of inversion parameters

    Returns
    -------
    lon_map: array
        longitudes at which the return values are calculated [deg]
    lat_map: array
        latitudes at which the return values are calculated [deg]
    Be_map: array
        Map of the solution magnetic field eastward component [nT]
    Bn_map: array
        Map of the solution magnetic field northward component [nT]
    Bu_map: array
        Map of the solution magnetic field upward component [nT]
    je_map: array
        Map of the solution divergence-free current eastward component [mA/m]
    jn_map: array
        Map of the solution divergence-free current northward component [mA/m]
    grid: CSgrid
        The cubed sphere grid used in the inversion
    m: array
        Solution vector - the SECS amplitudes defined on each grid point
    """

    if not data_lat.size == data_lon.size == data_r.size == Be.size == Bn.size == Bu.size == cov_ee.size == cov_nn.size == cov_uu.size == cov_en.size == cov_eu.size == cov_nu.size == times.size:
        raise ValueError('All input must have the same size')

    data = pd.DataFrame({'lat':data_lat.flatten(), 'lon':data_lon.flatten(), 'r':data_r.flatten(), # data coordinates 
                         'Be':Be.flatten(), 'Bn':Bn.flatten(), 'Bu':Bu.flatten(),  # data points
                          'cov_ee':cov_ee.flatten(), 'cov_nn':cov_nn.flatten(), 'cov_uu':cov_uu.flatten(), 'cov_en':cov_en.flatten(), 'cov_eu':cov_eu.flatten(), 'cov_nu':cov_nu.flatten()},
                          index = times)

    full_timestep = timedelta(seconds = map_params['total_time_window'])
    print(data.shape)
    data = data[(data.index >= (t0 - full_timestep/2)) & (data.index <= (t0 + full_timestep/2))]
    print(data.shape)

    # set up the grid
    position = (sc_lon0, sc_lat0)
    orientation = (sc_vn0, -sc_ve0) # align coordinate system such that xi axis points right wrt to satellite velocity vector, and eta along velocity

    projection = CSprojection(position, orientation)
    L, W, LRES, WRES, wshift = map_params['L'], map_params['W'], map_params['LRES'], map_params['WRES'], map_params['wshift']
    grid = CSgrid(projection, L, W, LRES, WRES, wshift = wshift, R = map_params['RI'] *  1e-3)

    # set up matrix that produces gradients in the magnetic eastward direction, and use to construct regularization matrix LL:
    Le, Ln = grid.get_Le_Ln()
    f1 = get_f1(grid.lon.flatten(), grid.lat.flatten())
    f1 = f1/np.linalg.norm(f1, axis = 0) # normalize
    L = Le * f1[0].reshape((-1, 1)) + Ln * f1[1].reshape((-1, 1))
    LL = L.T.dot(L)


    # construct covariance matrix and invert it
    Wen = np.diagflat(data['cov_en'].values)
    Weu = np.diagflat(data['cov_eu'].values)
    Wnu = np.diagflat(data['cov_nu'].values)
    Wee = np.diagflat(data['cov_ee'].values)
    Wnn = np.diagflat(data['cov_nn'].values)
    Wuu = np.diagflat(data['cov_uu'].values)
    We = np.hstack((Wee, Wen, Weu))
    Wn = np.hstack((Wen, Wnn, Wnu))
    Wu = np.hstack((Weu, Wnu, Wuu))
    W  = np.vstack((We, Wn, Wu))
    Q  = np.linalg.inv(W)


    # construct design matrix
    Ge, Gn, Gu = get_SECS_B_G_matrices(data['lat'].values, data['lon'].values, data['r'].values,
                                       grid.lat.flatten(), grid.lon.flatten(), 
                                       current_type = 'divergence_free', RI = map_params['RI'])
    G = np.vstack((Ge, Gn, Gu))
    d = np.hstack(data[['Be', 'Bn', 'Bu']].values.T)

    GTQG = G.T.dot(Q).dot(G)
    GTQd = G.T.dot(Q).dot(d)
    scale = np.max(GTQG)
    R = np.eye(GTQG.shape[0]) * scale * inversion_params['lambda1'] + LL / np.abs(LL).max() * scale * inversion_params['lambda2']

    SS = np.linalg.inv(GTQG + R).dot(G.T.dot(Q))
    m = SS.dot(d).flatten()

    return(m, grid)


 

if __name__ == '__main__':
    """ load synthetic data and call the get_AE_map function """
    import cases
    from dipole import Dipole # https://github.com/klaundal/dipole
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from secsy import spherical 
    import datetime as dt
    import pandas as pd
    from importlib import reload
    reload(cases) 
    d2r = np.pi / 180

    RI = RE + 110e3 # radius of the ionosphere

    dpl = Dipole(epoch = 2020) # initialize Dipole object

    # load parameters and data file names
    info = cases.cases['case_1']
    timeres = info['timeres'] # time resolution of the data [sec]
    DT = info['DT'] # time window in minutes


    OBSHEIGHT = info['observation_height'] * 1e3 # observation height in m

    data = pd.read_pickle(info['filename'])

    # convert all geographic coordinates and vector components in data to geomagnetic:
    for i in range(4):
        i = i + 1
        data['lat_' + str(i)], data['lon_' + str(i)], data['dbe_' + str(i)], data['dbn_' + str(i)] = dpl.geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_' + str(i)].values, data['dbn_' + str(i)].values)
    data['sat_lat'], data['sat_lon'] = dpl.geo2mag(data['sat_lat'].values, data['sat_lon'].values)

    # calculate SC velocity
    te, tn = spherical.tangent_vector(data['sat_lat'][:-1].values, data['sat_lon'][:-1].values,
                                      data['sat_lat'][1 :].values, data['sat_lon'][1: ].values)

    data['ve'] = np.hstack((te, np.nan))
    data['vn'] = np.hstack((tn, np.nan))

    # get index of central point of analysis interval:
    tm = data.index[data.index.get_loc(info['tm'], method = 'nearest')]

    # limits of analysis interval:
    t0 = data.index[data.index.get_loc(tm - dt.timedelta(seconds = DT//timeres * 60), method = 'nearest')]
    t1 = data.index[data.index.get_loc(tm + dt.timedelta(seconds = DT//timeres * 60), method = 'nearest')]

    # get unit vectors pointing at satellite (Cartesian vectors)
    rs = []
    for t in [t0, tm, t1]:
        rs.append(np.array([np.cos(data.loc[t, 'sat_lat'] * d2r) * np.cos(data.loc[t, 'sat_lon'] * d2r),
                            np.cos(data.loc[t, 'sat_lat'] * d2r) * np.sin(data.loc[t, 'sat_lon'] * d2r),
                            np.sin(data.loc[t, 'sat_lat'] * d2r)]))

    # dimensions of analysis region/grid (in km)
    W = 600 + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km

    # spacecraft velocity at central time:
    v  = np.array((data.loc[tm, 've'], data.loc[tm, 'vn']))

    sc_lat0 = data.loc[tm, 'sat_lat'] 
    sc_lon0 = data.loc[tm, 'sat_lon'] 

    map_params = {'LRES':40.,
                  'WRES':20.,
                  'W': W, # along-track dimension of analysis grid (TODO: This is more a proxy than a precise description)
                  'L': 1400, # cross-track dimension of analysis grid (TODO: Same as above)
                  'wshift':25, # shift the grid center wres km in cross-track direction
                  'total_time_window':6*60,
                  'strip_time_window':30,
                  'RI':RI, # height of the ionosphere [m]
                  }

    inversion_params = {'lambda1': 1e0,
                        'lambda2': 1e3}


    obs = {'lat': [], 'lon': [], 'Be': [], 'Bn': [], 'Bu': [], 'cov_ee': [], 'cov_nn': [], 'cov_uu': [], 'cov_en': [], 'cov_eu': [], 'cov_nu': [], 'times':[]}
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

    for key in obs.keys():
        obs[key] = np.array(obs[key])


    """
    Make a function to calculate the magnetic eastward direction in the coordinate system of the other input data. In this case, since the input 
    data is in dipole coordinates, the eastward direction is just (1, 0) for all points. In a realistic scenario, we would use input data in 
    geographic coordinates and call for example apexpy's basevectors_qd function
    """
    get_f1 = lambda lon, lat: np.vstack((np.ones(lon.size), np.zeros(lon.size)))

    m, grid = get_AE_map(obs['lat'], obs['lon'], np.full_like(obs['lon'], RE + OBSHEIGHT), # data coordinates 
                         obs['Be'], obs['Bn'], obs['Bu'],  # data points
                         obs['cov_ee'], obs['cov_nn'], obs['cov_uu'], obs['cov_en'], obs['cov_eu'], obs['cov_nu'], # elements of data covariance matrix
                         obs['times'], # time stamps for each data point
                         tm, sc_lat0, sc_lon0, v[0], v[1], # central time and spacecraft position and velocity at this time
                         get_f1, # function to calculate eastward direction in QD coordinates
                         map_params, inversion_params )



    fig = plt.figure(figsize = (6, 14))
    axe_true      = plt.subplot2grid((16, 2), (0 , 0), rowspan = 5)
    axe_secs      = plt.subplot2grid((16, 2), (0 , 1), rowspan = 5)
    axn_true      = plt.subplot2grid((16, 2), (5 , 0), rowspan = 5)
    axn_secs      = plt.subplot2grid((16, 2), (5 , 1), rowspan = 5)
    axr_true      = plt.subplot2grid((16, 2), (10, 0), rowspan = 5)
    axr_secs      = plt.subplot2grid((16, 2), (10, 1), rowspan = 5)

    ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)


    # plot the data tracks:
    for i in range(4):

        lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
        xi, eta = grid.projection.geo2cube(lon, lat)
        for ax in [axe_secs, axn_secs, axr_secs]:
            ax.plot(xi, eta, color = 'C' + str(i), linewidth = 3)


    # set up G matrices for the magnetic field evaluated on a grid - for plotting maps
    Gde, Gdn, Gdu = get_SECS_B_G_matrices(grid.lat.flatten()+.1, grid.lon.flatten(), np.ones(grid.lon.size) * (RE + OBSHEIGHT), 
                                          grid.lat.flatten(), grid.lon.flatten(), 
                                          current_type = 'divergence_free', RI = RI)

    # get maps of MHD magnetic fields:
    mhdBu =  info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
    mhdBe =  info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
    mhdBn = -info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])

    # plot magnetic field in upward direction (MHD and retrieved)
    cntrs = axr_secs.contourf(grid.xi, grid.eta, Gdu.dot(m).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBu.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in eastward direction (MHD and retrieved)
    axe_secs.contourf(grid.xi, grid.eta, Gde.dot(m).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axe_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBe.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in northward direction (MHD and retrieved)
    axn_secs.contourf(grid.xi, grid.eta, Gdn.dot(m).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBn.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot colorbar:
    ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
    ax_cbar.set_xlabel('nT')
    ax_cbar.set_yticks([])


    # calculate the equivalent current of retrieved magnetic field:
    jlat = grid.lat_mesh[::2, ::2].flatten()
    jlon = grid.lon_mesh[::2, ::2].flatten()    
    Gje, Gjn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
    je, jn = Gje.dot(m).flatten(), Gjn.dot(m).flatten()
    xi, eta, jxi, jeta = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)

    # plot the equivalent current in the SECS panels:
    for ax in [axe_secs, axn_secs, axr_secs]:
        ax.quiver(xi, eta, jxi, jeta, linewidth = 2, scale = 6e9, zorder = 40, color = 'black')#, scale = 1e10)

    # calcualte the equivalent current corresponding to MHD output with perfect coverage:
    Ge_Bj, Gn_Bj, Gu_Bj = get_SECS_B_G_matrices(grid.lat_mesh, grid.lon_mesh, RE + OBSHEIGHT, grid.lat[::2, ::2], grid.lon[::2, ::2], RI = RI)
    mj = np.linalg.lstsq(np.vstack((Ge_Bj, Gn_Bj, Gu_Bj)), np.hstack((mhdBe, mhdBn, mhdBu)), rcond = 1e-2)[0]

    Ge_j, Gn_j = get_SECS_J_G_matrices(jlat, jlon, grid.lat[::2, ::2], grid.lon[::2, ::2], current_type = 'divergence_free', RI = RI)
    mhd_je, mhd_jn = Ge_j.dot(mj), Gn_j.dot(mj)
    xi, eta, mhd_jxi, mhd_jeta = grid.projection.vector_cube_projection(mhd_je, mhd_jn, jlon, jlat)

    # plot the MHD equivalent current in eaach panel
    for ax in [axe_true, axn_true, axr_true , axe_secs, axn_secs, axr_secs]:
        ax.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 2, scale = 6e9, color = 'grey', zorder = 38)#, scale = 1e10)


    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [axe_secs, axe_true, axn_secs, axn_true, axr_secs, axr_true]:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        ax.axis('off')

        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')


    # Write labels:
    for ax, label in zip([axe_secs, axe_true, axn_secs, axn_true, axr_secs, axr_true],
                         ['Be SECS', 'Be MHD', 'Bn SECS', 'Bn MHD', 'Br SECS', 'Br MHD']):
        
        ax.text(grid.xi.min() - 25/(RI * 1e-3), grid.eta.max() - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)


    # plot grid in top left panel to show spatial dimensions:
    xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                  np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
    for i in range(xigrid.shape[0]):
        axe_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
    for j in range(xigrid.shape[1]):
        axe_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)

    # set plot limits:
    for ax in [axe_secs, axe_true, axn_secs, axn_true, axr_secs, axr_true]:
        ax.set_xlim(grid.xi.min(), grid.xi.max())
        ax.set_ylim(grid.eta.min(), grid.eta.max())
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    # save plots:
    #plt.savefig('./figures/' + info['outputfn'] + 'inversion_example.png', dpi = 250)
    #plt.savefig('./figures/' + info['outputfn'] + 'inversion_example.pdf')


    # save the relevant parts of the datafile for publication
    #columns = ['lat_1', 'lon_1', 'dbe_1', 'dbn_1', 'dbu_1', 'lat_2', 'lon_2', 'dbe_2', 'dbn_2', 'dbu_2', 'lat_3', 'lon_3', 'dbe_3', 'dbn_3', 'dbu_3', 'lat_4', 'lon_4', 'dbe_4', 'dbn_4', 'dbu_4', 'sat_lat', 'sat_lon', 'dbe_measured_1', 'dbn_measured_1', 'dbu_measured_1', 'cov_ee_1', 'cov_nn_1', 'cov_uu_1', 'cov_en_1', 'cov_eu_1', 'cov_nu_1', 'dbe_measured_2', 'dbn_measured_2', 'dbu_measured_2', 'cov_ee_2', 'cov_nn_2', 'cov_uu_2', 'cov_en_2', 'cov_eu_2', 'cov_nu_2', 'dbe_measured_3', 'dbn_measured_3', 'dbu_measured_3', 'cov_ee_3', 'cov_nn_3', 'cov_uu_3', 'cov_en_3', 'cov_eu_3', 'cov_nu_3', 'dbe_measured_4', 'dbn_measured_4', 'dbu_measured_4', 'cov_ee_4', 'cov_nn_4', 'cov_uu_4', 'cov_en_4', 'cov_eu_4', 'cov_nu_4']
    #savedata = data[t0:t1][columns]
    #savedata.index = dt = (savedata.index-savedata.index[0]).seconds
    #savedata.index.name = 'seconds'
    #savedata.to_csv(info['outputfn'] + 'electrojet_inversion_data.csv')


    plt.show()
