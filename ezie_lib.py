#%% Import
import numpy as np
import scipy
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices, CSgrid, CSprojection
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import copy
from scipy.optimize import curve_fit
from kneed import KneeLocator
from scipy.interpolate import splrep, BSpline

#%% Standard solver and visualization

def standard_retrieval(obs, t0, sc_lon0, sc_lat0, sc_ve0, sc_vn0, 
                    map_params, get_f1, reg_params, PD=[],
                    plot_dir='.', plot_name='test'):
    
    RI = map_params['RI']
    Rez = map_params['Rez']
    
    # Solve inverse problem
    inv_result, grid = standard_solver(obs, t0, sc_lon0, sc_lat0, 
                                       sc_ve0, sc_vn0, 
                                       map_params, get_f1, reg_params)
    
    # Plot model predictions at h=80
    plt.ioff()
    axs, cax = basic_plot(inv_result, grid, obs, RI, Rez, PD)
    plt.savefig('{}/{}.png'.format(plot_dir, plot_name), bbox_inches='tight')
    plt.savefig('{}/{}.pdf'.format(plot_dir, plot_name), format='pdf', bbox_inches='tight')
    plt.close('all')
    plt.ion()
    
    # Plot model predictions at h=0
    plt.ioff()
    axs, cax = basic_plot(inv_result, grid, obs, RI, map_params['RE'], PD)
    plt.savefig('{}/{}_ground.png'.format(plot_dir, plot_name), bbox_inches='tight')
    plt.savefig('{}/{}_ground.pdf'.format(plot_dir, plot_name), format='pdf', bbox_inches='tight')
    plt.close('all')
    plt.ion()
    
    # Plot posterior model covariance in data space
    s_m                 = np.sqrt(np.diag(inv_result['Cpm']))
    s_je, s_jn          = get_J_std(inv_result['Cpm'], grid, RI)
    s_be, s_bn, s_bu    = get_B_std(inv_result['Cpm'], grid, RI, Rez)
    plt.ioff()
    axs, cax = basic_plot_Cpm([s_m, s_je, s_jn, s_be, s_bn, s_bu], grid, obs, RI, PD)
    plt.savefig('{}/{}_Cpm.png'.format(plot_dir, plot_name), bbox_inches='tight')
    plt.savefig('{}/{}_Cpm.pdf'.format(plot_dir, plot_name), format='pdf', bbox_inches='tight')
    plt.close('all')
    plt.ion()
    
    # Plot spatial resolution
    xi_FWHM, eta_FWHM, xi_flag, eta_flag = get_resolution(inv_result['R'], grid)
    ef = (xi_flag==1) & (eta_flag==1)
    plt.ioff()
    axs, cax = basic_plot_resolution(xi_FWHM, eta_FWHM, ef, grid, obs, RI, PD)
    plt.savefig('{}/{}_resolution.png'.format(plot_dir, plot_name), bbox_inches='tight')
    plt.savefig('{}/{}_resolution.pdf'.format(plot_dir, plot_name), format='pdf', bbox_inches='tight')
    plt.close('all')
    plt.ion()
    
    return inv_result, grid

#%% Lambda relation

def lambda_relation_plot_spot(obs, sc_lon0, sc_lat0, sc_ve0, sc_vn0, 
                              map_params, m_id=0,
                              plot_dir='.', plot_name='test'):

    grid = get_grid(sc_lon0, sc_lat0, sc_ve0, sc_vn0, map_params)
    
    plt.ioff()
    plt.figure()
    for i in range(4):
        xi, eta = grid.projection.geo2cube(obs['lon_'+str(i+1)], obs['lat_'+str(i+1)])    
        plt.plot(xi, eta)
    
    plt.plot(grid.xi.flatten()[m_id], grid.eta.flatten()[m_id], '.', markersize=10)
    for i in range(grid.xi_mesh.shape[0]):
        plt.plot(grid.xi_mesh[i, :], grid.eta_mesh[i, :], color='gray', linewidth=0.4)
    for i in range(grid.xi_mesh.shape[1]):
        plt.plot(grid.xi_mesh[:, i], grid.eta_mesh[:, i], color='gray', linewidth=0.4)
    
    plt.xlim([np.min(grid.xi), np.max(grid.xi)])
    plt.ylim([np.min(grid.eta), np.max(grid.eta)])
    
    plt.savefig('{}/{}_find_it.png'.format(plot_dir, plot_name), bbox_inches='tight')
    plt.savefig('{}/{}_find_it.pdf'.format(plot_dir, plot_name), format='pdf', bbox_inches='tight')
    plt.close('all')
    plt.ion()
    

def lambda_relation_get_gini(obs, t0, sc_lon0, sc_lat0, sc_ve0, sc_vn0, 
                              map_params, get_f1, l1s, l2s, verbose=True):
    
    RI = map_params['RI']
    Rez = map_params['Rez']
    
    grid = get_grid(sc_lon0, sc_lat0, sc_ve0, sc_vn0, map_params)
    
    LL = get_LL(grid, get_f1)
    
    Q = get_cov_inv(obs)
    
    ginis = calc_gini(np.ones(obs['lat'].size)*Rez, obs['lat'], obs['lon'],
                                        grid.lat, grid.lon, LL, Q, RI, 
                                        l1s, l2s, verbose)
        
    return ginis

def lambda_relation_find_ridge(hoyer, ps=10):
    # 1st axis is l1
    # 2nd axis is l2
    
    # Check for negatives
    for i in range(hoyer.shape[0]):
        if np.any(hoyer[i, :] < 0):
            hoyer[i, :] -= np.min(hoyer[i, :])
        
    # Allocate space for ridge
    l2_opt_id = np.zeros((hoyer.shape[0], ps)).astype(int)
    
    # Loop over all l1s
    for i in range(hoyer.shape[0]):
        try:
            l2_opt_id[i, 0] = np.argmax(hoyer[i, :])
            if l2_opt_id[i, 0] < 5:                
                kn, popt, pcov = if_no_max(hoyer[i, :])
                
                ids = [kn.knee]
                left = kn.knee - 1
                right = kn.knee + 1
                while len(ids) < ps:
                    if hoyer[i, left] > hoyer[i, right]:
                        ids.append(left)
                        left -= 1
                    else:
                        ids.append(right)
                        right += 1
                
                l2_opt_id[i, :] = ids
            else:                
                hoyer[i, :np.argmax(hoyer[i, :])-ps] = -1
                hoyer[i, np.argmax(hoyer[i, :])+ps:] = -1
                for j in range(ps):
                    l2_opt_id[i, j] = np.argmax(hoyer[i, :])
                    hoyer[i, l2_opt_id[i, j]] = -1
        except:
            l2_opt_id[i, :] = -1
            print('error')

    return l2_opt_id

def if_no_max(y):

    def step_function(x, a0, a1, a2, a3):
        return a0 / (1 + np.exp(a2*(x-a3))) + a1
    
    def dd_step_function(x, a0, a1, a2, a3):
        return (a0 * a2**2 * np.exp(a2*x) * (np.exp(a2*x) - 1)) / (np.exp(a2*x)+1)**3

    x = np.linspace(-100, 100, 200)
    
    ymax = np.max(y)
    ymin = np.min(y)
    
    bounds = ([0, 0, 1e-20, -90], [1.5*ymax, ymax, 10, 90])
    p0 = [ymax-ymin, ymin, 0.5, 0]
    
    popt, pcov = curve_fit(step_function, x, y, bounds=bounds, p0=p0, loss='huber')
    yy = step_function(x, *popt)
    
    yy_kn = yy[:100+int(popt[-1])]
    kn = KneeLocator(np.arange(len(yy_kn)), yy_kn, curve='concave', direction='decreasing')
    
    return kn, popt, pcov    
    
def make_spline_fit(xx, yy, steps=100, s=10500):
    tck = splrep(xx, yy, s=s, k=3)
    x_new = np.linspace(np.min(xx), np.max(xx), steps)
    l2_opt_fit = BSpline(*tck)(x_new)
    
    return l2_opt_fit, tck

#%% L-curve

def iterative_retrieval(obs, t0, sc_lon0, sc_lat0, sc_ve0, sc_vn0, 
                        map_params, get_f1, l1s, l2s,
                        full=False, Lcurve=False):
        
    RE = map_params['RE']
    RI = map_params['RI']
    Rez = map_params['Rez']
    
    grid = get_grid(sc_lon0, sc_lat0, sc_ve0, sc_vn0, map_params)
    
    LL = get_LL(grid, get_f1)
    
    Q = get_cov_inv(obs)
    
    inv_results = solve_inverse_problem_iterative(np.ones(obs['lat'].size)*Rez,
                                                  obs['lat'], obs['lon'], 
                                                  obs['Be'], obs['Bn'], obs['Bu'], 
                                                  grid.lat, grid.lon, 
                                                  LL, Q, RI, l1s, l2s, 
                                                  full=full, Lcurve=Lcurve)
    
    return inv_results
    
def robust_Kneedle(rnorm, mnorm):
    kn_id = 0
    i = 0
    while kn_id < 1:
        kn = KneeLocator(np.log10(rnorm[i:]), np.log10(mnorm[i:]), curve='convex', direction='decreasing')
        kn_id = np.argmin(abs((np.log10(rnorm) - kn.knee)))
        i += 1
    return kn_id, i-1
    
#%% Extrapolation

def get_ID_and_SED(obs, t0, sc_lon0, sc_lat0, sc_ve0, sc_vn0, map_params, 
                   get_f1, reg_params):
    
    inv_result, grid = standard_solver(obs, t0, sc_lon0, sc_lat0, 
                                       sc_ve0, sc_vn0, 
                                       map_params, get_f1, reg_params)
    
    xi_FWHM, eta_FWHM, xi_flag, eta_flag = get_resolution(inv_result['R'], grid)
    ef = (xi_flag==1) & (eta_flag==1)
    
    # Determine ID
    ID = np.zeros(ef.shape).astype(int)
    ID[ef] = 1
        
    SED = np.zeros(grid.shape).astype(int)
    for row in range(ID.shape[0]):
        for col in range(ID.shape[1]):
            
            # Check if we are in the ID
            if ID[row, col] == 0:
                continue
            
            # Calculate radius
            xi_radius = xi_FWHM.reshape(grid.shape)[row, col]
            eta_radius = eta_FWHM.reshape(grid.shape)[row, col]
    
            xi_radius = int((xi_radius / grid.Lres) / 2)
            eta_radius = int((eta_radius / grid.Wres) / 2)    
    
            # Mark the area the ellipse covers
            for j in range(eta_radius+1):
                xi_radius_j = int(np.round(np.sqrt(xi_radius**2 * (1 - j**2/eta_radius**2)), 0))        
                SED[row+j, col-xi_radius_j:col+xi_radius_j+1] += 1
                if j != 0:
                    SED[row-j, col-xi_radius_j:col+xi_radius_j+1] += 1    
    
    # Make sure the ID is part of the SED
    SED[ef] += 1
    
    # Make it binary
    SED[SED != 0] = 1
    
    return ID, SED, grid
    
def get_largest_rectangle(SED, grid):
    
    # Need to flip 1's and 0's to make boolean logic easier
    matrix = np.where((SED==0)|(SED==1), SED^1, SED)
    
    # find the center of the grid - approximately
    row_mid = int(SED.shape[0]/2)
    col_mid = int(SED.shape[1]/2)
    
    # Determine all possible rectangle configurations
    sleft_r = np.arange(1, col_mid).astype(int)
    sright_r = np.arange(1, SED.shape[1]-col_mid-1).astype(int)
    sbot_r = np.arange(1, row_mid).astype(int)
    stop_r = np.arange(1, SED.shape[0]-row_mid-1).astype(int)
    
    # Information on the current largest rectangle
    area = 0 # Area
    sizes = [] # Parameters
    
    # loop over all possible rectangles (brute force) - test case has 1.5e5
    for i, sleft in enumerate(sleft_r):
        for j, sright in enumerate(sright_r):
            for k, stop in enumerate(stop_r):
                for l, sbot in enumerate(sbot_r):
                    
                    # Get coordinates of the rectangle
                    cleft, cright, ctop, cbot = get_rectangle(row_mid, col_mid, 
                                                              sleft, sright, 
                                                              stop, sbot)
                    
                    # Chekc if rectangle exceeds the SED
                    okay = sum(matrix[np.hstack((cleft[0], cright[0], 
                                                 ctop[0], cbot[0])),
                                      np.hstack((cleft[1], cright[1], 
                                                 ctop[1], cbot[1]))]) == 0
                    
                    # Compare rectangle to the current largest rectangle
                    if okay:
                        area_i = ((sright+sleft+1)*40)*((stop+sbot+1)*20)
                        if area_i > area:
                            area = copy.deepcopy(area_i)
                            sizes = [sleft, sright, stop, sbot]

    # Get coordinates for the optimal rectangle
    cleft, cright, ctop, cbot = get_rectangle(row_mid, col_mid, sizes[0], 
                                              sizes[1], sizes[2], sizes[3])
    
    # Get the min and max in CS coordinates
    rows = np.hstack((cleft[0], cright[0], ctop[0], cbot[0]))
    cols = np.hstack((cleft[1], cright[1], ctop[1], cbot[1]))
    PD_xi = [np.min(grid.xi[rows, cols]), np.max(grid.xi[rows, cols])]
    PD_eta = [np.min(grid.eta[rows, cols]), np.max(grid.eta[rows, cols])]
    
    return PD_xi, PD_eta

def get_rectangle(row_mid, col_mid, sleft, sright, stop, sbot):
    
    cleft = [np.arange(row_mid-sbot, row_mid+stop+1).astype(int),
             np.ones(stop+sbot+1).astype(int)*col_mid-sleft]
    
    cright = [np.arange(row_mid-sbot, row_mid+stop+1).astype(int),
              np.ones(stop+sbot+1).astype(int)*col_mid+sright]
    
    ctop = [np.ones(sleft+sright+1).astype(int)*row_mid+stop,
            np.arange(col_mid-sleft, col_mid+sright+1).astype(int)]
    
    cbot = [np.ones(sleft+sright+1).astype(int)*row_mid-sbot,
            np.arange(col_mid-sleft, col_mid+sright+1).astype(int)]
    
    return cleft, cright, ctop, cbot

#%% Spatial resolution

def get_resolution(R, grid):
    xi_FWHM  = np.zeros(grid.size)
    eta_FWHM = np.zeros(grid.size)
    xi_flag  = np.zeros(grid.size).astype(int)
    eta_flag = np.zeros(grid.size).astype(int)
    for i in range(R.shape[0]):
        
        #PSF = abs(R[:, i].reshape(grid.shape))
        PSF = R[:, i].reshape(grid.shape)
        
        PSF_xi = np.sum(PSF, axis=0)
        i_left, i_right, flag = left_right(PSF_xi)
        xi_FWHM[i] = (i_right-i_left)*grid.Lres
        xi_flag[i] = flag
        
        PSF_eta = np.sum(PSF, axis=1)
        i_left, i_right, flag = left_right(PSF_eta)
        eta_FWHM[i] = (i_right-i_left)*grid.Wres
        eta_flag[i] = flag
        
    xi_FWHM = xi_FWHM.reshape(grid.shape)
    eta_FWHM = eta_FWHM.reshape(grid.shape)
    xi_flag = xi_flag.reshape(grid.shape)
    eta_flag = eta_flag.reshape(grid.shape)
        
    return xi_FWHM, eta_FWHM, xi_flag, eta_flag

def left_right(PSF_i, fraq=0.5, inside=False, x='', x_min='', x_max=''):
    
    if inside:
        PSF_ii = copy.deepcopy(PSF_i)
        valid = False
        while not valid:
            i_max = np.argmax(PSF_ii)
            
            x_i = x[i_max]
            if (x_i >= x_min) and (x_i <= x_max):
                valid = True
            else:
                PSF_ii[i_max] = np.min(PSF_i)            
                        
    else:
        i_max = np.argmax(PSF_i)    
    
    PSF_max = PSF_i[i_max]
        
    j = 0
    i_left = 0
    left_edge = True
    while (i_max - j) >= 0:
        if PSF_i[i_max - j] < fraq*PSF_max:
            
            dPSF = PSF_i[i_max - j + 1] - PSF_i[i_max - j]
            dx = (fraq*PSF_max - PSF_i[i_max - j]) / dPSF
            i_left = i_max - j + dx
            
            left_edge = False
            
            break
        else:
            j += 1

    j = 0
    i_right = len(PSF_i) - 1
    right_edge = True
    while (i_max + j) < len(PSF_i):
        if PSF_i[i_max + j] < fraq*PSF_max:
            
            dPSF = PSF_i[i_max + j] - PSF_i[i_max + j - 1]
            dx = (fraq*PSF_max - PSF_i[i_max + j - 1]) / dPSF
            i_right = i_max + j - 1 + dx 
            
            right_edge = False
            
            break
        else:
            j += 1
    
    flag = True
    if left_edge and right_edge:
        print('I think something is wrong')
        flag = False
    elif left_edge:
        i_left = i_max - (i_right - i_max)
        flag = False
    elif right_edge:
        i_right = i_max + (i_max - i_left)
        flag = False
    
    return i_left, i_right, flag

#%% Model variance of predictions

def get_J_std(Cpm, grid, RI):
    Ge, Gn = get_SECS_J_G_matrices(grid.lat_mesh.flatten(), grid.lon_mesh.flatten(), grid.lat.flatten(), grid.lon.flatten(), current_type='divergence_free', RI=RI)
    s_je = np.sqrt(np.diag(Ge.dot(Cpm).dot(Ge.T)))
    s_jn = np.sqrt(np.diag(Gn.dot(Cpm).dot(Gn.T)))
    return s_je, s_jn

def get_B_std(Cpm, grid, RI, Rez):
    Ge, Gn, Gu = get_SECS_B_G_matrices(grid.lat_mesh.flatten(), grid.lon_mesh.flatten(), np.ones(grid.xi_mesh.size)*Rez, 
                                       grid.lat.flatten(), grid.lon.flatten(), current_type='divergence_free', RI=RI)
    s_be = np.sqrt(np.diag(Ge.dot(Cpm).dot(Ge.T)))
    s_bn = np.sqrt(np.diag(Gn.dot(Cpm).dot(Gn.T)))
    s_bu = np.sqrt(np.diag(Gu.dot(Cpm).dot(Gu.T)))
    return s_be, s_bn, s_bu

#%% Function that returns grid object

def get_grid(sc_lon0, sc_lat0, sc_ve0, sc_vn0, map_params):
    # set up the grid
    position = (sc_lon0, sc_lat0)
    orientation = (sc_vn0, -sc_ve0) # align coordinate system such that xi axis points right wrt to satellite velocity vector, and eta along velocity
    projection = CSprojection(position, orientation)
    L, W, LRES, WRES, wshift = map_params['L'], map_params['W'], map_params['LRES'], map_params['WRES'], map_params['wshift']
    grid = CSgrid(projection, L, W, LRES, WRES, wshift = wshift, R = map_params['RI'] *  1e-3)
    return grid

#%% Function that returns the R=L.T.dot(L) matrix for east/west smoothing

def get_LL(grid, get_f1):
    # set up matrix that produces gradients in the magnetic eastward direction, and use to construct regularization matrix LL:
    Le, Ln = grid.get_Le_Ln()
    f1 = get_f1(grid.lon.flatten(), grid.lat.flatten())
    f1 = f1/np.linalg.norm(f1, axis = 0) # normalize
    L = Le * f1[0].reshape((-1, 1)) + Ln * f1[1].reshape((-1, 1))
    LL = L.T.dot(L)
    return LL

#%% Function that returns the inverted data covariance matrix

def get_cov_inv(obs):
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
    return Q

#%% Wrapper for calling inverse problem function

def standard_solver(obs, t0, sc_lon0, sc_lat0, sc_ve0, sc_vn0, 
                    map_params, get_f1, reg_params):
    
    RI = map_params['RI']
    Rez = map_params['Rez']
    
    grid = get_grid(sc_lon0, sc_lat0, sc_ve0, sc_vn0, map_params)
    
    LL = get_LL(grid, get_f1)
    
    Q = get_cov_inv(obs)
    
    inv_result = solve_inverse_problem(np.ones(obs['lat'].size)*Rez,
                                       obs['lat'], obs['lon'], 
                                       obs['Be'], obs['Bn'], obs['Bu'], 
                                       grid.lat, grid.lon, 
                                       LL, Q, RI, l1=reg_params['lambda1'], l2=reg_params['lambda2'],
                                       full=True)
    
    return inv_result, grid

#%% Function for solving the inverse problem and a bit more

def solve_inverse_problem(r, lat, lon, Be, Bn, Bu, lat_secs, lon_secs, LL, Q, RI, l1=0, l2=0, full=False, Lcurve=False):
    
    d = np.hstack((Be, Bn, Bu))
    
    Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs, current_type='divergence_free', RI=RI)
    G = np.vstack((Ge, Gn ,Gu))
    
    GTQG = G.T.dot(Q).dot(G)
    GTQd = G.T.dot(Q).dot(d)
    
    gtg_mag = np.median(np.diag(GTQG))
    LL_mag = np.max(LL)
    
    reg = l1*gtg_mag*np.eye(LL.shape[0]) + l2*gtg_mag/LL_mag*LL
    
    inv_result = {}
    inv_result['d'] = d
    
    if full:
        inv_result['Cpm'] = scipy.linalg.solve(GTQG + reg, np.eye(reg.shape[0]))
        inv_result['R'] = inv_result['Cpm'].dot(GTQG)
        inv_result['m'] = inv_result['Cpm'].dot(GTQd)
        inv_result['d_pred'] = G.dot(inv_result['m'])
        inv_result['G'] = [Ge, Gn, Gu]
        
    else:
        inv_result['m'] = scipy.linalg.solve(GTQG + reg, GTQd)
        inv_result['d_pred'] = G.dot(inv_result['m'])

    if Lcurve:
        inv_result['res'] = d - inv_result['d_pred']
        inv_result['dnorm'] = np.sqrt(inv_result['res'].T.dot(Q).dot(inv_result['res']))
        inv_result['mnorm'] = np.sqrt(inv_result['m'].T.dot(gtg_mag*np.eye(LL.shape[0]) + gtg_mag/LL_mag*LL).dot(inv_result['m']))
        inv_result['mnorm_l1'] = np.sqrt(inv_result['m'].T.dot(gtg_mag*np.eye(LL.shape[0])).dot(inv_result['m']))
        inv_result['mnorm_l2'] = np.sqrt(inv_result['m'].T.dot(gtg_mag/LL_mag*LL).dot(inv_result['m']))
        
    return inv_result

#%% Function for solving the inverse problem and a bit more

def solve_inverse_problem_iterative(r, lat, lon, Be, Bn, Bu, lat_secs, lon_secs, LL, Q, RI, l1s, l2s, full=False, Lcurve=False, verbose=True):
    
    d = np.hstack((Be, Bn, Bu))
    
    Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs, current_type='divergence_free', RI=RI)
    G = np.vstack((Ge, Gn ,Gu))
    
    GTQG = G.T.dot(Q).dot(G)
    GTQd = G.T.dot(Q).dot(d)
    
    gtg_mag = np.median(np.diag(GTQG))
    LL_mag = np.max(LL)
    
    inv_results = []
    for i, (l1, l2) in enumerate(zip(l1s, l2s)):
        if verbose:
            print('Iteration: {}/{}'.format(i+1, len(l1s)))
                
        reg = l1*gtg_mag*np.eye(LL.shape[0]) + l2*gtg_mag/LL_mag*LL
    
        inv_result = {}
        inv_result['d'] = d
        if full:
            inv_result['Cpm'] = scipy.linalg.solve(GTQG + reg, np.eye(reg.shape[0]))
            inv_result['R'] = inv_result['Cpm'].dot(GTQG)
            inv_result['m'] = inv_result['Cpm'].dot(GTQd)
            inv_result['d_pred'] = G.dot(inv_result['m'])
        
        else:
            inv_result['m'] = scipy.linalg.solve(GTQG + reg, GTQd)
            inv_result['d_pred'] = G.dot(inv_result['m'])

        if Lcurve:
            inv_result['res'] = d - inv_result['d_pred']
            inv_result['dnorm'] = np.sqrt(inv_result['res'].T.dot(Q).dot(inv_result['res']))
            inv_result['mnorm'] = np.sqrt(inv_result['m'].T.dot(gtg_mag*np.eye(LL.shape[0]) + gtg_mag/LL_mag*LL).dot(inv_result['m']))
            inv_result['mnorm_l1'] = np.sqrt(inv_result['m'].T.dot(gtg_mag*np.eye(LL.shape[0])).dot(inv_result['m']))
            inv_result['mnorm_l2'] = np.sqrt(inv_result['m'].T.dot(gtg_mag/LL_mag*LL).dot(inv_result['m']))
            
        inv_results.append(inv_result)
        
    return inv_results

#%%

def calc_gini(r, lat, lon, lat_secs, lon_secs, LL, Q, RI, l1s=[0], l2s=[0], verbose=True):
    
    Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs, current_type='divergence_free', RI=RI)
    G = np.vstack((Ge, Gn ,Gu))
    
    GTQG = G.T.dot(Q).dot(G)
    
    gtg_mag = np.median(np.diag(GTQG))
    LL_mag = np.max(LL)
    
    ginis = np.zeros((G.shape[1], len(l1s), len(l2s)))
    for i, l1 in enumerate(l1s):
        for j, l2 in enumerate(l2s):
            if verbose:
                print('l1: {}/{} , l2: {}/{}'.format(i+1, len(l1s), j+1, len(l2s)))
            
            reg = l1*gtg_mag*np.eye(LL.shape[0]) + l2*gtg_mag/LL_mag*LL
            R = scipy.linalg.solve(GTQG + reg, GTQG)
            
            N = R.shape[0]
            k = np.tile(np.arange(1, N+1), (N, 1))
            
            R = np.sort(abs(R), axis=1)
            #R = np.sort(R, axis=1)
            norm_l1 = np.tile(np.sum(abs(R), axis=1).reshape(-1, 1), (1, N))
            
            gini = (R / norm_l1) * ((N - k + 0.5)/N)
            gini = 1 - 2*np.sum(gini, axis=1)
            
            ginis[:, i, j] = gini
            
    return ginis
    
#%% Function for basic plot of the reconstructed B

def basic_plot(inv_result, grid, obs, RI, Rez, PD=[]):
    
    Ge, Gn, Gu = get_SECS_B_G_matrices(grid.lat_mesh.flatten(), grid.lon_mesh.flatten(), np.ones(grid.xi_mesh.size)*Rez, 
                                       grid.lat.flatten(), grid.lon.flatten(), current_type='divergence_free', RI=RI)
    Be = Ge.dot(inv_result['m'])
    Bn = Gn.dot(inv_result['m'])
    Bu = Gu.dot(inv_result['m'])
    
    jlat = grid.lat_mesh[::4, ::4].flatten()
    jlon = grid.lon_mesh[::4, ::4].flatten()
    Ge, Gn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type='divergence_free', RI=RI)
    
    je, jn = Ge.dot(inv_result['m']), Gn.dot(inv_result['m'])
    xi, eta, jxi, jeta = grid.projection.vector_cube_projection(je, jn, jlon, jlat)
    
    # Start figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))    
    cax = fig.add_axes([0.1, 0, 0.8, 0.03]) 
    
    # colorbar range
    vmax = np.max(abs(np.hstack((Be, Bn, Bu))))
    clevels = np.linspace(-vmax, vmax, 40)

    # plot magnetic field in upward direction (MHD and retrieved)
    for (ax, Bi, label) in zip(axs, [Be, Bn, Bu], 
                               ['$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS']):
        cc = plot_map(ax, grid.xi_mesh.flatten(), grid.eta_mesh.flatten(), 
                      Bi.flatten(),
                      obs, grid, RI, 'bwr', clevels, label, PD=PD)
        
    # plot colorbar:
    cax.contourf(np.vstack((cc.levels, cc.levels)), np.vstack((np.zeros(cc.levels.size), np.ones(cc.levels.size))), np.vstack((cc.levels, cc.levels)), cmap='bwr', levels=cc.levels)
    cax.set_xlabel('nT')
    cax.set_yticks([])

    # plot the equivalent current in the SECS panels:
    for ax in axs:
        ax.quiver(xi, eta, jxi, jeta, linewidth=2, scale=1e10, zorder=40, color = 'black')#, scale = 1e10)

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    return axs, cax

#%% Function for basic plot of the posterior model covariance

def basic_plot_Cpm(ss, grid, obs, RI, PD=[]):
    
    # Start figure
    fig, axs = plt.subplots(2, 3, figsize=(15, 12))
    cax = fig.add_axes([0.1, 0, 0.8, 0.03])
    cax_j = fig.add_axes([0.1, -.05, 0.8, 0.001])
    cax_b = fig.add_axes([0.1, -.1, 0.8, 0.001])
    
    for axi in ['left', 'top', 'right']:
        cax_j.spines[axi].set_visible(False)
        cax_b.spines[axi].set_visible(False)
    
    # colorbar range
    vmax_m = np.max(ss[0])
    vmax_j = np.max(np.hstack((ss[1], ss[2])))
    vmax_b = np.max(np.hstack((ss[3], ss[4], ss[5])))
    clvl_m = np.linspace(0, vmax_m, 40)
    clvl_j = np.linspace(0, vmax_j, 40)
    clvl_b = np.linspace(0, vmax_b, 40)

    # plot magnetic field in upward direction (MHD and retrieved)
    
    for (ax, si, label) in zip(axs[0, 1:], ss[1:3], ['std Je', 'std Jn']):
        cc = plot_map(ax, grid.xi_mesh.flatten(), grid.eta_mesh.flatten(), 
                      si.flatten(), obs, grid, RI, 'Reds', clvl_j, label, PD=PD)
    
    for (ax, si, label) in zip(axs[1, :], ss[3:], ['std Be', 'std Bn', 'std Bu']):
        cc = plot_map(ax, grid.xi_mesh.flatten(), grid.eta_mesh.flatten(), 
                      si.flatten(), obs, grid, RI, 'Reds', clvl_b, label, PD=PD)
    
    cc = plot_map(axs[0,0], grid.xi.flatten(), grid.eta.flatten(), 
                  ss[0], obs, grid, RI, 'Reds', clvl_m, 'std SECS', PD=PD)
    
    # plot colorbar:
    cax.contourf(np.vstack((cc.levels, cc.levels)), np.vstack((np.zeros(cc.levels.size), np.ones(cc.levels.size))), np.vstack((cc.levels, cc.levels)), cmap='Reds', levels=cc.levels)
    cax.set_xlabel('SECS []')
    cax.set_yticks([])

    cax_j.set_xticks(np.arange(0, vmax_j, int(vmax_j/8)))
    cax_j.set_xlabel('J []')
    cax_j.set_yticks([])
    
    cax_b.set_xticks(np.arange(0, vmax_b, int(vmax_b/8)))
    cax_b.set_xlabel('B [nT]')
    cax_b.set_yticks([])

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    return axs, cax

#%% Function for basic plot of the model resolution

def basic_plot_resolution(xi_FWHM, eta_FWHM, ef, grid, obs, RI, PD=[]):
    
    # Start figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 11))
    cax = fig.add_axes([0.1, 0, 0.8, 0.03]) 
    
    mask = np.zeros(grid.shape)
    mask[ef] = 1
    
    # colorbar range
    vmax = np.max([np.max(xi_FWHM[ef]), np.max(eta_FWHM[ef])])
    clevels = np.linspace(0, vmax, 40)

    # plot magnetic field in upward direction (MHD and retrieved)
    for (ax, var, label) in zip(axs, [xi_FWHM, eta_FWHM], 
                               ['xi', 'eta']):
        cc = plot_map(ax, grid.xi, grid.eta, 
                      var,
                      obs, grid, RI, 'Reds', clevels, label, mask=mask, PD=PD)
        
    # plot colorbar:
    cax.contourf(np.vstack((cc.levels, cc.levels)), np.vstack((np.zeros(cc.levels.size), np.ones(cc.levels.size))), np.vstack((cc.levels, cc.levels)), cmap='Reds', levels=cc.levels)
    cax.set_xlabel('km')
    cax.set_yticks([])

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    return axs, cax

#%%

def plot_map(ax, xiv, etav, var, obs, grid, RI, cmap, clevels, label, mask=-1, PD=[]):
    
    ximin = np.min(xiv)
    ximax = np.max(xiv)
    etamin = np.min(etav)
    etamax = np.max(etav)
    
    fill = False
    if np.all(mask != -1):
        fill = True
        var = np.ma.array(var, mask=mask < 0.6)

    # Check if Prediction Domain is used
    pe1 = [mpe.Stroke(linewidth=6, foreground='white',alpha=1), mpe.Normal()]
    if len(PD) != 0:
        ax.plot([PD[0][0], PD[0][0], PD[0][1], PD[0][1], PD[0][0]],
                [PD[1][0], PD[1][1], PD[1][1], PD[1][0], PD[1][0]],
                color='k', linewidth=5, path_effects=pe1)

    # plot the data tracks:
    for i in range(4):
        lon = obs['lon_' + str(i+1)]
        lat = obs['lat_' + str(i+1)]
        xi, eta = grid.projection.geo2cube(lon, lat)
        ax.plot(xi, eta, color = 'C' + str(i), linewidth = 5, path_effects=pe1)

    # plot map
    if fill:
        cc = ax.contourf(xiv, etav, var, 
                         levels=clevels, cmap=cmap, zorder=0, extend='both')
    else:
        cc = ax.tricontourf(xiv, etav, var,
                            levels=clevels, cmap=cmap, zorder=0, extend='both')
    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for l in np.r_[60:90:5]:
        xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
        ax.plot(xi, eta, color='lightgrey', linewidth=.5, zorder=1)

    for l in np.r_[0:360:15]:
        xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
        ax.plot(xi, eta, color='lightgrey', linewidth=.5, zorder=1)

    ax.axis('off')

    # Write labels:
    ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)

    # set plot limits and write label:
    ax.set_xlim(ximin, ximax)
    ax.set_ylim(etamin, etamax)
    ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)
        
    ax.set_adjustable('datalim') 
    ax.set_aspect('equal')

    return cc

#%% Unnecessary code to satisfy Sam

def geod2geoc(gdlat, height, Bn, Bu):
    """
    Convert from geocentric to geodetic coordinates

    Example:
    --------
    theta, r, B_th, B_r = geod2lat(gdlat, height, Bn, Bu)

    Parameters
    ----------
    gdlat : array
        Geodetic latitude [degrees]
    h : array
        Height above ellipsoid [km]
    Bn : array
        Vector in northward direction, relative to ellipsoid
    Bu : array
        Vector in upward direction, relative to ellipsoid

    Returns
    -------
    theta : array
        Colatitudes [degrees]
    r : array
        Radius [km]
    B_th : array
        Vector component in theta direction
    B_r : array
        Vector component in radial direction
    """
    
    # World Geodetic System 84 parameters:
    WGS84_e2 = 0.00669437999014
    WGS84_a  = 6378.137 # km
    
    a = WGS84_a
    b = a*np.sqrt(1 - WGS84_e2)

    # Convert geodetic latitude angles to radians
    gdlat_rad = np.radians(gdlat)

    sin_alpha_2 = np.sin(gdlat_rad)**2
    cos_alpha_2 = np.cos(gdlat_rad)**2

    # calculate geocentric latitude and radius
    tmp = height * np.sqrt(a**2 * cos_alpha_2 + b**2 * sin_alpha_2)
    beta = np.arctan((tmp + b**2)/(tmp + a**2) * np.tan(gdlat_rad))
    theta = np.pi/2 - beta
    r = np.sqrt(height**2 + 2 * tmp + a**2 * (1 - (1 - (b/a)**4) * sin_alpha_2) / (1 - (1 - (b/a)**2) * sin_alpha_2))

    # calculate geocentric components
    psi  =  np.sin(gdlat_rad) * np.sin(theta) - np.cos(gdlat_rad) * np.cos(theta)
    
    B_r  = -np.sin(psi) * Bn + np.cos(psi) * Bu
    B_th = -np.cos(psi) * Bn - np.sin(psi) * Bu

    # Convert theta to degrees
    theta = np.degrees(theta)

    return theta, r, B_th, B_r


def geoc2geod(theta, r, B_th, B_r, matrix=False):
    """
    Convert from geodetic to geocentric coordinates

    Based on Matlab code by Nils Olsen, DTU

    Example:
    --------
    gdlat, height, Bn, Bu = geod2lat(theta, r, B_th, B_r)

    Parameters
    ----------
    theta : array
        Colatitudes [degrees]
    r : array
        Radius [km]
    B_th : array
        Vector component in theta direction
    B_r : array
        Vector component in radial direction

    Returns
    -------
    gdlat : array
        Geodetic latitude [degrees]
    h : array
        Height above ellipsoid [km]
    Bn : array
        Vector in northward direction, relative to ellipsoid
    Bu : array
        Vector in upward direction, relative to ellipsoid
    """
    
    WGS84_a  = 6378.137 # km
    WGS84_e2 = 0.00669437999014
    a = WGS84_a
    b = a*np.sqrt(1 - WGS84_e2)

    E2 = 1.-(b/a)**2
    E4 = E2*E2
    E6 = E4*E2
    E8 = E4*E4
    OME2REQ = (1.-E2)*a
    A21 =     (512.*E2 + 128.*E4 + 60.*E6 + 35.*E8)/1024.
    A22 =     (                        E6 +     E8)/  32.
    A23 = -3.*(                     4.*E6 +  3.*E8)/ 256.
    A41 =    -(           64.*E4 + 48.*E6 + 35.*E8)/1024.
    A42 =     (            4.*E4 +  2.*E6 +     E8)/  16.
    A43 =                                   15.*E8 / 256.
    A44 =                                      -E8 /  16.
    A61 =  3.*(                     4.*E6 +  5.*E8)/1024.
    A62 = -3.*(                        E6 +     E8)/  32.
    A63 = 35.*(                     4.*E6 +  3.*E8)/ 768.
    A81 =                                   -5.*E8 /2048.
    A82 =                                   64.*E8 /2048.
    A83 =                                 -252.*E8 /2048.
    A84 =                                  320.*E8 /2048.
    
    GCLAT = (90-theta)
    SCL = np.sin(np.radians(GCLAT))
    
    RI = a/r
    A2 = RI*(A21 + RI * (A22 + RI* A23))
    A4 = RI*(A41 + RI * (A42 + RI*(A43+RI*A44)))
    A6 = RI*(A61 + RI * (A62 + RI* A63))
    A8 = RI*(A81 + RI * (A82 + RI*(A83+RI*A84)))
    
    CCL = np.sqrt(1-SCL**2)
    S2CL = 2.*SCL  * CCL
    C2CL = 2.*CCL  * CCL-1.
    S4CL = 2.*S2CL * C2CL
    C4CL = 2.*C2CL * C2CL-1.
    S8CL = 2.*S4CL * C4CL
    S6CL = S2CL * C4CL + C2CL * S4CL
    
    DLTCL = S2CL * A2 + S4CL * A4 + S6CL * A6 + S8CL * A8
    gdlat = DLTCL + np.radians(GCLAT)
    height = r * np.cos(DLTCL)- a * np.sqrt(1 -  E2 * np.sin(gdlat) ** 2)


    # magnetic components 
    theta_rad = np.radians(theta)
    psi = np.sin(gdlat) * np.sin(theta_rad) - np.cos(gdlat) * np.cos(theta_rad)
    
    # Convert gdlat to degrees
    gdlat = np.degrees(gdlat)
    
    if matrix:
        T = np.array([[1, 0, 0], 
                     [0, -np.cos(psi), -np.sin(psi)], 
                     [0, -np.sin(psi), np.cos(psi)]])
        return gdlat, height, T
    
    Bn = -np.cos(psi) * B_th - np.sin(psi) * B_r 
    Bu = -np.sin(psi) * B_th + np.cos(psi) * B_r 
    return gdlat, height, Bn, Bu
