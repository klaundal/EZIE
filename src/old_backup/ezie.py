import numpy as np
from simulation_utils import get_MHD_dB
from utils import get_satellite_state, get_observation_points, get_dipole_unit_b
import datetime 
from cubedsphere import CSprojection, CSgrid
from secs_utils import get_SECS_B_G_matrices
from scipy.interpolate import RectBivariateSpline


d2r = np.pi / 180
OBSHEIGHT = 85

RESOLUTION = 25.


class EZIE(object):
    def __init__(self, tleinfo, date = datetime.datetime(2015, 2, 14, 15, 43, 40), 
                       view_angles = [45, -5, -22, -45], 
                       sampling_rate = 0.5):
        """ 

        Parameters
        ----------
        tleinfo: tuple of strings
            name of tle file (two-line element) + satellite id (both strings)
        date: datetime
            time 
        view_angles: iterable
            angles of look direction. The look directions are in the plane
            that is perpendicular to the satellite velocity. The angles 
            are relative to a vector that is the intersection of this plane
            with the plane spanned by the velocity and nadir
        sampling_rate: float
            Number of samples per second
        """
        self.tle, self.tleid = tleinfo
        self.date = date

        self.los_angles = np.array(view_angles).ravel() # line-of-sight angles

        # set up cubed sphere projection:
        self.r, self.theta, self.phi, self.v = get_satellite_state(self.date, tlefile = self.tle, id = self.tleid)
        vc = 2 * np.pi / (24 * 60**2) * self.r * np.sin(self.theta * d2r) # correction for Earth rotation
        self.projection = CSprojection((self.phi, 90 - self.theta), (self.v[0] - vc, self.v[1]))

        self.dt = datetime.timedelta(seconds = 1) /sampling_rate # seconds betwen samples


    def set_grid(self, dt, resolution = 15, extension = .2, GRID_H = 110, divisions = 2):
        """ Set up SECS grid by specifiying a time window and a spatial resolution
        
        Parameters
        ----------
        dt: float
            size of time window, in minutes
        resolution: float, optional
            Spatial resolution of SECS grid at center of grid, in km
            Default is 15 km.
        extension: float, optional
            How much that the grid should be extended in all directions
            in % of length covered by data 
        GRID_H: height of the grid, in km
        divisions: int, optional
            number of SECS nodes between each observation - default 3

        """
        R = 6371.2 + GRID_H

        timewindow = datetime.timedelta(seconds = dt * 60)

        # number samples in time window:
        nsteps = timewindow // self.dt

        # dates of the samples:
        dates = [self.date + self.dt * (i - nsteps//2) for i in range(nsteps)]

        # position and velocity of satellite at time of samples:
        r, theta, phi, v = get_satellite_state(dates, tlefile = self.tle, id = self.tleid)
        self.lon_fp = phi
        self.lat_fp = 90 - theta


        # the length of the grid is the distance between end points in projected coordinates:
        xi, eta = self.projection.geo2cube(phi, 90 - theta)
        L = (eta[-1] - eta[0]) * R * (1 + 2 * extension)

        # the width of the grid is given by the distance between the outer observation points
        op0 = get_observation_points(self.r.reshape((1)), self.theta.reshape((1)), self.phi.reshape((1)), self.v.reshape((1, -1)), alphas = self.los_angles, h = OBSHEIGHT)
        xi, eta = self.projection.geo2cube(op0[0], op0[1])
        xi = np.sort(xi.flatten())        
        W = (xi[-1] - xi[0]) * R * (1 + 2 * extension)


        # ready to set up the grid:
        self.grid = CSgrid(self.projection, L, W, float(resolution), float(resolution))

        self.lon_obs, self.lat_obs = get_observation_points(r, theta, phi, v, alphas = self.los_angles, h = OBSHEIGHT)
        self.xi_obs, self.eta_obs = self.projection.geo2cube(self.lon_obs, self.lat_obs)

        # get xi values for secs grid:
        xi_secs_slice = self.get_secs_xi(xi, divisions = divisions, extension = extension)
        
        # get eta values:
        deta = np.median(np.diff(self.eta_obs[0]))
        eta_secs_slice = np.r_[self.grid.eta.min():self.grid.eta.max()+.1e-6:deta]
        
        # and finally the secs grid:
        self.xi_secs, self.eta_secs = [x.flatten() for x in np.meshgrid(xi_secs_slice, eta_secs_slice, indexing = 'ij')]

        # calculate the geocentric coords of the secs grid:
        self.lon_secs, self.lat_secs = self.projection.cube2geo(self.xi_secs, self.eta_secs)

        # make interpolation matrix that interpolates from regular grid to secs points
        Q = np.empty((self.grid.xi.size, self.xi_secs.size))
        for i in range(self.xi_secs.size):
            M = np.zeros(self.xi_secs.size)
            M[i] = 1
            spl = RectBivariateSpline(xi_secs_slice, eta_secs_slice, M.reshape((xi_secs_slice.size, eta_secs_slice.size)))

            Q[:, i] = spl.ev(self.grid.xi.flatten(), self.grid.eta.flatten())

        self.Q = Q






    def get_data(self, lat, lon, height = '85', type = 'B'):
        """ get data from the look directions """
        return get_simulation_output(lat, lon, height, type)

    def get_secs_xi(self, a, divisions = 3, extension = .2):
        """ calculate optimal SECS grid, considering angles """

        a = np.sort(np.array(a).flatten())
        extent = a[-1] - a[0]
        a = np.hstack((a[0] - extent * extension, a, a[-1] + extent * extension))

        secs_xi = []
        for i in range(len(a) - 1):
            da = (a[i + 1] - a[i]) / (divisions + 1)
            for j in range(1, divisions + 1):
                secs_xi.append(a[i] + da * j)
        
        return np.array(secs_xi)
   


"""
if __name__ == '__main__':

    fig = plt.figure()
    ax  = fig.add_subplot(111)

    ezie = EZIE(tlefile, time)
    ezie.set_time_window(4)
    ezie.make_grid() 

    # plot map in FOV
    ezie.plotmap(ax)

    # plot magnetic field on map
    B = get_data(ezie.secsgrid.lat, ezie.secsgrid.lon)
    x, y = ezie.projection(ezie.secsgrid.lat, ezie.secsgrid.lon)
    ax.contourf(x, y, B)

    # make SECS inversion
    ezie.SECS_inversion()
"""
import matplotlib.pyplot as plt
ezie = EZIE(('./tle/swarmc.tle', 'SWARMC'), sampling_rate = 1)
ezie.set_grid(4, RESOLUTION)



fig = plt.figure(figsize = (24, 10))
ax_p  = fig.add_subplot(331)

# plot the grid
ax    = fig.add_subplot(333)
ax.scatter(ezie.xi_secs.flatten(), ezie.eta_secs.flatten())
ax.scatter(ezie.xi_obs.flatten(), ezie.eta_obs.flatten(), c = 'red')
ax.scatter(ezie.grid.xi.flatten(), ezie.grid.eta.flatten(), marker = '.', s = 1, c = 'black')
ax.plot(*ezie.projection.geo2cube(ezie.lon_fp, ezie.lat_fp), linewidth = 5, color = 'black', zorder = 8)

# plot the satellite FOVs with simulation output
views = []
for i in range(8):
    row = 1 + i*3 // 11
    col = i*3 % 12
    print(i, row, col, col + 1, '/n')
    views.append({'ax_model':plt.subplot2grid((3, 11), (row, col)),
                  'ax_fit':plt.subplot2grid((3, 11)  , (row, col + 1))})


for i in range(8):
    shift = i * 3 * 15
    dB = get_MHD_dB(ezie.grid.lat, (ezie.grid.lon + shift) % 360)
    print(dB.max())
    views[i]['ax_model'].contourf(ezie.grid.xi, ezie.grid.eta, dB, levels = np.linspace(0, 800, 21))

    #dB = get_MHD_dB(ezie.lat_secs, (ezie.lon_secs + shift)  % 360)
    #views[i]['ax_fit'].scatter(ezie.xi_secs, ezie.eta_secs, c = dB, vmin = 0, vmax = 800, s = 2)
    #dB_interp = ezie.Q.dot(dB).reshape(ezie.grid.xi.shape)
    #views[i]['ax_fit'].contourf(ezie.grid.xi, ezie.grid.eta, dB_interp, levels = np.linspace(0, 800, 21))

    Ge, Gn, Gu = get_SECS_B_G_matrices(ezie.lat_obs, ezie.lon_obs / 15, 
                                       ezie.lat_secs, (ezie.lon_secs  % 360)/15,
                                       6371.2 + OBSHEIGHT)
    be, bn, bu = list(map(lambda x: x.reshape((-1, 1)), get_dipole_unit_b(ezie.lat_obs, 6371.2 + OBSHEIGHT)))
    G = Ge*be + Gn*bn + Gu*bu
    dB_obs = get_MHD_dB(ezie.lat_obs.flatten(), (ezie.lon_obs.flatten() + shift)  % 360)
    m = np.linalg.lstsq(G, dB_obs, rcond = .1)[0]

    Goe, Gon, Gou = get_SECS_B_G_matrices(ezie.grid.lat.flatten(), ezie.grid.lon.flatten() / 15, 
                                          ezie.lat_secs, (ezie.lon_secs  % 360)/15,
                                          6371.2 + OBSHEIGHT)
    be, bn, bu = list(map(lambda x: x.reshape((-1, 1)), get_dipole_unit_b(ezie.grid.lat.flatten(), 6371.2 + OBSHEIGHT)))
    Go = Goe*be + Gon*bn + Gou*bu
    B = Go.dot(m)
    views[i]['ax_fit'].contourf(ezie.grid.xi, ezie.grid.eta, B.reshape(ezie.grid.eta.shape), levels = np.linspace(0, 800, 21))




plt.show()






plt.show()

