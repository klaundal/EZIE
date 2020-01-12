from spacepy import pycdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from EZIE.secs import get_SECS_B_G_matrices, get_SECS_J_G_matrices
from EZIE.utils.polarsubplot import Polarsubplot
from EZIE.utils.coords import sph_to_sph, car_to_sph
from EZIE.utils.sunlight import subsol
from scipy.interpolate import RectSphereBivariateSpline, griddata
from scipy.spatial.distance import cdist

d2r = np.pi/180

SECS_STEP = 3 # place one SECS pole for every SECS_STEP measurement
MIN_SECS_DATA_DISTANCE = 1
VECTOR_SPACE = 73 # distance between plotted vectors (multiples of 30 seems to avoid singularities)
SMOOTH_STEPS = 1 # size of convolution kernel. Should be determined by how many data points are inside 40 km bins. 

COMPONENTWIZE = True

NOISE = 110
NOISE_BE = 880
NOISE_BN = 689
NOISE_BU = 107
LAMBDA = 1e-22 # daping parameters Tikhonov regularization

SHRINK = 1.

# load information about divergence-free current 
# calculated using an expansion of total current in terms of
# DF and CF currents. This is used for comparison with estimates
jedf = np.load('jedf.npy')
jndf = np.load('jndf.npy')
vvvlat = np.load('Vlat.npy')
vvvmlt = np.load('Vmlt.npy')
shape = (len(np.unique(vvvmlt)), len(np.unique(vvvlat)))

la = vvvlat.reshape(shape)[0  , ::-1]
lo = vvvmlt.reshape(shape)[:-1,    0]

# make functions to evaluate divergence-free current at arbirary point
# using inteprolation
getjedf = RectSphereBivariateSpline(la * d2r, lo * d2r * 15, np.roll(jedf.reshape(shape)[:-1, :].T[::-1, :], 180, axis = 1)).ev
getjndf = RectSphereBivariateSpline(la * d2r, lo * d2r * 15, np.roll(jndf.reshape(shape)[:-1, :].T[::-1, :], 180, axis = 1)).ev


# define EZIE viewing tracks
tracks = {}
tracks[0] = {'offset':(-500) / SHRINK, 'duplicate' : None} # at edges)
tracks[1] = {'offset':( 0  ) / SHRINK, 'duplicate' : None} # duplicate: if != None, indicates if 
tracks[2] = {'offset':( 150) / SHRINK, 'duplicate' : None} # the data at this track should be copied
tracks[3] = {'offset':( 400) / SHRINK, 'duplicate' : None} # from another (to reduce longitudinal gradient

# define SECS poles along tracks at certain distances from viewing tracks:
secs = {}
secs[0 ] = {'offset': ( -850 ) / SHRINK}
secs[1 ] = {'offset': ( -750 ) / SHRINK}
secs[2 ] = {'offset': ( -625 ) / SHRINK}
secs[3 ] = {'offset': ( -375 ) / SHRINK}
secs[4 ] = {'offset': ( -250 ) / SHRINK}
secs[5 ] = {'offset': ( -125 ) / SHRINK}
secs[6 ] = {'offset': ( 37.5 ) / SHRINK}
secs[7 ] = {'offset': ( 75.0 ) / SHRINK}
secs[8 ] = {'offset': ( 102.5) / SHRINK}
secs[9 ] = {'offset': ( 212.5) / SHRINK}
secs[10] = {'offset': ( 275  ) / SHRINK}
secs[11] = {'offset': ( 337.5) / SHRINK}
secs[12] = {'offset': ( 462.5) / SHRINK}
secs[13] = {'offset': ( 525  ) / SHRINK}
secs[14] = {'offset': ( 600  ) / SHRINK}

vector_spacing_x = [-600, -400, -300, -200, -100, 30, 60, 90, 120, 200, 250, 300, 350, 500]


DETECTION_HEIGHT = 85.

def get_B_dipole(lat, r, B0 = 3.12*1e-5):
    """ return east, north, up components of Earth's dipole field 
        output has shape like r * lat
    r: radius in m
    lat : latitude in degrees
    """
    B0 = 3.12*1e-5
    a = B0 * ((6371.2 * 1e3)/r)**3
    Be = np.zeros_like(lat * r)
    Bn =      a * np.cos(lat * np.pi/180)
    Bu = -2 * a * np.sin(lat * np.pi/180)

    return Be, Bn, Bu


def get_orbit_coord_converter(lat0, lon0, lat1, lon1):
    """ Make a function that converts from one spherical coordinate system to
        another, defined by the input parameters as follows: 

        Let r0 and r1 be position vectors at (lat0, lon0) and (lat1, lon1). The 
        new coordinate system will have r0 and r1 are in the prime meridional plane,
        and the point r0 + (r1 - r0)/2 points at the equator (new x axis). The north
        pole is perpendicular to this direction, towards r1. 

        The new coordinate system will allow for cylindrical projections of points
        near the orbit, without much distortion (unless the orbit segment is very long)

        The function returns a function which will do the coordinate transformation.
    """

    r0 = np.array((np.cos(lat0 * d2r) * np.cos(lon0 * d2r), 
                   np.cos(lat0 * d2r) * np.sin(lon0 * d2r), 
                   np.sin(lat0 * d2r)))
    r1 = np.array((np.cos(lat1 * d2r) * np.cos(lon1 * d2r), 
                   np.cos(lat1 * d2r) * np.sin(lon1 * d2r), 
                   np.sin(lat1 * d2r)))

    x = (r0 + (r1 - r0)/2) / np.linalg.norm(r0 + (r1 - r0)/2)
    y = np.cross(r1, r0) / np.linalg.norm(np.cross(r1, r0))
    z = np.cross(x, y)

    _, xcolat, xlon = car_to_sph(x)
    _, zcolat, zlon = car_to_sph(z)

    return lambda lat, lon: sph_to_sph(lat, lon, 90 - xcolat, xlon, 90 - zcolat, zlon, deg = True)

def ll2xy(lat, lon, R = 6371.2, inverse = False):
    """ convert from lat, lon to x, y, using Mercator projection 

        parameters
        ----------
        lat : array
            latitude in degrees
        lon : array
            longitude in degrees
        R : float, optional
            Earth radius (ignoring ellipsoid)
        inverse : bool, optional
            Set to true if inverse operation is wanated. Then
            input should be x and y, and output will be lat and lon

    """
    
    if inverse:
        x, y = lat, lon
        lon = x / d2r / R % 360
        lat = np.arctan(np.sinh(y / R)) / d2r
        return lat, lon
    else:
        londiff = (lon + 180) % 360 - 180
        x = R * londiff * d2r
        y = R * np.log(np.tan(np.pi/4 + lat/2 * d2r))
        return x, y



class orbit_plot(object):
    """ class to contain orbit plot """
    def __init__(self, lat, lon, times, ax, tracks):
        self.r0 = np.vstack((np.cos(lat * d2r) * np.cos(lon * d2r), 
                             np.cos(lat * d2r) * np.sin(lon * d2r),
                             np.sin(lat * d2r)))
        
        # make tangent and normal vectors to orbit in order to produce parallel tracks 
        t = (self.r0[:, 10:] - self.r0[:, :-10]) / np.linalg.norm(self.r0[:, 10:] - self.r0[:, :-10], axis = 0) # difference vector along orbit
        n = np.cross(t.T, self.r0[:, :-10].T).T/np.linalg.norm(np.cross(t.T, self.r0[:, :-10].T).T, axis = 0)
        self.n = n
        self.r0 = self.r0[:, :-10] # truncate position vector array so that it has same shape as the tangent and normal vectors
        lat = np.arcsin(self.r0[2]/np.linalg.norm(self.r0, axis = 0)) / d2r
        lon = np.arctan2(self.r0[1], self.r0[0]) / d2r

        # subsolar point is needed to calculate lt
        sslat, sslon = subsol(times[:-10])
        self.sslon = sslon
        lt  =  ((180 + lon - np.array(sslon))/15 - 12 )% 24 

        # make the coordinates of the tracks, and calculate the magnetic field measurement along these coordinates
        for i in range(len(tracks.keys())):
            tracks[i]['r']   = (6371.2 + DETECTION_HEIGHT) * 1e3 * self.r0 + n * tracks[i]['offset'] * 1e3
            tracks[i]['lat'] = np.arcsin(tracks[i]['r'][2]/np.linalg.norm(tracks[i]['r'], axis = 0)) / d2r
            tracks[i]['lon'] = np.arctan2(tracks[i]['r'][1], tracks[i]['r'][0]) / d2r
            tracks[i]['lt']  = ((180 + tracks[i]['lon'] - np.array(sslon))/15 - 12) % 24

        self.tracks = tracks

        # make conversion functions
        self.g_to_sat = get_orbit_coord_converter(lat[0], lt[0]*15, lat[-1], lt[-1]*15)
        xlat, xlon    = self.g_to_sat(np.array(0 ), np.array(0))
        zlat, zlon    = self.g_to_sat(np.array(90), np.array(0))
        self.sat_to_g = lambda la, lo: sph_to_sph(la, lo, xlat, xlon, zlat, zlon, deg = True)

        satlat, satlon = self.g_to_sat(lat, lt * 15)
        x, y = ll2xy(satlat, satlon)

        for i in range(len(self.tracks.keys())):
            satlat, satlon = self.g_to_sat(self.tracks[i]['lat'], self.tracks[i]['lt'] * 15)
            x, y = ll2xy(satlat, satlon)
            self.tracks[i]['x'] = x
            self.tracks[i]['y'] = y
            if self.tracks[i]['duplicate'] != None:
                continue
            ax.plot(x, y, color = 'C' + str(i), linestyle = '--', linewidth = 3, zorder = 2)


        ax.set_aspect('equal')

        # plot coordinate grid
        ylim = ax.get_ylim()
        yr = ylim[1]-ylim[0]
        ax.set_ylim(ylim[0] + .1 * yr, ylim[1] - .1 * yr)
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0] - 50, xlim[1] + 50)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        self.xx, self.yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
        latxx_, lonxx_ = ll2xy(self.xx, self.yy, inverse = True)
        self.latxx , self.lonxx  = self.sat_to_g(latxx_, lonxx_)

        self.ltxx = self.lonxx/15.

        latlevels = np.unique(np.int8(self.latxx/10) * 10)[1:]
        ltlevels  = np.unique(np.int8(self.ltxx/3  ) * 3 )

        contours = ax.contour(self.xx, self.yy, self.latxx.reshape(self.xx.shape), levels = latlevels, colors = 'lightgrey', linewidths = 1, zorder = 1)
        for c, val in zip(contours.collections, contours.levels):
            x, y = c.get_paths()[0].vertices.T
            if np.abs(x[0] - xlim[0]) < np.abs(x[0] - xlim[1]):
                ax.text(x[0], y[0], str(int(val)) + '$^\circ$', ha = 'right', va = 'center', color = 'grey')
            else:
                ax.text(x[0], y[0], str(int(val)) + '$^\circ$', ha = 'left', va = 'center', color = 'grey')



        laxx = np.linspace(-90, 90, 180)
        for l in ltlevels:
            la, lo = self.g_to_sat(laxx, np.ones_like(laxx) * l * 15)
            x, y = ll2xy(la, lo)
            x[(x < xlim[0]) | (x > xlim[1])] = np.nan
            ax.plot(x, y, color = 'lightgrey', linewidth = 1, zorder = 1)

        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim)
        ax.set_xlabel('km')
        ax.set_yticks([])

        plt.setp(ax.spines.values(), color = 'lightgrey')
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color = 'lightgrey')

        self.ax = ax








    
# read synthetic magnetic field data from file from Slava
names = ['db_p_mag', 'db_t_mag', 'db_r_mag', 'db_p_fac', 'db_t_fac', 'db_r_fac', 'db_p_ion', 'db_t_ion', 'db_r_ion', 'db_p_tot', 'db_t_tot', 'db_r_tot' ]
mhd = pd.read_table('AEM_LFM_data/deltaB_h' + str(int(DETECTION_HEIGHT)) + '_2016-08-09T09-48-00Z.txt', sep = ' ', skipinitialspace = True, skiprows=[0], index_col = [0, 1], names = names)


bu   = RectSphereBivariateSpline(np.r_[60.5:90:1] * d2r, np.r_[.5:360:1] * d2r , np.roll( mhd.db_r_tot.unstack().values.T[::-1, :], 180, axis = 1))
be   = RectSphereBivariateSpline(np.r_[60.5:90:1] * d2r, np.r_[.5:360:1] * d2r , np.roll( mhd.db_p_tot.unstack().values.T[::-1, :], 180, axis = 1))
bn   = RectSphereBivariateSpline(np.r_[60.5:90:1] * d2r, np.r_[.5:360:1] * d2r , np.roll(-mhd.db_t_tot.unstack().values.T[::-1, :], 180, axis = 1))

mhd_b = lambda lat, lon : np.vstack((be.ev(lat * d2r, lon * d2r) * 1e-9, bn.ev(lat * d2r, lon * d2r) * 1e-9, bu.ev(lat * d2r, lon * d2r) * 1e-9))

# read a swarm datafile to get some orbit coordinates and store in a dataframe
f = pycdf.CDF('swarm_data.cdf')
df = pd.DataFrame({'lat':f['Latitude'][:], 'lon':f['Longitude'][:], 'r':f['Radius'][:]}, index = f['Timestamp'][:])
df_orbit = df['2018-06-06 03:47':'2018-06-06 04:31']
df_orbit = df_orbit[df_orbit['lat'] > 60]
orbit_lat, orbit_lon = df_orbit['lat'].values, df_orbit['lon'].values + 180
_, orbit_sslon = subsol(df_orbit.index.to_pydatetime())
orbit_lt = ((180 + orbit_lon - np.array(orbit_sslon))/15 - 12 )% 24 


df = df['2018-06-06 03:57':'2018-06-06 04:01'][::2]
df.lon = df.lon + 180

mlatxx, mltxx = np.meshgrid(np.r_[60.5:90:1], np.r_[.5:360:1]/15)
dB_pax = np.sqrt(mhd.db_r_tot.unstack().values**2 + mhd.db_p_tot.unstack().values**2 + mhd.db_t_tot.unstack().values**2)[:, ::-1]

fig = plt.figure(figsize = (3.8, 12))
MLT_SHIFT = 12
ax = plt.subplot2grid((4, 1), (2, 0), rowspan = 2, colspan = 1)
oax = orbit_plot(df.lat.values, (df.lon.values - MLT_SHIFT*15) % 360, df.index.to_pydatetime(), ax, tracks)

# make secs coordinates:
for i in range(len(secs.keys())):
    secs[i]['r']   = (6371.2 + DETECTION_HEIGHT) * 1e3 * oax.r0[:, ::SECS_STEP] + oax.n[:, ::SECS_STEP] * secs[i]['offset'] * 1e3
    secs[i]['lat'] = np.arcsin(secs[i]['r'][2]/np.linalg.norm(secs[i]['r'], axis = 0)) / d2r
    secs[i]['lon'] = np.arctan2(secs[i]['r'][1], secs[i]['r'][0]) / d2r
    secs[i]['lt']  = ((180 + secs[i]['lon'] - np.array(oax.sslon[::SECS_STEP]))/15 - 12) % 24

## turn all into arrays
secs_lt  = np.hstack([secs[i]['lt']  for i in range(len(secs.keys()))])
secs_lat = np.hstack([secs[i]['lat'] for i in range(len(secs.keys()))])
sat_la, sat_lo = oax.g_to_sat(secs_lat, secs_lt * 15)
secsxx, secsyy = ll2xy(sat_la, sat_lo)

## prepare data arrays in individual tracks:
for key in oax.tracks.keys():
    oax.tracks[key]['dB']     = mhd_b(oax.tracks[key]['lat'], oax.tracks[key]['lt' ] * 15) # disturbance field (3, N)
    oax.tracks[key]['B0']     = get_B_dipole(oax.tracks[key]['lat'], (6371.2 + DETECTION_HEIGHT) * 1e3, B0 = 3.0e-5) # main field (3, N)
    oax.tracks[key]['B0_abs'] = np.linalg.norm(oax.tracks[key]['B0'], axis = 0) # main field magnitude (N)
    oax.tracks[key]['b0e'], oax.tracks[key]['b0n'], oax.tracks[key]['b0u'] = oax.tracks[key]['B0'] / oax.tracks[key]['B0_abs'] # unit vector components in direction of B0 (3, N)
    oax.tracks[key]['d'] = np.sum( (oax.tracks[key]['dB'] + oax.tracks[key]['B0']) * oax.tracks[key]['B0'] / oax.tracks[key]['B0_abs'], axis = 0) # full field projected on b_parallel (N)
    oax.tracks[key]['d'] = np.convolve(oax.tracks[key]['d'], np.ones(SMOOTH_STEPS)/SMOOTH_STEPS, mode = 'same') # smooth the data to account for overlapping pxiels
    oax.tracks[key]['noise'] = np.random.normal(scale = NOISE * 1e-9, size = oax.tracks[key]['d'].shape) # noise (N)
    oax.tracks[key]['data'] = oax.tracks[key]['d'] - oax.tracks[key]['B0_abs'] + oax.tracks[key]['noise'] # the measurement (N)


    oax.tracks[key]['de'] = np.convolve(oax.tracks[key]['B0'][0] + oax.tracks[key]['dB'][0], np.ones(SMOOTH_STEPS)/SMOOTH_STEPS, mode = 'same') # smooth the data to account for overlapping pxiels
    oax.tracks[key]['dn'] = np.convolve(oax.tracks[key]['B0'][1] + oax.tracks[key]['dB'][1], np.ones(SMOOTH_STEPS)/SMOOTH_STEPS, mode = 'same') # smooth the data to account for overlapping pxiels
    oax.tracks[key]['du'] = np.convolve(oax.tracks[key]['B0'][2] + oax.tracks[key]['dB'][2], np.ones(SMOOTH_STEPS)/SMOOTH_STEPS, mode = 'same') # smooth the data to account for overlapping pxiels
    oax.tracks[key]['noise_be'] = np.random.normal(scale = NOISE_BE * 1e-9, size = oax.tracks[key]['d'].shape) # noise east (N)
    oax.tracks[key]['noise_bn'] = np.random.normal(scale = NOISE_BN * 1e-9, size = oax.tracks[key]['d'].shape) # noise north (N)
    oax.tracks[key]['noise_bu'] = np.random.normal(scale = NOISE_BU * 1e-9, size = oax.tracks[key]['d'].shape) # noise up (N)

    oax.tracks[key]['data_e'] = oax.tracks[key]['de'] - oax.tracks[key]['B0'][0] + oax.tracks[key]['noise_be'] # the measurement in east (N)
    oax.tracks[key]['data_n'] = oax.tracks[key]['dn'] - oax.tracks[key]['B0'][1] + oax.tracks[key]['noise_bn'] # the measurement in north (N)
    oax.tracks[key]['data_u'] = oax.tracks[key]['du'] - oax.tracks[key]['B0'][2] + oax.tracks[key]['noise_bu'] # the measurement in up (N)



    oax.tracks[key]['data'] = oax.tracks[key]['d'] - oax.tracks[key]['B0_abs'] + oax.tracks[key]['noise'] # the measurement (N)

    oax.tracks[key]['data'] = oax.tracks[key]['data'][SMOOTH_STEPS//2:-SMOOTH_STEPS//2]
    oax.tracks[key]['data_e'] = oax.tracks[key]['data_e'][SMOOTH_STEPS//2:-SMOOTH_STEPS//2]
    oax.tracks[key]['data_n'] = oax.tracks[key]['data_n'][SMOOTH_STEPS//2:-SMOOTH_STEPS//2]
    oax.tracks[key]['data_u'] = oax.tracks[key]['data_u'][SMOOTH_STEPS//2:-SMOOTH_STEPS//2]
    oax.tracks[key]['b0e'] = oax.tracks[key]['b0e'][SMOOTH_STEPS//2:-SMOOTH_STEPS//2]
    oax.tracks[key]['b0n'] = oax.tracks[key]['b0n'][SMOOTH_STEPS//2:-SMOOTH_STEPS//2]
    oax.tracks[key]['b0u'] = oax.tracks[key]['b0u'][SMOOTH_STEPS//2:-SMOOTH_STEPS//2]
    oax.tracks[key]['lat'] = oax.tracks[key]['lat'][SMOOTH_STEPS//2:-SMOOTH_STEPS//2]
    oax.tracks[key]['lt' ] = oax.tracks[key]['lt' ][SMOOTH_STEPS//2:-SMOOTH_STEPS//2]
    oax.tracks[key]['noise'] = oax.tracks[key]['noise'][SMOOTH_STEPS//2:-SMOOTH_STEPS//2]

# merge the data arrays from individual passes:
data     = np.hstack([oax.tracks[key]['data'] for key in oax.tracks.keys()])
datae     = np.hstack([oax.tracks[key]['data_e'] for key in oax.tracks.keys()])
datan     = np.hstack([oax.tracks[key]['data_n'] for key in oax.tracks.keys()])
datau     = np.hstack([oax.tracks[key]['data_u'] for key in oax.tracks.keys()])
b0e      = np.hstack([oax.tracks[key]['b0e']  for key in oax.tracks.keys()])
b0n      = np.hstack([oax.tracks[key]['b0n']  for key in oax.tracks.keys()])
b0u      = np.hstack([oax.tracks[key]['b0u']  for key in oax.tracks.keys()])
data_lat = np.hstack([oax.tracks[key]['lat']  for key in oax.tracks.keys()])
data_lt  = np.hstack([oax.tracks[key]['lt']   for key in oax.tracks.keys()])


# make design matrix
Ge, Gn, Gu = get_SECS_B_G_matrices(data_lat, data_lt, secs_lat, secs_lt, (6371.2 + DETECTION_HEIGHT) * 1e3, RI = (6371.2 + 130.) * 1e3)
if COMPONENTWIZE:
    data = np.hstack((datae, datan, datau))
    G = np.vstack((Ge, Gn, Gu))
    w = 1/ np.hstack((np.ones(datae.size) * NOISE_BE, np.ones(datan.size) * NOISE_BN, np.ones(datau.size) * NOISE_BU))
    w = w / w.max()
    G = G * w[:, np.newaxis]
    data = data * w
else:
    G = Ge * b0e[:, np.newaxis] + Gn * b0n[:, np.newaxis] + Gu * b0u[:, np.newaxis] # design matrix for field-aligned perturbations

# solve
gtg = G.T.dot(G)
gtd = G.T.dot(data)
m = np.linalg.lstsq(gtg + np.identity(gtg.shape[0]) * LAMBDA, gtd, rcond = .0)[0]



xv, yv = np.meshgrid(vector_spacing_x, np.r_[oax.ax.get_ylim()[0]:oax.ax.get_ylim()[1]:VECTOR_SPACE])
satlat, satlon = ll2xy(xv, yv, inverse = True)
geolat, geolon = oax.sat_to_g(satlat, satlon)
Gje, Gjn = get_SECS_J_G_matrices(geolat.flatten(), geolon.flatten() /15, secs_lat, secs_lt, constant = 1/(4 * np.pi), RI = (6371.2 + 130) * 1e3)

je, jn = Gje.dot(m) * 1e3, Gjn.dot(m) * 1e3 # current in mA/m
oax.ax.quiver(xv, yv, je, jn, scale = 5000, width = .008, zorder = 6)

jedf = getjedf(geolat * d2r, geolon * d2r) * 1e3
jndf = getjndf(geolat * d2r, geolon * d2r) * 1e3
oax.ax.quiver(xv, yv, jedf, jndf, scale = 5000, width = .008, zorder = 0, color = 'black', alpha = .3)


GV = get_SECS_J_G_matrices(oax.latxx.flatten(), oax.ltxx.flatten(), secs_lat, secs_lt, type = 'potential', RI = (6371.2 + 105) * 1e3)
V = GV.dot(m)

# in order to avoid singularities, mask the points close to the SECS poles, and then fill in using interpolation
mindist = cdist(np.vstack((oax.xx.flatten(), oax.yy.flatten())).T, np.vstack((secsxx, secsyy)).T).min(axis = 1)
V[mindist < MIN_SECS_DATA_DISTANCE] = np.nan
V = V.reshape(oax.xx.shape)
V = np.ma.masked_invalid(V)
x1 = oax.xx[~V.mask]
y1 = oax.yy[~V.mask]
Vmasked = V[~V.mask]
V = griddata(np.vstack((x1, y1)).T, Vmasked.ravel(), np.vstack((oax.xx.flatten(), oax.yy.flatten())).T, method = 'cubic')

# plot dB
dBxx = np.linalg.norm(mhd_b(oax.latxx, oax.ltxx * 15), axis = 0)

pax = Polarsubplot(plt.subplot2grid((4, 1), (0, 0), rowspan = 1, colspan = 1), minlat = 60, linestyle = '-', linewidth = .2, color = 'black')
pax.contourf(mlatxx, mltxx + 12, dB_pax, levels = np.linspace(0, 1000, 11), extend = 'both')

# plot V in map
minlats = oax.latxx.reshape((100, 100)).min(axis = 1)
maxlats = oax.latxx.reshape((100, 100)).min(axis = 1)
laxx_ = oax.latxx.reshape((100, 100))[(minlats > 60) & (maxlats < 89), :]
loxx_ = oax.lonxx.reshape((100, 100))[(minlats > 60) & (maxlats < 89), :]

# plot the orbit plot boundaries on the polar plot
xs = np.linspace(oax.ax.get_xlim()[0], oax.ax.get_xlim()[1], 100)
ys = np.linspace(oax.ax.get_ylim()[0], oax.ax.get_ylim()[1], 100)

# top
lat, lon = ll2xy(xs, np.ones_like(xs) * oax.ax.get_ylim()[1], inverse  =True)
latp, lonp = oax.sat_to_g(lat, lon)
pax.plot(latp, lonp/15, color = 'black')
# bottom
lat, lon = ll2xy(xs, np.ones_like(xs) * oax.ax.get_ylim()[0], inverse  =True)
latp, lonp = oax.sat_to_g(lat, lon)
pax.plot(latp, lonp/15, color = 'black')
# left
lat, lon = ll2xy(np.ones_like(ys) * oax.ax.get_xlim()[0], ys, inverse  =True)
latp, lonp = oax.sat_to_g(lat, lon)
pax.plot(latp, lonp/15, color = 'black')
# right
lat, lon = ll2xy(np.ones_like(ys) * oax.ax.get_xlim()[1], ys, inverse  =True)
latp, lonp = oax.sat_to_g(lat, lon)
pax.plot(latp, lonp/15, color = 'black')


jdiff = np.sqrt((je - jedf)**2 + (jn - jndf)**2)

oax.ax.set_title('RMS error = %.0f mA/m' % np.mean(jdiff), size = 8, loc = 'right')

pax.plot(orbit_lat, orbit_lt - MLT_SHIFT, color = 'black')


ax_data = plt.subplot2grid((4, 1), (1, 0))
for key in np.sort(list(oax.tracks.keys())):
    ax_data.plot(np.ones_like(oax.tracks[key]['data']) * key * 500, 'k--', linewidth = .4)
    ax_data.plot((oax.tracks[key]['data'] - oax.tracks[key]['noise']) * 1e9 + key * 500, color = 'C' + str(key), linewidth = 2, zorder = 0)
    ax_data.plot(oax.tracks[key]['data'] * 1e9 + key * 500, color = 'C' + str(key), linewidth = 1)

ax_data.errorbar(-4, 1500, yerr = 250, barsabove = True, color = 'black', elinewidth = 1, capsize = 6)
ax_data.text(-5, 1500, '500nT', rotation = 90, ha = 'right', va = 'center', size = 6)
ax_data.text(0, 2000, 'Measured $\delta B_\parallel$', size = 12, ha = 'left', va = 'bottom')


ax_data.set_axis_off()

pax.writeMLTlabels(mlat = 60)



plt.tight_layout()

plt.show()


