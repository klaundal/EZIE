""" 
get_observation_points - a function that allows you to calculate the points in the mesosphere that
intersect look directions which are perpendicular to satellite velocity vector, with some given angle off nadir.
Useful for sampling synthetic data from simulation output.

enu_to_ecef(v, lon, lat, reverse = False)
ecef_to_enu(v, lon, lat)
get_observation_points(dates, tlefile = 'swarmc.tle', id = 'SWARMC')

"""

import numpy as np
import datetime as dt
import pathlib
from orbit_predictor.sources import EtcTLESource # needed for tle -> orbit info

RE = 6371.2
d2r = np.pi/180

path = str(pathlib.Path(__file__).parent.absolute())

def enu_to_ecef(v, lon, lat, reverse = False):
    """ convert vector(s) v from ENU to ECEF (or opposite)

    Parameters
    ----------
    v: array
        N x 3 array of east, north, up components
    lat: array
        N array of latitudes (degrees)
    lon: array
        N array of longitudes (degrees)
    reverse: bool (optional)
        perform the reverse operation (ecef -> enu). Default False

    Returns
    -------
    v_ecef: array
        N x 3 array of x, y, z components


    Author: Kalle, March 2020
    """

    # construct unit vectors in east, north, up directions:
    ph = lon * d2r
    th = (90 - lat) * d2r

    e = np.vstack((-np.sin(ph)             ,               np.cos(ph), np.zeros_like(ph))).T # (N, 3)
    n = np.vstack((-np.cos(th) * np.cos(ph), -np.cos(th) * np.sin(ph), np.sin(th)       )).T # (N, 3)
    u = np.vstack(( np.sin(th) * np.cos(ph),  np.sin(th) * np.sin(ph), np.cos(th)       )).T # (N, 3)

    # rotation matrices (enu in columns if reverse, in rows otherwise):
    R_EN_2_ECEF = np.stack((e, n, u), axis = 1 if reverse else 2) # (N, 3, 3)

    # perform the rotations:
    return np.einsum('nij, nj -> ni', R_EN_2_ECEF, v)


def ecef_to_enu(v, lon, lat):
    """ convert vector(s) v from ECEF to ENU

    Parameters
    ----------
    v: array
        N x 3 array of x, y, z components
    lat: array
        N array of latitudes (degrees)
    lon: array
        N array of longitudes (degrees)

    Returns
    -------
    v_ecef: array
        N x 3 array of east, north, up components

    See enu_to_ecef for implementation details
    """
    return enu_to_ecef(v, lon, lat, reverse = True)


def get_satellite_state(dates, tlefile = path + '/tle/swarmc.tle', id = 'SWARMC'):
    """ get satellite position and velocity for a at given time from tle file

    Parameters
    ----------
    dates: datetime or list/array of datetimes
        The time(s) of interest. Can be iterable, in which case
        a corresponding number of return parameters are produced
    tlefile: string
        name and path of tle file
    id: string
        ID of spacecraft, must match information in tle file

    Returns
    -------
    r: array
        radius at satellite position
    theta: array
        colatitudes/polar angle at satellite position (degrees)
        geocentric coordinates
    phi: array
        longitude at satellite position (degrees) geocentric 
        coordinates
    v: array
        (N, 3) array of satellite velocity in geocentric ENU coordinates.
        N is the number of datetimes passed to the function
    """

    if not hasattr(dates, '__iter__'):
        assert type(dates) == type(dt.datetime(2000, 1, 1))
        dates = [dates]

    source = EtcTLESource(filename = tlefile)
    predictor = source.get_predictor(id)

    # get poisition in ECEF coords:
    r_ecef = np.array([predictor.get_position(date).position_ecef for date in dates]).T
    v_ecef = np.array([predictor.get_position(date).velocity_ecef for date in dates]).T

    # calculate spherical/geocentric coords:
    r     = np.linalg.norm(r_ecef, axis = 0)
    phi   = np.arctan2(r_ecef[1], r_ecef[0]) / d2r
    theta = np.arccos(r_ecef[2] / r) / d2r

    # calculate ENU components of velocity:
    R = np.stack((np.vstack((-                      np.sin(phi * d2r),                        np.cos(phi * d2r), np.zeros_like(phi) )),
                  np.vstack((-np.cos(theta * d2r) * np.cos(phi * d2r), -np.cos(theta * d2r) * np.sin(phi * d2r), np.sin(theta * d2r))),
                  np.vstack(( np.sin(theta * d2r) * np.cos(phi * d2r),  np.sin(theta * d2r) * np.sin(phi * d2r), np.cos(theta * d2r))) ), axis = 0)
    v_enu = np.einsum('jik, ik->kj', R, v_ecef)

    return tuple(map(np.squeeze, [r, theta, phi, v_enu]))



def get_observation_points(r, theta, phi, v, alphas = [45, -5, -25, -45], h = 85):
    """ for a given position and orientation (v), find the intersection with 
        h = H_m for four beams that are perpenicular to v and tilted alpha 
        degrees out from -r

        Note: Ignoring ellipsoidal Earth - assuming it is spherical with 
        radius 6371.2 km - also assuming constant zero yaw and roll

        v must be (N, 3)
        r, theta, phi must be (N)

        returns lat, lon of the observation points - shapes are (len(alphas), len(r))

    """
    v = (v.T / np.linalg.norm(v, axis = 1)).T #normalize

    # t is a horizontal port side unit vector 
    t = np.cross(np.array([0, 0, 1]), v) # ENU

    # n is perpendicular to v and t (pointing roughly downward)
    n = np.cross(t, v) # ENU

    # rv is a vector pointing at sc from Earth center - described in ENU
    rv = r[:, np.newaxis] * np.array([0, 0, 1])[np.newaxis, :]

    # a are K unit vectors pointing along view directions (K is number of angles)
    # shape is (K, N, 3)
    alpha = np.array(alphas).ravel()[:, np.newaxis, np.newaxis] * d2r
    a = np.cos(alpha) * n[np.newaxis, :, :] + np.sin(alpha) * t[np.newaxis, :, :]

    # s are distances between satellite and R = RE + h along a - shape (K, N)
    # Note:
    # This is the solution to a quadratic equation that appears from setting
    # rv + a * s = RE + h (and selecting the solution corresponding to the 
    # closest point of intersection with the sphere, hence the minus sign)
    ar = np.sum(a * rv[np.newaxis, :, :], axis = -1) # a dot rv
    s  = -ar - np.sqrt(ar**2 + (RE + h)**2 - r[np.newaxis, :]**2)

    # ra are vectors pointing at observation points - in ENU coords
    ra = rv[np.newaxis, :, :] + a * s[:, :, np.newaxis]
    ra = ra / np.linalg.norm(ra, axis = 2)[:, :, np.newaxis] 

    # convert ra to ecef:
    ph = phi * d2r
    th = theta * d2r
    east  = np.vstack((-             np.sin(ph),               np.cos(ph), np.zeros_like(ph))).T # (N, 3)
    north = np.vstack((-np.cos(th) * np.cos(ph), -np.cos(th) * np.sin(ph), np.sin(th)       )).T # (N, 3)
    up    = np.vstack(( np.sin(th) * np.cos(ph),  np.sin(th) * np.sin(ph), np.cos(th)       )).T # (N, 3)
    R_ENUtoECEF = np.stack((east, north, up), axis = 2) # (N, 3, 3)
    ra_ecef = np.einsum('nij, anj -> ani', R_ENUtoECEF, ra) # (K, N, 3)


    # extract coordinates:
    theta = np.arccos( ra_ecef[:, :,  2] ) / d2r
    phi   = np.arctan2(ra_ecef[:, :,  1] , ra_ecef[:, :, 0]) / d2r

    return phi, 90 - theta



def get_dipole_unit_b(lat, r):
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

    B = np.sqrt(Be**2 + Bn**2 + Bu**2)

    return Be/B, Bn/B, Bu/B




if __name__ == '__main__':

    # TESTING ENU/ECEF CONVERSION:
    v = np.array([[1, 1, 0], [1, 0, 0]])
    lat = np.array([-90, 0])
    lon = np.array([0., 0])
    print(enu_to_ecef(v, lat, lon))

    v = (np.random.random((30, 3)) - .5)*300
    lat = (np.random.random(30) - .5) * 180
    lon = np.random.random(30) * 360
    print('This number should be small: ', np.max(enu_to_ecef(enu_to_ecef(v, lat, lon), lat, lon, reverse = True) - v)**2)

    t = [dt.datetime(2000, 1, 2, 3, 23), dt.datetime(2001, 3, 3, 10, 21)]
    r, theta, phi, v = get_satellite_state(t)
    print(v.shape)
    print(v)