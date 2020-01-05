""" 
Get the matrix representing the system of equations relating horizontal convection measurements to spherical elementary convection cells

EXAMPLE:
from scipy.linalg import lstsq

# GET:
mlat_m    # measurement
mlt_m     # measurement
bearing_m # measurement (radians)
v_los     # measurement

# DEFINE:
mlat_secc # secc grid
mlt_secc  # secc grid

# SOLVE
Ge, Gn = get_SECC_G_matrix(mlat_m, mlt_m, mlat_secc, mlt_secc)
G = Ge * np.sin(bearing_m) + Gn * np.cos(bearing_m)
V = lstsq(G, v_los, cond = 0.05)[0]

# PLOT
mlat_p # plot grid
mlt_p  # plot grid

Ge, Gn = get_SECC_G_matrix(mlat_p, mlt_p, mlat_secc, mlt_secc)

ve = Ge.dot(V)
vn = Gn.dot(V)

... and plot ve, vn at mlat_p, mlt_p

"""

import numpy as np
d2r = np.pi/180
MU0 = 4 * np.pi * 1e-7
RE = 6371.2 * 1e3

def get_SECS_J_G_matrices(mlat, mlt, mlat_secc, mlt_secc, type = 'divergence_free', constant = 1./(4*np.pi), RI = RE + 130 * 1e3):
    """ return the G matrices - Ge and Gn, which relate the SECC intensities at mlat_secc, mlt_secc to the convection velocities at mlat, mlt:

        j_e = Ge.dot(V)
        j_n = Gn.dot(V) 

    """
    
    # convert mlat, mlt to to column vectors, and mlat_secc, mlt_secc to row vectors:
    mlat      = np.array(mlat).flatten()[:, np.newaxis]
    mlt       = np.array(mlt).flatten()[:, np.newaxis]
    mlat_secc = np.array(mlat_secc).flatten()[np.newaxis, :]
    mlt_secc  = np.array(mlt_secc).flatten()[ np.newaxis, :]

    # ECEF position vectors of data points - should be N by 3, where N is number of data points
    ecef_r_data = np.hstack(( np.cos(mlat * d2r)      * np.cos(mlt * np.pi/12)     , np.cos(mlat * d2r)      * np.sin(mlt * np.pi/12)     , np.sin(mlat * d2r) ))
    
    # position vectors SECCs - should be 3 by M, where M is number of SECCs - these are the z axes of each SECC system
    ecef_r_secc = np.vstack(( np.cos(mlat_secc * d2r) * np.cos(mlt_secc * np.pi/12), np.cos(mlat_secc * d2r) * np.sin(mlt_secc * np.pi/12), np.sin(mlat_secc * d2r) )).T
    
    # unit vector pointing from SECC to magnetomer - (M, N, 3) 
    ecef_t = ecef_r_secc[np.newaxis, :, :] - ecef_r_data[:, np.newaxis, :] # difference vector - not tangential yet
    ecef_t = ecef_t - np.einsum('ijk,ik->ij', ecef_t, ecef_r_data)[:, :, np.newaxis] * ecef_r_data[:, np.newaxis, :] # subtract radial part of the vector to make it tangential
    ecef_t = ecef_t/np.linalg.norm(ecef_t, axis = 2)[:, :, np.newaxis] # normalize the result
        
    # make N rotation matrices to rotate ecef_t to enu_t - one rotation matrix per SECC:
    R = np.hstack( (np.dstack((-np.sin(mlt * np.pi/12)                      ,  np.cos(mlt * np.pi/12)                     , np.zeros_like(mlat) )),
                    np.dstack((-np.cos(mlt * np.pi/12)  * np.sin(mlat * d2r), -np.sin(mlt * np.pi/12) * np.sin(mlat * d2r), np.cos(mlat * d2r)  )),
                    np.dstack(( np.cos(mlt * np.pi/12)  * np.cos(mlat * d2r),  np.sin(mlt * np.pi/12) * np.cos(mlat * d2r), np.sin(mlat * d2r)  ))) )

    # apply rotation matrices to make enu vectors pointing from data points to SECCs
    enu_t = np.einsum('lij, lkj->lki', R, ecef_t)[:, :, :-1] # remove last component (up), which should deviate from zero only by machine precicion
    
    if type == 'divergence_free':
        # rotate these vectors to get vectors pointing eastward with respect to SECC systems at each data point
        enu_vec = np.dstack((enu_t[:, :, 1], -enu_t[:, :, 0])) # north -> east and east -> south
    elif type == 'curl_free':
        enu_vec = -enu_t # outward from SECC
    elif type in ['potential', 'scalar']:
        enu_vec = 1
    else:
        raise Exception('type must be "divergence_free", "curl_free", "potential", or "sclar"')


    # get the scalar part of Amm's divergence-free SECS:    
    theta = np.arccos(np.einsum('ij,kj->ik', ecef_r_secc, ecef_r_data))
    if type in ['divergence_free', 'curl_free']:
        coeff = constant/np.tan(theta/2)/(RI)
        # G matrices
        Ge = coeff * enu_vec[:, :, 0].T
        Gn = coeff * enu_vec[:, :, 1].T
    
        return Ge.T, Gn.T
    else: # type is 'potential' or 'scalar'
        if type == 'potential':
            return -2*constant*np.log(np.sin(theta/2)).T
        elif type == 'scalar':
            return    constant      / np.tan(theta/2).T
        


def get_theta(mlat, mlt, mlat_secs, mlt_secs):
    """" calculate theta angle - the angle between data point and secs node. 
        Output will be a 2D array with shape (mlat.size, mlat_secs.size)
    """

    # convert mlat, mlt to to column vectors, and mlat_secs, mlt_secs to row vectors:
    mlat      = np.array(mlat).flatten()[:, np.newaxis]
    mlt       = np.array(mlt).flatten()[:, np.newaxis]
    mlat_secs = np.array(mlat_secs).flatten()[np.newaxis, :]
    mlt_secs  = np.array(mlt_secs).flatten()[ np.newaxis, :]

    # ECEF position vectors of data points - should be N by 3, where N is number of data points
    ecef_r_data = np.hstack(( np.cos(mlat * d2r)      * np.cos(mlt * np.pi/12)     , np.cos(mlat * d2r)      * np.sin(mlt * np.pi/12)     , np.sin(mlat * d2r) ))

    # position vectors for the SECS - should be 3 by M, where M is number of SECCs - these are the z axes of each SECC system
    ecef_r_secs = np.vstack(( np.cos(mlat_secs * d2r) * np.cos(mlt_secs * np.pi/12), np.cos(mlat_secs * d2r) * np.sin(mlt_secs * np.pi/12), np.sin(mlat_secs * d2r) )).T

    # the polar angles (N, M):
    theta = np.arccos(np.einsum('ij, kj -> ik', ecef_r_data, ecef_r_secs))

    return theta



def get_SECS_B_G_matrices(mlat, mlt, mlat_secs, mlt_secs, r, RI = 6371.2 + 110., current_type = 'divergence_free'):
    """ calculate Ge, Gn, Gu, the G matrices relating divergence free spherical elementary
        current systems at (mlat_secs, mlt_secs) to the east, north, an d upward components
        of the magnetic field at (mlat, mlt)

        Based on equations (9) and (10) of Amm and Viljanen, 1999
    """

    # convert mlat, mlt to to column vectors, and mlat_secs, mlt_secs to row vectors:
    mlat      = np.array(mlat).flatten()[:, np.newaxis]
    mlt       = np.array(mlt).flatten()[:, np.newaxis]
    mlat_secs = np.array(mlat_secs).flatten()[np.newaxis, :]
    mlt_secs  = np.array(mlt_secs).flatten()[ np.newaxis, :]
    r         = np.array(r).flatten()[:, np.newaxis]

    # ECEF position vectors of data points - should be N by 3, where N is number of data points
    ecef_r_data = np.hstack(( np.cos(mlat * d2r)      * np.cos(mlt * np.pi/12)     , np.cos(mlat * d2r)      * np.sin(mlt * np.pi/12)     , np.sin(mlat * d2r) ))

    # position vectors for the SECS - should be 3 by M, where M is number of SECCs - these are the z axes of each SECC system
    ecef_r_secs = np.vstack(( np.cos(mlat_secs * d2r) * np.cos(mlt_secs * np.pi/12), np.cos(mlat_secs * d2r) * np.sin(mlt_secs * np.pi/12), np.sin(mlat_secs * d2r) )).T

    # the polar angles (N, M):
    theta = np.arccos(np.einsum('ij, kj -> ik', ecef_r_data, ecef_r_secs))

    # G matrix for upward field component
    if current_type == 'divergence_free':
        # G matrix for local northward component   
        Gu = MU0/(4 * np.pi * r) * (1 / np.sqrt(1 - 2 * r * np.cos(theta) / RI + (r/RI)**2 ) - 1)
        Gn_ =  MU0/(4 * np.pi * r * np.sin(theta)) * ((r/RI - np.cos(theta)) / np.sqrt(1 - 2 * r * np.cos(theta) / RI + (r/RI)**2 ) + np.cos(theta)) 
    elif current_type == 'curl_free':
        # G matrix for local eastward component
        Ge_ = -MU0/(4 * np.pi * r) / np.tan(theta / 2)
        Ge_[r.flatten() < RI, :]*= 0 # no magnetic field under current sheet

    # Now make matrices that can be used to transform Gn_ into east/north components in a global coordinate system
    # unit vector pointing from SECS to magnetomer - (M, N, 3) 
    ecef_t = ecef_r_secs[np.newaxis, :, :] - ecef_r_data[:, np.newaxis, :] # difference vector - not tangential yet
    ecef_t = ecef_t - np.einsum('ijk,ik -> ij', ecef_t, ecef_r_data)[:, :, np.newaxis] * ecef_r_data[:, np.newaxis, :] # subtract radial part of the vector to make it tangential
    ecef_t = ecef_t/np.linalg.norm(ecef_t, axis = 2)[:, :, np.newaxis] # normalize the result
        
    # make N rotation matrices to rotate ecef_t to enu_t - one rotation matrix per SECC:
    R = np.hstack( (np.dstack((-np.sin(mlt * np.pi/12)                      ,  np.cos(mlt * np.pi/12)                     , np.zeros_like(mlat) )),
                    np.dstack((-np.cos(mlt * np.pi/12)  * np.sin(mlat * d2r), -np.sin(mlt * np.pi/12) * np.sin(mlat * d2r), np.cos(mlat * d2r)  )),
                    np.dstack(( np.cos(mlt * np.pi/12)  * np.cos(mlat * d2r),  np.sin(mlt * np.pi/12) * np.cos(mlat * d2r), np.sin(mlat * d2r)  ))) )

    # apply rotation matrices to make enu vectors pointing from data points to SECCs
    enu_t = np.einsum('lij, lkj -> lki', R, ecef_t)[:, :, :-1] # remove last component (up), which should deviate from zero only by machine precicion
    
    if current_type == 'divergence_free':
        Ge =  Gn_ * enu_t[:, :, 0]
        Gn =  Gn_ * enu_t[:, :, 1]
    elif current_type == 'curl_free':
        Ge = -Ge_ * enu_t[:, :, 1] # eastward component of enu_t is northward component of enu_e (unit vector in local east direction)
        Gn =  Ge_ * enu_t[:, :, 0] # northward component of enu_t is eastward component of enu_e
        Gu =  Ge_ * 0              # no radial component

    return Ge, Gn, Gu




if __name__ == '__main__':


    print( """ testing the field of a divergence-free current at mlat, mlt = 73., 13., with amplitude 1, at the following points: (83, 3), (58, 22), (80, 18), (70, 12), (64, 13), (73., 13.1), at r = 85""")

    m = np.array([1.])
    r = 6371.1 + 85.
    RI = 6371.1 + 105
    mlat_secs, mlt_secs = np.array([73.]), np.array([13.])
    mlat, mlt = np.array([83., 58., 80., 70., 64, 73.]), np.array([3., 22., 18., 12., 13, 13.1])

    r_secs   = np.array( [np.cos(mlat_secs * d2r) * np.cos(mlt_secs * np.pi/12), np.cos(mlat_secs * d2r) * np.sin(mlt_secs * np.pi/12), np.sin(mlat_secs * d2r)]).reshape((3, 1))
    r_points = np.vstack((np.cos(mlat      * d2r) * np.cos(mlt      * np.pi/12), np.cos(mlat      * d2r) * np.sin(mlt      * np.pi/12), np.sin(mlat      * d2r)))
    theta = np.arccos( np.sum(r_secs * r_points, axis = 0) )

    # calculate Br and Btheta (local), using Amm and Viljanen equations 9 and 10
    Br  =  MU0 * m / (4 * np.pi * r                ) * (                      1  / np.sqrt( 1 - 2 * r * np.cos(theta) / RI + (r / RI) ** 2) - 1            )
    Bth = -MU0 * m / (4 * np.pi * r * np.sin(theta)) * ((r / RI - np.cos(theta)) / np.sqrt( 1 - 2 * r * np.cos(theta) / RI + (r / RI) ** 2) + np.cos(theta))
    B = np.sqrt(Br**2 + Bth**2)

    Ge, Gn, Gu = get_SECS_B_G_matrices(mlat, mlt, mlat_secs, mlt_secs, r, RI = RI)
    print( B * 1e9)
    print( np.sqrt(Ge.dot(m)**2 + Gn.dot(m)**2 + Gu.dot(m)**2) * 1e9)
    print( '--')
    print( Br * 1e9)
    print( Gu.dot(m) * 1e9)
    print( 'This should be 0:', Ge.dot(m)[-2] * 1e9)
    print( '%.4f should be very small compared to this %.4f' % (Gn.dot(m)[-1] * 1e9, Ge.dot(m)[-1] * 1e9))
    print( Bth/np.sqrt(Bth**2 + Br**2))



