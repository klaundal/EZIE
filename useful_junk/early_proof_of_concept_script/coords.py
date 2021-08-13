import numpy as np

d2r = np.pi/180
r2d = 180/np.pi

def sph_to_car(sph, deg = True):
    """ convert from spherical to cartesian coordinates

        input: 3 X N array:
           [r1    , r2    , ..., rN    ]
           [colat1, colat2, ..., colatN]
           [lon1  , lon2  , ..., lonN  ]

        output: 3 X N array:
           [x1, x2, ... xN]
           [y1, y2, ... yN]
           [z1, z2, ... zN]

        deg = True if lat and lon are given in degrees, 
              False if radians
    """

    r, theta, phi = sph

    if deg == False:
        conv = 1.
    else:
        conv = d2r


    return np.vstack((r * np.sin(theta * conv) * np.cos(phi * conv), 
                      r * np.sin(theta * conv) * np.sin(phi * conv), 
                      r * np.cos(theta * conv)))

def car_to_sph(car, deg = True):
    """ convert from cartesian to spherical coordinates

        input: 3 X N array:
           [x1, x2, ... xN]
           [y1, y2, ... yN]
           [z1, z2, ... zN]

        output: 3 X N array:
           [r1    , r2    , ..., rN    ]
           [colat1, colat2, ..., colatN]
           [lon1  , lon2  , ..., lonN  ]

        deg = True if lat and lon is wanted in degrees
              False if radians
    """

    x, y, z = car

    if deg == False:
        conv = 1.
    else:
        conv = r2d

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)*conv
    phi = ((np.arctan2(y, x)*180/np.pi) % 360)/180*np.pi * conv

    return np.vstack((r, theta, phi))


def sph_to_sph(lat, lon, x_lat, x_lon, z_lat, z_lon, deg = True):
    """ calculate the latitude and longitude in a spherical coordinate system
        with the north pole at pole_lat, pole_lon. lat, lon are latitude and 
        longitude in the original coordinate system

        the cooridnates of the new z and x axes must be given. They must describe
        orthogonal positions, otherwise an exception is raised

        parameters
        ----------
        lat : array
            latitude of the points that will be converted - will be flattened
        lon : array
            longitude of the points that will be converted - will be flattened
        x_lat : float
            latitude of the new x axis
        x_lon : float
            longitude of the new x axis
        z_lat : float
            latitude of the new z axis
        z_lon : float
            longitude of the new z axis
        deg : bool, optional
            True if input and output in degrees, False if radians.
            Default is True

        output
        ------
        latitude and longitude in the new coordinate system. These are arrays with the same
        size as lat and lon (although, shape is not conserved if input dimensions are > 1)
    """
    lat, lon = lat.flatten(), lon.flatten()

    if deg == False:
        conv = 1.
    else:
        conv = d2r

    xyz = np.vstack((np.cos(lat * conv) * np.cos(lon * conv), 
                     np.cos(lat * conv) * np.sin(lon * conv), 
                     np.sin(lat * conv)))

    new_z = np.array([np.cos(z_lat * conv) * np.cos(z_lon * conv), 
                      np.cos(z_lat * conv) * np.sin(z_lon * conv),
                      np.sin(z_lat * conv)                          ])
    new_x = np.array([np.cos(x_lat * conv) * np.cos(x_lon * conv), 
                      np.cos(x_lat * conv) * np.sin(x_lon * conv),
                      np.sin(x_lat * conv)                          ])
    new_y = np.cross(new_z, new_x, axisa = 0, axisb = 0, axisc = 0)
    new_x, new_y, new_z = new_x.flatten(), new_y.flatten(), new_z.flatten()

    # if new_y is not a unit vector, new_x and new_z are not orthogonal:
    if not np.isclose(np.linalg.norm(new_y), 1):
        raise ValueError('x and z coords do not describe orthogonal positions')

    # make rotation matrix and do the rotation
    R = np.vstack((new_x, new_y, new_z))
    XYZ = R.dot(xyz)

    # convert back to spherical
    _, COLAT, LON = car_to_sph(XYZ, deg = deg)

    return 90 - COLAT, LON







