import numpy as np
import pandas as pd
from scipy.interpolate import RectSphereBivariateSpline

d2r = np.pi / 180

def get_MHD_dB_new(lat, lon, component = 'Br [nT]', fn = '../data/proposal_stage_mhd_data/gamera_dBs_Jfull_80km_2430'):
    """
    """
    lon, lat = np.array(lon), np.array(lat)
    lon = (lon) % 360


    names = ['R [km]', 'theta [radians]', 'phi [radians]', 'Br [nT]', 'Btheta [nT]', 'Bphi [nT]']
    table = pd.read_table(fn, sep = ' ', skipinitialspace = True, skiprows=[0], index_col = None, names = names)
    table['B'] = np.sqrt(table['Br [nT]']**2 + table['Btheta [nT]']**2 + table['Bphi [nT]']**2)
    table['lat'] = table['theta [radians]'] 
    table['phi'] = table['phi [radians]'] 
    print(table['lat'].max(), table['lat'].min())
    #table = table.sort_values(['lat'])
    #table['phi'] = -table.phi + np.pi

    shape = (np.unique(table.phi).size, np.unique(table.lat).size)

    lo = table.phi.values.reshape(shape)[:, 0]#[0, :]
    la = table.lat.values.reshape(shape)[0, :]#[:, 0]


    getB = RectSphereBivariateSpline(la[::-1],
                                     lo,
                                     table[component].values.reshape(shape)[:, ::-1].T, s = 0).ev

    return getB(lat * d2r, lon * d2r)

def get_MHD_dB(lat, lon, component = 'Br [nT]', fn = '../data/proposal_stage_mhd_data/gamera_dBs_Jfull_80km_2430'):
    """
    """
    lon = (lon) % 360


    names = ['R [km]', 'theta [radians]', 'phi [radians]', 'Br [nT]', 'Btheta [nT]', 'Bphi [nT]']
    table = pd.read_table(fn, sep = ' ', skipinitialspace = True, skiprows=[0], index_col = None, names = names)
    table['B'] = np.sqrt(table['Br [nT]']**2 + table['Btheta [nT]']**2 + table['Bphi [nT]']**2)
    table['lat'] = table['theta [radians]'] 
    table['phi'] = table['phi [radians]'] 
    #table = table.sort_values(['lat'])
    #table['phi'] = -table.phi + np.pi

    shape = (np.unique(table.lat).size, np.unique(table.phi).size)

    lo = table.phi.values.reshape(shape)[0, :]
    la = table.lat.values.reshape(shape)[:, 0]


    getB = RectSphereBivariateSpline((np.pi/2 - la)[::-1],
                                     lo,
                                     table[component].values.reshape(shape)[::-1], s = 0).ev

    return getB(lat * d2r, lon * d2r)


def get_MHD_jeq(lat, lon, fn = '../data/proposal_stage_mhd_data/Jequiv'):
    """

    """

    lon = (lon) % 360

    names = ['R [km]', 'theta [radians]', 'phi [radians]', 'Jr [A/m]', 'Jtheta [A/m]', 'Jphi [A/m]']
    table = pd.read_table(fn, sep = ' ', skipinitialspace = True, skiprows=[0], index_col = None, names = names)
    table['lat'] = table['theta [radians]'] 
    table['phi [radians]'][table['phi [radians]'] < 0] += 2*np.pi 
    ph = table['phi [radians]']
    ph = np.round(ph * 2 * 180 / np.pi)/2
    ph[ph == 360] = 0
    table['phi'] = ph * np.pi / 180
    table['lat'] = np.round(table['lat'] * 180 / np.pi * 2) / 2 * np.pi / 180
    #table = table.sort_values(['lat'])
    #table['phi'] = -table.phi + np.pi

    shape = (np.unique(table.lat).size, np.unique(table.phi).size)

    lo = table.phi.values.reshape(shape)[0, :]
    la = table.lat.values.reshape(shape)[:, 0]


    getJe = RectSphereBivariateSpline((np.pi/2 - la)[::-1],
                                      lo,
                                      table['Jphi [A/m]'].values.reshape(shape)[::-1], s = 0).ev

    getJs = RectSphereBivariateSpline((np.pi/2 - la)[::-1],
                                      lo,
                                      table['Jtheta [A/m]'].values.reshape(shape)[::-1], s = 0).ev


    return getJe(lat * d2r, lon * d2r), -getJs(lat * d2r, lon * d2r)
