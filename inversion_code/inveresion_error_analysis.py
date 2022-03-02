
""" run through a whole pass, and save the output in xarray format
"""

import numpy as np
import datetime as dt
import xarray as xr
from datetime import timedelta as td
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices, CSgrid, CSprojection
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dipole import geo2mag
from secsy import spherical 
from simulation_utils import get_MHD_jeq, get_MHD_dB, get_MHD_dB_new
import pandas as pd
import os
import new_cases

errors = np.empty((0, 37))

for case in new_cases.cases.keys():
    info = new_cases.cases[case]

    fn = 'inversion_results/' + info["outputfn"] + ".netcdf"

    ds = xr.load_dataset(fn)
    
    mhdBu = info['mhdfunc'](ds.lat.values.flatten(), ds.lon.values.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
    mhdBu = mhdBu.reshape(ds.Bu.values.shape)

    print(np.sqrt(np.mean((mhdBu - ds.Bu.values)**2, axis = 1)).shape)

    errors = np.vstack((errors, np.sqrt(np.mean((mhdBu - ds.Bu.values)**2, axis = 1))))


