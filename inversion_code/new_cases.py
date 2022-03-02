import datetime as dt
import numpy as np
import os
from simulation_utils import get_MHD_jeq, get_MHD_dB, get_MHD_dB_new


path = os.path.dirname(__file__) 


files     = [fn for fn in os.listdir(path + '/osse_pandas/') if fn.endswith('.pd')]
case_nums = ['case_' + fn.split('CASE')[1][:1] for fn in files]
mlts      = [fn.split('mlt_')[0][-2:] for fn in files]

times = {'case_1':dt.datetime(2023, 7, 4, 12, 18, 7),
         'case_2':dt.datetime(2023, 7, 4, 12, 26, 16),
         'case_3':dt.datetime(2023, 7, 4, 12, 25, 14),
         'case_4':dt.datetime(2023, 7, 4, 12, 25, 14)}

mhd_files = {'case_1':path + '/../data/OSSE_new/case_1/gamera_dBs_80km_2016-08-09T08_49_45.txt',
             'case_2':path + '/../data/OSSE_new/case_2/gamera_dBs_80km_2016-08-09T09_17_52.txt',
             'case_3':path + '/../data/OSSE_new/case_3/gamera_dBs_80km_2016-08-09T09_24_22.txt',
             'case_4':path + '/../data/OSSE_new/case_4/gamera_dBs_80km_2016-08-09T09_38_29.txt'}


cases = {}

for fn, case, mlt in zip(files, case_nums, mlts):
    key = case + '_mlt_' + mlt
    mapshift = int(mlt[-2:]) * 15
    cases[key] = {'filename':path + '/osse_pandas/' + fn,
                  'mhd_B_fn': mhd_files[case],
                  'mapshift': -mapshift,
                  'observation_height':80,
                  'output_path': 'new_events/',
                  'wshift':25,
                  'timeres':3,
                  'tm':times[case],
                  'mhdfunc':get_MHD_dB_new,
                  'clevels':np.linspace(-500, 500, 12),
                  'DT':6,
                  'signs':[-1, -1, 1],
                  'outputfn':key}
                 



