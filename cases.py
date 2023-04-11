import datetime as dt
import numpy as np
import os
from simulation_utils import get_MHD_jeq, get_MHD_dB, get_MHD_dB_new
path = os.path.dirname(__file__)

cases = {}

# PROPOSAL STAGE OSSE
cases['proposal_stage'] = {'filename':path + '/data/proposal_stage_sam_data/EZIE_event_simulation_ezie_simulation_case_1_look_direction_case_2_retrieved_los_mag_fields.pd',
        'mhd_B_fn':path + '/data/proposal_stage_mhd_data/gamera_dBs_Jfull_80km_2430',
        'mapshift':-210, # The MHD output was shifted by this amount when making synthetic data. The shift must be applied to my MHD readout functions
        'observation_height':80,
        'output_path':'final_figs/',
        'wshift':25,
        'tm':dt.datetime(2023, 7, 3, 2, 42, 22),
        'outputfn':'proposal_stage',
        'mhdfunc':get_MHD_dB,
        'clevels':np.linspace(-700, 700, 21),
        'DT':4,
        'signs':[1, 1, 1]} # multiply the OSSE components by this number to get consistent results (east, north up)


# OSSE CASE 1
cases['case_1'] = {'filename':path + '/data/OSSE_new/case_1/EZIE_event_simulation_CASE1_standard_EZIE_retrieved_los_mag_fields.pd',
        'mhd_B_fn':path + '/data/OSSE_new/case_1/gamera_dBs_80km_2016-08-09T08_49_45.txt',
        'mapshift':0, # The MHD output was shifted by this amount when making synthetic data. The shift must be applied to my MHD readout functions
        'observation_height':80,
        'output_path':'final_figs/',
        'wshift':25,
        'timeres':3,
        'tm':dt.datetime(2023, 7, 4, 12, 18, 7),
        'outputfn':'osse_case1',
        'mhdfunc':get_MHD_dB_new,
        'clevels':np.linspace(-300, 300, 12),
        'DT':6,
        'signs':[-1, -1, 1]} # multiply the OSSE components by this number to get consistent results (east, north up)

# OSSE CASE 2
cases['case_2'] = {'filename':path + '/data/OSSE_new/case_2/EZIE_event_simulation_CASE2_standard_EZIE_retrieved_los_mag_fields.pd',
        'mhd_B_fn':path + '/data/OSSE_new/case_2/gamera_dBs_80km_2016-08-09T09_17_52.txt',
        'mapshift':0, # The MHD output was shifted by this amount when making synthetic data. The shift must be applied to my MHD readout functions
        'observation_height':80,
        'output_path':'final_figs/',
        'wshift':25,
        'tm':dt.datetime(2023, 7, 4, 12, 26, 16),
        'outputfn':'osse_case2',
        'timeres':3,
        'mhdfunc':get_MHD_dB_new,
        'clevels':np.linspace(-500, 500, 12),
        'DT':6,
        'signs':[-1, -1, 1]} # multiply the OSSE components by this number to get consistent results (east, north up)

# OSSE CASE 3
cases['case_3'] = {'filename':path + '/data/OSSE_new/case_3/EZIE_event_simulation_CASE3_standard_EZIE_retrieved_los_mag_fields.pd',
        'mhd_B_fn':path + '/data/OSSE_new/case_3/gamera_dBs_80km_2016-08-09T09_24_22.txt',
        'mapshift':0, # The MHD output was shifted by this amount when making synthetic data. The shift must be applied to my MHD readout functions
        'observation_height':80,
        'output_path':'final_figs/',
        'wshift':25,
        'tm':dt.datetime(2023, 7, 4, 12, 25, 14),
        'outputfn':'osse_case3',
        'timeres':3,
        'mhdfunc':get_MHD_dB_new,
        'clevels':np.linspace(-300, 300, 12),
        'DT':6,
        'signs':[-1, -1, 1]} # multiply the OSSE components by this number to get consistent results (east, north up)

# OSSE CASE 4
cases['case_4'] = {'filename':path + '/data/OSSE_new/case_4/EZIE_event_simulation_CASE4_standard_EZIE_retrieved_los_mag_fields.pd',
        'mhd_B_fn':path + '/data/OSSE_new/case_4/gamera_dBs_80km_2016-08-09T09_38_29.txt',
        'mapshift':0, # The MHD output was shifted by this amount when making synthetic data. The shift must be applied to my MHD readout functions
        'observation_height':80,
        'output_path':'final_figs/',
        'wshift':25,
        'tm':dt.datetime(2023, 7, 4, 12, 25, 14),
        'outputfn':'osse_case4',
        'timeres':3,
        'mhdfunc':get_MHD_dB_new,
        'clevels':np.linspace(-300, 300, 12),
        'DT':6,
        'signs':[-1, -1, 1]} # multiply the OSSE components by this number to get consistent results (east, north up)


# OSSE CASE 4
cases['case_5'] = {'filename':path + '/data/OSSE_new/case_5/EZIE_event_simulation_CASE5_standard_EZIE_retrieved_los_mag_fields.pd',
        'mhd_B_fn':path + '/data/OSSE_new/case_4/gamera_dBs_80km_2016-08-09T09_38_29.txt',
        'mapshift':0, # The MHD output was shifted by this amount when making synthetic data. The shift must be applied to my MHD readout functions
        'observation_height':80,
        'output_path':'final_figs/',
        'wshift':25,
        'timeres':3,
        'tm':dt.datetime(2023, 7, 4, 12, 22, 13),
        'outputfn':'osse_case5',
        'mhdfunc':get_MHD_dB_new,
        'clevels':np.linspace(-300, 300, 12),
        'DT':6,
        'signs':[-1, -1, 1]} # multiply the OSSE components by this number to get consistent results (east, north up)
