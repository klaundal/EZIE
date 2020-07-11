from scipy.io.idl import readsav
import numpy as np
import pandas as pd

def get_data(fn = 'sam_data/EZIE_event_simulation_ezie_simulation_case_1_look_direction_case_2_retrieved_los_mag_fields.sav'):

    idldata = readsav(fn)


    lat, lon = idldata.pos['glat'], idldata.pos['glon']

    mag = idldata['ezie'].mag[0]

    obs_pos = idldata['ezie'].obs_pos[0]
    obs_lat = obs_pos['glat']
    obs_lon = obs_pos['glon']

    time = idldata['pos']['time'][0] 
    pdtime = pd.to_datetime(time + 365.15 * (idldata['pos']['date'][0][0] - 2025)*24*60**2, unit = 's', origin = '2024-1-1 00:00')

    data = pd.DataFrame({}, index = pdtime)

    for i in range(4):
        data['lat_' + str(i + 1)] = np.float32(obs_lat[i])
        data['lon_' + str(i + 1)] = np.float32(obs_lon[i])

        data['dbe_' + str(i + 1)] = np.float32( mag['dbe'][i])
        data['dbn_' + str(i + 1)] = np.float32( mag['dbn'][i])
        data['dbu_' + str(i + 1)] = np.float32(-mag['dbd'][i])
        data['db_current_' + str(i + 1)] = np.float32(mag['b_current'][i])
        data['Be_' + str(i + 1)] = np.float32( mag['be'][i])
        data['Bn_' + str(i + 1)] = np.float32( mag['bn'][i])
        data['Bu_' + str(i + 1)] = np.float32(-mag['bd'][i])

    data['sat_lat'] = np.float32(lat[0])
    data['sat_lon'] = np.float32(lon[0])
    data = data.astype(np.float32)




    return data
