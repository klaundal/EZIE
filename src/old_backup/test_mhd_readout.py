from pysymmetry.visualization import polarsubplot
import matplotlib.pyplot as plt
import simulation_utils
from importlib import reload
reload(simulation_utils)
import read_idl
reload(read_idl)
import numpy as np
from dipole import geo2mag

fig = plt.figure()
pax = polarsubplot.Polarsubplot(fig.add_subplot(111))

data = read_idl.get_data()
shift = 14
# convert all geographic coordinates and vector components in data to geomagnetic:
for i in range(4):
    i = i + 1
    data['lat_' + str(i)], data['lon_' + str(i)], data['dbe_' + str(i)], data['dbn_' + str(i)] = geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_' + str(i)].values, data['dbn_' + str(i)].values, epoch = 2020)

    pax.scatter(data['lat_' + str(i)], data['lon_' + str(i)]/15 - shift, c = data['dbu_' + str(i)], vmin = -800, vmax = 800, cmap = plt.cm.bwr, zorder = 100)

data['sat_lat'], data['sat_lon'] = geo2mag(data['sat_lat'].values, data['sat_lon'].values, epoch = 2020)
pax.plot(data['sat_lat'], data['sat_lon']/15 - shift, zorder = 101, color = 'black')



mlat, mlon = np.meshgrid(np.linspace(50, 89, 50), np.linspace(0, 360, 383))
mlatv, mlonv = np.meshgrid(np.linspace(50, 89, 10), np.linspace(0, 360, 24))

Bu = simulation_utils.get_MHD_dB(mlat.flatten(), mlon.flatten()).reshape(mlat.shape)
je, jn = simulation_utils.get_MHD_jeq(mlatv.flatten(), mlonv.flatten())


pax.contourf(mlat, mlon/15, Bu, levels = np.linspace(-800, 800, 22), cmap = plt.cm.bwr)
#pax.plotpins(mlatv.flatten(), mlonv.flatten()/15, je, jn, SCALE = .3)



plt.show()