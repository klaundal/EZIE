README file for Replication data for "Electrojet estimates from mesospheric magnetic field measurements"

The dataset consists of MHD simulation output (with metadata) and a dataset of simulated Zeeman magnetic field observations. More information in related publication "Electrojet estimates from mesospheric magnetic field measurements", doi:10.1002/essoar.10504160.1 

Contact: Karl Laundal (karl.laundal at uib.no)


sigmaH.2430.mix.reduced.h5
--------------------------
Magnetohydrodynamic (MHD simulation output in HDF5 format).

The HDF5 file is provided along with a metadata description in the XMF format. The latter can be read with a text editor (a version with .txt extension is also available), or opened in scientific visualization software such as Paraview and Visit. The HDF5 file can be accessed with the h5py Python package, or similar tools. 

The file contains the following 2D arrays:

X, Y are Cartesian coordinates of the grid corners in the SM frame in the units of ionospheric radius (assumed to be 6500 km).

Under the group Step#0 variables are given at cell-centers of the above grid for the Northern hemisphere:
"Field-aligned current NORTH": Field-aligned current density [microA/m^2]
"Hall conductance NORTH": Hall conductance [S]
"Pedersen conductance NORTH": Pedersen conductance [S]
"Potential NORTH": Electrostatic potential [kV]


electrojet_inversion_data.csv
-----------------------------
The columns include:
seconds: seconds since start of interval (4 min total)
sat_lat: latitude of satellite
sat_lon: longitude of satellite

The other columns end with _1, _2, _3, or _4, corresponding to the four look directions. The first part of the columns names are:
lat: latitude of measurement (at 80 km altitude)
lon: longitude of measurement (at 80 km altitude)
dbe: eastward component disturbance magnetic field [nT] (MHD output)
dbn: northward component disturbance magnetic field [nT] (MHD output)
dbu: upward component disturbance magnetic field [nT] (MHD output)
dbe_measured: simulated mesurement of eastward component disturbance magnetic field [nT], including noise
dbn_measured: simulated mesurement of northward component disturbance magnetic field [nT], including noise
dbu_measured: simulated mesurement of upward component disturbance magnetic field [nT], including noise
cov_ee, cov_en, cov_eu, cov_nn, cov_nu, cov_uu: elements of data covariance matrix corresponding to east (e), north (n) and (up) directions

NOTE: The full MHD output provided in separate file must be rotated counterclockwise by 210 degrees to line up with the data in this file. 


