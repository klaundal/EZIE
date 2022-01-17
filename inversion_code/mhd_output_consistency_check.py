""" Check if MHD output is consistent***

I will do this by checking that one component can be used to derive the other:
Br -> Be, Bn
Be -> Br, Bn
Bn -> Br, Be

In each case, the one single component is used to calculate an equivalent current pattern,
and then the equivalent current is used to calculate the corresponding magnetic field components.
I will use SECS analysis to do this. 

*** Note that the test will rely on my interpolation scheme - so if 
the test result is negative, it could be due to the interpolation and
not the MHD calculations
"""


import numpy as np
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices, CSgrid, CSprojection
import cases
import pandas as pd
from dipole import geo2mag
from secsy import spherical 
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pysymmetry.visualization.grids import sdarngrid
from importlib import reload
reload(cases) 

RE = 6372.e3
RI  = (6371.2 + 120) * 1e3 # SECS height (m)
DLAT = .5

levels = np.linspace(-600, 600, 22) # color levels for contour plotting [nT]

label = 'proposal_stage'
label = 'case_1'
#label = 'case_2'
#label = 'case_3'
#label = 'case_4'

info = cases.cases[label]

OBSHEIGHT = info['observation_height']


LRES = 40. # spatial resolution of SECS grid along satellite track
WRES = 20. # spatial resolution perpendicular to satellite tarck
W = 1500
L = 1500

orientation = np.array([1, 0])
p = (0 + (180 if label == 'proposal_stage' else 0), 70)
projection = CSprojection(p, orientation)
grid = CSgrid(projection, L, W, LRES, WRES)



# make an extended grid
xi_e   = np.r_[grid. xi_mesh.min() - 8 * grid.dxi : grid. xi_mesh.max() + 8 * grid. dxi:grid.dxi ] - grid.dxi  / 2 
eta_e  = np.r_[grid.eta_mesh.min() - 8 * grid.deta: grid.eta_mesh.max() + 8 * grid.deta:grid.deta] - grid.deta / 2 

extended_grid = CSgrid(CSprojection(grid.projection.position, grid.projection.orientation),
                       grid.L, grid.W, grid.Lres, grid.Wres, 
                       edges = (xi_e, eta_e), R = RI)


# plot grid and extended grid together
fig, ax = plt.subplots()

for gr, color in zip((grid, extended_grid), ['C0', 'C1']):
    lines = tuple(zip(gr.xi, gr.eta)) + tuple(zip(gr.xi.T, gr.eta.T))
    lines = [np.array(l).T for l in lines]
    lc = LineCollection(lines, linewidths = .8, colors = color)
    ax.add_collection(lc)

ax.set_xlim(extended_grid.xi_min , extended_grid.xi_max )
ax.set_ylim(extended_grid.eta_min, extended_grid.eta_max)
ax.set_aspect('equal')
ax.set_axis_off()
ax.set_title('Data grid (inner)\nSECS grid (outer)')
plt.savefig('figures/mhd_output_consistency_check_grids.png', dpi = 200)


# get MHD magnetic fields at inner grid points:
mhdBu =  info['mhdfunc'](grid.lat.flatten(), grid.lon.flatten(), fn = info['mhd_B_fn'])
mhdBe =  info['mhdfunc'](grid.lat.flatten(), grid.lon.flatten(), component = 'Bphi [nT]', fn = info['mhd_B_fn'])
mhdBn = -info['mhdfunc'](grid.lat.flatten(), grid.lon.flatten(), component = 'Btheta [nT]', fn = info['mhd_B_fn'])
components = [mhdBe, mhdBn, mhdBu]

# calculate SECS matrices
matrices = get_SECS_B_G_matrices(grid.lat, grid.lon, RE + OBSHEIGHT * 1e3, extended_grid.lat, extended_grid.lon, RI = RI)


# plot the result
fig, axes = plt.subplots(nrows = 3, ncols = 4, figsize = (12, 8))

# plot the MHD output in the left column:
for axs, B in zip(axes, [mhdBe, mhdBn, mhdBu]):
    axs[0].contourf(grid.xi, grid.eta, B.reshape(grid.shape), cmap = plt.cm.bwr, levels = levels)

    # make LineCollection to plot grid
    lines = tuple(zip(grid.xi, grid.eta)) + tuple(zip(grid.xi.T, grid.eta.T))
    lines = [np.array(l).T for l in lines]
    lc = LineCollection(lines, linewidths = .2, colors = 'grey')

    cs = axs[0].contour(grid.xi, grid.eta, grid.lat, levels = np.r_[50:86:5], colors = 'black', linewidths = .8)
    axs[0].clabel(cs, inline=1, fontsize=10)

    axs[0].add_collection(lc)
    axs[0].set_aspect('equal')
    axs[0].set_axis_off()

# plot inverison based on each component:
for j in range(3): # column
    m = np.linalg.lstsq(matrices[j], components[j], rcond = 1e-2)[0] # create model
    for i in range(3): # row
        axes[i, j+1].contourf(grid.xi, grid.eta, matrices[i].dot(m).reshape(grid.shape), cmap = plt.cm.bwr, levels = levels)

        # make LineCollection to plot grid
        lines = tuple(zip(grid.xi, grid.eta)) + tuple(zip(grid.xi.T, grid.eta.T))
        lines = [np.array(l).T for l in lines]
        lc = LineCollection(lines, linewidths = .2, colors = 'grey')

        cs = axes[i, j+1].contour(grid.xi, grid.eta, grid.lat, levels = np.r_[50:86:5], colors = 'black', linewidths = .8)
        axes[i, j+1].clabel(cs, inline=1, fontsize=10)

        axes[i, j+1].add_collection(lc)
        axes[i, j+1].set_aspect('equal')
        axes[i, j+1].set_axis_off()



axes[0, 0].text(axes[0, 0].get_xlim()[0] - 3*grid.dxi, 0, '$B_e$', size = 14, ha = 'right', rotation = 90, va = 'center')
axes[1, 0].text(axes[1, 0].get_xlim()[0] - 3*grid.dxi, 0, '$B_n$', size = 14, ha = 'right', rotation = 90, va = 'center')
axes[2, 0].text(axes[2, 0].get_xlim()[0] - 3*grid.dxi, 0, '$B_u$', size = 14, ha = 'right', rotation = 90, va = 'center')

axes[0, 0].set_title('MHD output'            , size = 14)
axes[0, 1].set_title('$B_e$-based inversions', size = 14)
axes[0, 2].set_title('$B_n$-based inversions', size = 14)
axes[0, 3].set_title('$B_u$-based inversions', size = 14)


plt.savefig('figures/mhd_output_consistency_check_' + label + '.png', dpi = 200)

plt.show()



#m = np.linalg.lstsq(np.vstack((Ge, Gn, Gu)), np.hstack((mhdBe, mhdBn, mhdBu)), rcond = 1e-2)[0]
#m = np.linalg.lstsq(np.vstack((Gu)), np.hstack((mhdBu)), rcond = 1e-4)[0]
