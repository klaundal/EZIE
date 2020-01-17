""" script to produce figure showing height dependence of electrojet
    magnetic field. 
"""

import numpy as np
import matplotlib
matplotlib.use('qt5agg')
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
matplotlib.rc('font',family='AppleGothic')
from sh import SHkeys, get_legendre

CURRENT_HEIGHT = 105. # km
A = 6371.2 # reference height, km
MAXN = 410 # maximum spherical harmonic degree to consider
MAXM = 0   # maximum spherical harmonic order (only important in map plot) # (RIGHT NOW ONLY 1 WORKS!)
MINN = 10


SQRTPOWER = True # True to plot sqrt(B^2) as function of h instead of B^2(h)
SQUARE_AMPLITUDE = True # should probably be True (Gjerloev 2011 says it is the "square root of the power")
INCLUDE_LARGESCALES = True # True to include scales that are larger than ST5 study in maps (extrapolate) 

fig = plt.figure(figsize = (15, 5))
ax_ps = fig.add_subplot(131) # FAC spatial power spectrum
ax_Bh = fig.add_subplot(132) # dB height dependence
ax_Bm = plt.subplot2grid((10, 3), (0, 2), rowspan = 9, projection = '3d') # Maps of B at ground and 85 km
ax_Bm_cbar = plt.subplot2grid((30, 3), (29, 2)) # Colorbar for maps

# Data from Gjerloev et al. 2011 Fig 10 (Nightside,  disturbed)
scale_size = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]) # km
amplitude  = np.array([2, 12, 19, 24, 29, 33, 37, 41, 45, 48, 51, 54, 57, 60, 62, 64, 67, 70, 73]) # nT


#############################
# Plot spatial power spectrum
#############################
ax_ps.plot(scale_size, amplitude, linewidth = 3, color = 'black', label = 'Measured')
ax_ps.set_xlabel('SCALE SIZE [km]', fontname = 'AppleGothic')
ax_ps.set_ylabel('FAC MAGNETIC FIELD AMPLITUDE [nT]', fontname = 'AppleGothic')


# convert FAC scale size [km at 200 km] to spherical harmonic degree:
n = np.pi * (A + 200) // scale_size

ax_ps.spines['right'].set_visible(False)
ax_ps.yaxis.set_ticks_position('left')


#########################
# Calculate and plot B(h)
#########################


# amplitude vs scale_size is quite linear on log-log scale so
# we fit a line to get a general relationship: 
p = np.polyfit(np.log(n[1:]), np.log(amplitude[1:]), 1)

# amplitude as function of arbitrary n:
amplitude_of_n = lambda n: np.exp(p[1])*n**p[0]


# make plots of B vs h and FAC scale sizes for n including large scales
heights = np.linspace(0, CURRENT_HEIGHT, 200).reshape((-1, 1))
nn      = np.r_[MINN:MAXN + 1:1].reshape((1, -1))
if SQUARE_AMPLITUDE:
    An      = amplitude_of_n(nn)**2 / nn * (A / (A + CURRENT_HEIGHT)) ** (2 * nn - 2)
else:
    An      = amplitude_of_n(nn) / nn * (A / (A + CURRENT_HEIGHT)) ** (2 * nn - 2)
Bvsh    = (nn * ((A + heights) / A) ** (2 * nn - 2) * An).sum(axis = 1)
if SQRTPOWER:
    Bvsh    = np.sqrt(Bvsh/Bvsh[0])*100
else:
    Bvsh    = Bvsh/Bvsh[0]*100
ax_ps.plot(np.pi*(6371.2 + 200) / nn.flatten() , amplitude_of_n(nn.flatten()), color = 'grey', label = 'Fitted and extrapolated')
ax_ps.legend(frameon = False, loc = 4)

# add second x axis to power spectrum plot with spherical harmonic degree:
ax_psn = ax_ps.twiny()
n_tick_locations = [10, 20, 30, 50, 400]
ax_psn.set_xticks(np.pi * (A + 200) / np.array(n_tick_locations))
ax_psn.set_xticklabels(n_tick_locations)
ax_psn.set_xlabel('SPHERICAL HARMONIC DEGREE', fontname = 'AppleGothic')
ax_psn.set_xlim(ax_ps.get_xlim())
ax_psn.spines['right'].set_visible(False)
ax_psn.yaxis.set_ticks_position('left')

ax_Bh.plot(Bvsh, heights, color = 'grey', zorder = 2)
ax_Bh.plot(ax_Bh.get_xlim(), [85, 85], color = 'lightgray')


# repeat, but only with small-scale currents (within the range from ST-5 study)
nn      = np.r_[int(np.nanmin(n)):MAXN + 1:1].reshape((1, -1))
if SQUARE_AMPLITUDE:
    An      = amplitude_of_n(nn)**2 / nn * (A / (A + CURRENT_HEIGHT)) ** (2 * nn - 2)
else:
    An      = amplitude_of_n(nn) / nn * (A / (A + CURRENT_HEIGHT)) ** (2 * nn - 2)
Bvsh_smallscale = (nn * ((A + heights) / A) ** (2 * nn - 2) * An).sum(axis = 1)
if SQRTPOWER:
    Bvsh_smallscale = np.sqrt(Bvsh_smallscale/Bvsh_smallscale[0])*100
else:
    Bvsh_smallscale = Bvsh_smallscale/Bvsh_smallscale[0]*100

ax_Bh.plot(Bvsh_smallscale, heights, 'k-', linewidth = 3, zorder = 1)
ax_Bh.set_xlim(70, ax_Bh.get_xlim()[1])
ax_Bh.plot(ax_Bh.get_xlim(), [85, 85], color = 'lightgray')
ax_Bh.plot(ax_Bh.get_xlim(), [CURRENT_HEIGHT, CURRENT_HEIGHT], color = 'lightgray')

#ax_Bh.set_title('MAGNETIC FIELD MAGNITUDE VS HEIGHT', fontname = 'AppleGothic')
ax_Bh.set_ylabel('HEIGHT [km]', fontname = 'AppleGothic')
ax_Bh.set_xlabel('MAGNETIC FIELD STRENGTH [nT]', fontname = 'AppleGothic')
ax_Bh.text(100, 85, '85 km', va = 'center', backgroundcolor = 'white', fontname = 'AppleGothic')
ax_Bh.text(100, CURRENT_HEIGHT, str(int(CURRENT_HEIGHT)) + ' km', va = 'center', backgroundcolor = 'white', fontname = 'AppleGothic')

ax_Bh.annotate('Normalized\n to 100 nT on ground', xy=(100, 0), xytext=(150, 20),
            arrowprops=dict(facecolor='gray', shrink=0.01, width = 2, edgecolor = 'gray'), color = 'gray', fontname = 'AppleGothic', ha = 'left'
            )
ax_Bh.set_ylim(0, CURRENT_HEIGHT + 1)


# Hide the right and top spines
ax_Bh.spines['right'].set_visible(False)
ax_Bh.spines['top'].set_visible(False)
ax_Bh.yaxis.set_ticks_position('left')
ax_Bh.xaxis.set_ticks_position('bottom')


#####################################################
# plot maps of synthetic magnetic field perturbations 
#####################################################

if INCLUDE_LARGESCALES:
    nn = np.r_[MINN:MAXN + 1:1].reshape((1, -1))
    if SQUARE_AMPLITUDE:
        An  = amplitude_of_n(nn)**2 / nn * (A / (A + CURRENT_HEIGHT)) ** (2 * nn - 2)
    keys = SHkeys(MAXN, MAXM).MleN().setNmin(MINN)
else:
    keys = SHkeys(MAXN, MAXM).MleN().int(np.nanmin(n))



# create dictionaries of synethic SH coefficients that describe
# field with the correct power
q = {} 
s = {}


# Use Sabaka et al. 2010 equation 118 to find the coefficients corresponding 
# to the given power spectrum at r = RE + HEIGHT for a field of external origin
for key in keys: 
    n, m = key
    angle = np.random.random() * 2 * np.pi # random angle between 0 and 2 pi
    
    q[key] = np.sqrt( An[:, n  - keys.n.min()]) * np.cos(angle) 
    s[key] = np.sqrt( An[:, n  - keys.n.min()]) * np.sin(angle)

qq = np.array([q[key] for key in keys])
ss = np.array([s[key] for key in keys])
model = np.vstack((qq, ss))

# check that the power spectrum matches the one based on FACs
if MAXM != 0:
    raise Exception('MAXM > 0 not implemented yet... ')
power = ((qq**2 + ss**2).flatten() * nn.flatten() * ((A + CURRENT_HEIGHT)/A)**(2 * nn - 2)).flatten()
assert(np.all(np.isclose(power - amplitude_of_n(nn.flatten())**2, 0)))

# set up map to get coordinates of grid
fig2 = plt.figure()
_ = fig2.add_subplot(111)
m = Basemap(width=6000000, height=6000000, projection='aea', resolution='l',lat_1=29.5, lat_2=45.5, lat_0=38.5, lon_0=-96., ax = _)
xx, yy = np.meshgrid(np.linspace(m.xmin, m.xmax, 100), np.linspace(m.ymin, m.ymax, 100))
xx, yy = xx.flatten(), yy.flatten()
lon, lat = m(xx, yy, inverse = True)
colat = 90 - lat[:, np.newaxis]
lon   = lon[:, np.newaxis]

# matrices
PdP = get_legendre(MAXN, MAXM, colat, keys = keys)
P, dP = np.split(PdP, 2, axis = 1)
nn = keys.n
mm = keys.m
cosmphi = np.cos(mm * lon * np.pi/180)
sinmphi = np.sin(mm * lon * np.pi/180)

# field components
def get_B(h):
    rr = ((A + h)/A) ** (nn - 1) 
    Br     = np.hstack((-rr * nn *  P * cosmphi, -rr * nn *  P * sinmphi)).dot(model)
    Btheta = np.hstack((-rr *      dP * cosmphi, -rr *      dP * sinmphi)).dot(model)
    Bphi   = np.hstack(( rr * mm *  P * sinmphi, -rr * mm *  P * cosmphi)).dot(model)
    return np.sqrt(Br**2 + Btheta**2 + Bphi**2)

collection = m.drawcoastlines()
coastlines = collection.get_segments()
plt.close(fig = fig2)

BH = get_B(85)
NORMALIZATION = 100 / BH.max()
BH = BH * NORMALIZATION
MIN, MAX = 0, 100


ax_Bm.contourf(xx.reshape((100, 100)), yy.reshape((100, 100)), NORMALIZATION * get_B(85) .reshape((100, 100)), zdir = 'z', offset = 85, levels = np.linspace(MIN, MAX, 16), cmap = plt.cm.gray_r)#, extend = 'both')#, , extend = 'both')
ax_Bm.contourf(xx.reshape((100, 100)), yy.reshape((100, 100)), NORMALIZATION * get_B(0 ) .reshape((100, 100)), zdir = 'z', offset = 0 , levels = np.linspace(MIN, MAX, 16), cmap = plt.cm.gray_r)#, extend = 'both')#, , extend = 'both')
ax_Bm.set_zlim(0, 105)

for cc in coastlines:
    ax_Bm.plot(cc.T[0], cc.T[1], np.zeros_like(cc.T[0]), color = 'black', zorder = 100, linewidth = .5)

ax_Bm.set_xticks([])
ax_Bm.set_yticks([])
ax_Bm.set_xlim(m.xmin, m.xmax)
ax_Bm.set_ylim(m.ymin, m.ymax)
ax_Bm.view_init(elev=32., azim=-58)
ax_Bm.set_zlabel('HEIGHT [km]')

zz = np.vstack((np.linspace(MIN, MAX, 16), np.linspace(MIN, MAX, 16)))
ax_Bm_cbar.contourf(zz, np.vstack((np.zeros(16), np.ones(16))), zz, levels = zz[0], cmap = plt.cm.gray_r)
ax_Bm_cbar.set_xlabel('MAGNETIC FIELD STRENGTH (NORMALIZED)')

ax_Bm_cbar.set_yticks([])


ax_Bm.text2D(0, 1, ' C)', ha = 'left', va = 'top', transform=ax_Bm .transAxes, bbox = dict(facecolor = 'white', boxstyle='square', alpha = 1), zorder = 10, weight = 'bold')
ax_psn.text (0, 1, ' A)', ha = 'left', va = 'top', transform=ax_psn.transAxes, bbox = dict(facecolor = 'white', boxstyle='square', alpha = 1), zorder = 10, weight = 'bold')
ax_Bh.text  (0, 1, ' B)', ha = 'left', va = 'top', transform=ax_Bh .transAxes, bbox = dict(facecolor = 'white', boxstyle='square', alpha = 1), zorder = 10, weight = 'bold')

plt.subplots_adjust(top=0.88, bottom=0.11, left=0.06, right=0.96, hspace=0.2, wspace=0.15)

plt.savefig('./paper/figures/radial_dependence.pdf')
plt.savefig('./paper/figures/radial_dependence.svg')
plt.savefig('./paper/figures/radial_dependence.png', dpi = 250)



plt.show()
