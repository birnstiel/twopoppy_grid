import dipsy
import numpy as np

au = dipsy.cgs_constants.au
year = dipsy.cgs_constants.year

r0 = 0.05 * au
r1 = 2000. * au
nr = 400

t0 = 1e4 * year
t1 = 1e7 * year
nt = 30

# Define bumps and locations

r_bumps = [1 / 3, 2 / 3]
a_bumps = [0, 0]
gap_profile = None  # 'kanagawa' or None

alpha = [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2]
Md = np.logspace(np.log10(1e-3), np.log10(0.2), 10)
Rc = [10, 30, 50, 80, 100, 130, 150, 180, 200, 230]
v_frag = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
M_star = np.logspace(np.log10(0.2), np.log10(2), 10)
mass_ratio = [3e-4, 1e-3]

cores = 10
filename = 'test'

# Store them in a list

param = (alpha, Md, Rc, v_frag, M_star)
