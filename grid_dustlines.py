import numpy as np

import dipsy
from model import run_bump_model2

au = dipsy.cgs_constants.au
year = dipsy.cgs_constants.year
M_sun = dipsy.cgs_constants.M_sun
sigma_sb = dipsy.cgs_constants.sigma_sb

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
filename = 'dustlines'

# Store them in a list
param = (alpha, Md, Rc, v_frag, M_star)

# make the grids

r = np.logspace(np.log10(r0), np.log10(r1), nr + 1)
time = np.hstack((0, np.logspace(np.log10(t0), np.log10(t1), nt)))


# -------------- define worker function here ---------------


def parallel_run(param):

    try:
        # Calculate the luminosity of every star depending on the mass

        L, R, Teff = dipsy.get_stellar_properties(param[4] * M_sun, 1e6 * year)

        # set flaring angle

        phi = 0.05

        # calculate the temperature

        Temp = ((phi * L / (4 * np.pi * sigma_sb * r**2)) + 1e4)**0.25

        res = run_bump_model2(
            *param,
            r,
            Temp,
            time,
            r_bumps,
            a_bumps,
            mass_ratio,
            gap_profile=gap_profile)

        return res
    except Exception as err:
        print(err)
        return False
