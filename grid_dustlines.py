import numpy as np
from pathlib import Path

import dipsy

# local imports need to be imported in a special way
model = dipsy.utils.remote_import(Path(__file__).absolute().parent / 'model.py')
run_bump_model2 = model.run_bump_model2


au = dipsy.cgs_constants.au
year = dipsy.cgs_constants.year
M_sun = dipsy.cgs_constants.M_sun
sigma_sb = dipsy.cgs_constants.sigma_sb

r0 = 0.05 * au
r1 = 2000. * au
nr = 200

t0 = 1e4 * year
t1 = 1e7 * year
nt = 30

# Define bumps and locations

r_bumps = [1 / 3, 2 / 3]
a_bumps = [0, 0]
gap_profile = None  # 'kanagawa' or None

alpha = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
Md = np.logspace(np.log10(1e-3), np.log10(0.2), 6)
Rc = [30, 60, 100, 130, 160, 200]
v_frag = [100, 300, 600, 1000, 1500, 2000]
M_star = np.logspace(np.log10(0.2), np.log10(2), 6)
mass_ratio = [3e-4, 1e-3]

cores = 50
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
