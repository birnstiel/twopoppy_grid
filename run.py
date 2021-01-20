#!/usr/bin/env python
"""Runs a twopoppy simulation grid.

Executes a large list of parameters and stores the results in an HDF5 file.
"""
# %% ------------ imports ------------
import itertools
import importlib.util
import sys
import time as walltime
from multiprocessing import Pool
from pathlib import Path
import argparse

import numpy as np

import dipsy
from model import run_bump_model2

start = walltime.process_time()

year = dipsy.cgs_constants.year
M_sun = dipsy.cgs_constants.M_sun
sigma_sb = dipsy.cgs_constants.sigma_sb

# %% -------------- argument parsing ------------------

RTHF = argparse.RawTextHelpFormatter
PARSER = argparse.ArgumentParser(description=__doc__, formatter_class=RTHF)
PARSER.add_argument('grid', help='python file with grid setup', type=str)
PARSER.add_argument('-c', '--cores', help='how many cores to use, overwrites grid setting', type=int, default=0)
PARSER.add_argument('-t', '--test', help='for testing: only run this single model', type=int, default=None)
ARGS = PARSER.parse_args()

# %% ----------- get the grid parameters --------------

grid_file = Path(ARGS.grid).resolve()

if not grid_file.is_file():
    print(f'grid file {grid_file} not found.')
    sys.exit(1)

print(f'importing grid from {grid_file}')
spec = importlib.util.spec_from_file_location(grid_file.name, str(grid_file))
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)

param = grid.param

r0 = grid.r0
r1 = grid.r1
nr = grid.nr

t0 = grid.t0
t1 = grid.t1
nt = grid.nt

r_bumps = grid.r_bumps
a_bumps = grid.a_bumps
mass_ratio = grid.mass_ratio
gap_profile = grid.gap_profile

filename = grid.filename

cores = grid.cores
if ARGS.cores > 0:
    cores = ARGS.cores

# %% -------------- set up parameter list & grids ---------------

# make a list of all possible combinations

param_val = list(itertools.product(*param))

if ARGS.test is not None:
    print(f'TESTING: only running simulation #{ARGS.test}')
    param_val = [param_val[ARGS.test]]

# make the grids

r = np.logspace(np.log10(r0), np.log10(r1), nr + 1)
time = np.hstack((0, np.logspace(np.log10(t0), np.log10(t1), nt)))

# %% -------------- define worker function ---------------


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


# %% --------------- parallel execution ---------------

pool = Pool(processes=cores)

results = []
n_sim = len(param_val)

for i, res in enumerate(pool.imap_unordered(parallel_run, param_val)):
    results.append(res)
    print(f'\rRunning ... {(i+1) / n_sim:.1%}', end='', flush=True)

print('\r--------- DONE ---------')

elapsed_time = walltime.process_time() - start
print('{} of {} simulations finished in {:.3g} minutes'.format(len(results) - results.count(False), len(results), elapsed_time))

# %% ------------ output --------------

# to write to an hdf5 file, we need to put everything simulation result + parameters into one dict

dicts = [res._asdict() for res in results]
del(results)

for d, params in zip(dicts, param_val):
    d['params'] = params

# now write to file

dipsy.utils.write_to_hdf5(Path(filename).with_suffix('.hdf5'), dicts)
