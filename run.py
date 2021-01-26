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

import dipsy

start = walltime.time()

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
filename = grid.filename
parallel_run = grid.parallel_run
cores = grid.cores

if ARGS.cores > 0:
    cores = ARGS.cores

# %% -------------- set up parameter list & grids ---------------

# make a list of all possible combinations

param_val = list(itertools.product(*param))

if ARGS.test is not None:
    print(f'TESTING: only running simulation #{ARGS.test}')
    param_val = [param_val[ARGS.test]]


# %% --------------- parallel execution ---------------

pool = Pool(processes=cores)

results = []
n_sim = len(param_val)

for i, res in enumerate(pool.imap(parallel_run, param_val)):
    results.append(res)
    print(f'\rRunning ... {(i+1) / n_sim:.1%}', end='', flush=True)

print('\r--------- DONE ---------')

sims_done = walltime.time()
print('{} of {} simulations finished in {:.3g} minutes'.format(len(results) - results.count(False), len(results), (sims_done - start) / 60))

# %% ------------ output --------------

# to write to an hdf5 file, we need to put everything simulation result + parameters into one dict

dicts = [res._asdict() for res in results]
del(results)

for d, params in zip(dicts, param_val):
    d['params'] = params

# now write to file

dipsy.utils.write_to_hdf5(Path(filename).with_suffix('.hdf5'), dicts)

saving_done = walltime.time()
print(f'saving done (took {(saving_done - sims_done) / 60:.3g} min)')
