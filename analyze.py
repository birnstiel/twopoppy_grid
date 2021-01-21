#!/usr/bin/env python
"""
Batch analyze simulation results to derive dust sizes and fluxes
"""
# %%
import argparse
from multiprocessing import Pool
import time as walltime
from pathlib import Path

import numpy as np
import h5py

import dipsy
from model import bumpmodel_result

au = dipsy.cgs_constants.au

start = walltime.time()

# %% -------------- argument parsing ------------------

RTHF = argparse.RawTextHelpFormatter
PARSER = argparse.ArgumentParser(description=__doc__, formatter_class=RTHF)
PARSER.add_argument('file', help='HDF5 file with the simulation data', type=str)
PARSER.add_argument('-c', '--cores', help='how many cores to use', type=int, default=1)
PARSER.add_argument('-t', '--test', help='for testing: only run this single model', type=int, default=None)

PARSER.add_argument('-l', '--lam', help='wavelength in cm', type=float, default=0.085)
PARSER.add_argument('-q', '--q', help='size distribution slope', type=float, default=3.5)
PARSER.add_argument('--flux-fraction', help='flux fraction to determine disk radius', type=float, default=0.68)
PARSER.add_argument('-o', '--opacity', help='which opacity to use', type=str, default='ricci_compact.npz')

# ARGS = PARSER.parse_args(['test.hdf5'])
ARGS = PARSER.parse_args()

opac = dipsy.Opacity(input=ARGS.opacity)

lam = ARGS.lam
q = ARGS.q
flux_fraction = ARGS.flux_fraction
fname = ARGS.file

fname_out = Path(fname)
fname_out = fname_out.with_name(f'{fname_out.stem }_analysis_lam{1e4 * lam:0.0f}_q{q:.1f}_f{100 * flux_fraction:.0f}{fname_out.suffix}')

# %% -------- open the data file -----------

with h5py.File(fname, 'r') as fid:
    n_data = len(fid)
    keys = list(fid.keys())

# %% -------- define the worker function -----------


def parallel_analyze(key):

    with h5py.File(fname, 'r') as fid:
        group = fid[key]
        b = bumpmodel_result(*[group[f][()] for f in bumpmodel_result._fields])
        params = group['params'][()]

    rf_t, flux_t, *_ = dipsy.get_all_observables(b, opac, lam, q=q, flux_fraction=flux_fraction)

    out = {
        'alpha': params[0],
        'Mdisk': params[1],
        'r_c': params[2],
        'v_frag': params[3],
        'M_star': params[4],
        'rf_t': np.squeeze(rf_t / au).tolist(),
        'flux_t': np.squeeze(flux_t).tolist(),
        'time': b.time
    }

    return out


# %% ----------------- parallel execution ---------------
indices = range(n_data)
if ARGS.test is not None:
    print(f'TESTING: only analyzing simulation #{ARGS.test}')
    indices = [ARGS.test]

pool = Pool(processes=ARGS.cores)

results = []
n_sim = len(indices)
keys = [keys[i] for i in indices]

for i, res in enumerate(pool.imap_unordered(parallel_analyze, keys)):
    results.append(res)
    print(f'\rRunning ... {(i+1) / n_sim:.1%}', end='', flush=True)

print('\r--------- DONE ---------')


elapsed_time = (walltime.time() - start) / 60
print('{} of {} simulations finished in {:.3g} minutes'.format(len(results) - results.count(False), len(results), elapsed_time))

# %% --------------- output --------------

# write to an hdf5 file

df = dipsy.utils.write_to_hdf5(fname_out, results)
