#!/usr/bin/env python
"""
Batch analyze simulation results to derive dust sizes and fluxes
"""
# %%
import argparse
from multiprocessing import Pool
import time as walltime
from pathlib import Path

import pandas as pd
import numpy as np
import h5py

import dipsy
from model import bumpmodel_result

au = dipsy.cgs_constants.au

start = walltime.process_time()

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
fname_out = fname_out.with_name(f'{fname_out.stem }_analysis_lam{1e4 * lam:0.0f}_q{q:.1f}_f{100 * flux_fraction:.0f}').with_suffix(fname_out.suffix)

# %% -------- open the data file -----------

fid = h5py.File(fname, 'r')

n_data = len(fid)

# %% -------- define the worker function -----------


def parallel_analyze(i):

    d = dipsy.utils.read_from_hdf5(fname, f'{i:07d}')

    b = bumpmodel_result(*[d[f] for f in bumpmodel_result._fields])

    rf_t, flux_t, *_ = dipsy.get_all_observables(b, opac, lam, q=q, flux_fraction=flux_fraction)

    out = {
        'alpha': d['params'][0],
        'Mdisk': d['params'][1],
        'r_c': d['params'][2],
        'v_frag': d['params'][3],
        'M_star': d['params'][4],
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
results = pool.map(parallel_analyze, indices)
elapsed_time = walltime.process_time() - start
print('{} of {} simulations finished in {:.3g} minutes'.format(len(results) - results.count(False), len(results), elapsed_time))

# %% --------------- output --------------

# write to an hdf5 file

df = dipsy.utils.write_to_hdf5(fname_out, results)
