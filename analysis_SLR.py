"""
Analyze a simulation from the grid to get data for
the Size-Luminosity-Relation (SLR), that is:

- the input parameters alpha, Mdisk, r_c, v_frag, M_star
- characteristic radius for all snapshots
- flux for all snapshots
- snapshot times

"""
import argparse
from pathlib import Path

import numpy as np
import h5py

import dipsy

# local imports need to be imported in a special way

model = dipsy.utils.remote_import(Path(__file__).absolute().parent / 'model.py')
bumpmodel_result = model.bumpmodel_result

au = dipsy.cgs_constants.au

# %% define a second parser that processes analyis-specific options
RTHF = argparse.RawTextHelpFormatter
PARSER = argparse.ArgumentParser(description=__doc__, formatter_class=RTHF)
PARSER.add_argument('file', help='HDF5 file with the simulation data', type=str)
PARSER.add_argument('-l', '--lam', help='wavelength in cm', type=float, default=0.085)
PARSER.add_argument('-l2', '--lam2', help='second wavelength for alpha in cm', type=float, default=0.3)
PARSER.add_argument('-q', '--q', help='size distribution slope', type=float, default=3.5)
PARSER.add_argument('--no-scattering', dest='scattering', help='turn off scattering', default=True, action='store_false')
PARSER.add_argument('--flux-fraction', help='flux fraction to determine disk radius', type=float, default=0.68)
PARSER.add_argument('-o', '--opacity', help='which opacity to use', type=str, default='ricci_compact.npz')


def process_args(ARGS):
    """Process parsed arguments

    Parameters
    ----------
    ARGS : namespace
        the already parsed arguments

    Returns
    -------
    dict
        dictionary containing all relevant settings
    """
    opac = dipsy.Opacity(input=ARGS.opacity)

    lam = ARGS.lam
    lam2 = ARGS.lam2
    q = ARGS.q
    scattering = ARGS.scattering
    flux_fraction = ARGS.flux_fraction
    fname_in = ARGS.file
    fname_out = Path(fname_in)
    fname_out = fname_out.with_name(f'{fname_out.stem }_analysis_lam{1e4 * lam:0.0f}_q{q:.1f}_f{100 * flux_fraction:.0f}_s{int(scattering)}{fname_out.suffix}')

    return {
        'lam': lam,
        'lam2': lam2,
        'q': q,
        'flux_fraction': flux_fraction,
        'fname_in': fname_in,
        'fname_out': fname_out,
        'opac': opac,
        'scattering': scattering,
    }


def parallel_analyze(key, settings=None):
    """Analyze simulation `key` with given settings

    Parameters
    ----------
    key : str
        name of the simulation dataset in hdf5 file
    settings : dict
        dictionary containing all relevant settings

    Returns
    -------
    dict
        result dictionary
    """
    if settings is None:
        raise ValueError('settings must be given as keyword')

    # get all settings
    fname = settings['fname_in']
    opac = settings['opac']
    lam = settings['lam']
    lam2 = settings['lam2']
    q = settings['q']
    scattering = settings['scattering']
    flux_fraction = settings['flux_fraction']

    # get the data from file and do the processing

    with h5py.File(fname, 'r') as fid:
        group = fid[key]
        b = bumpmodel_result(*[group[f][()] for f in bumpmodel_result._fields])
        params = group['params'][()]

    rf_t, flux_t, *_ = dipsy.get_all_observables(b, opac, [lam, lam2], q=q, flux_fraction=flux_fraction, scattering=scattering)

    alpha_mm = - np.log(flux_t[:, 1] / flux_t[:, 0]) / np.log(lam2 / lam) - 2.0

    # store the relevant results in a dict

    out = {
        'alpha': params[0],
        'Mdisk': params[1],
        'r_c': params[2],
        'v_frag': params[3],
        'M_star': params[4],
        'rf_t': np.squeeze(rf_t[0] / au).tolist(),
        'flux_t': np.squeeze(flux_t).tolist(),
        'alpha_mm': alpha_mm,
        'time': b.time
    }

    return out
