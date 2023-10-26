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
import pandas as pd

import dipsy

year = dipsy.cgs_constants.year
au = dipsy.cgs_constants.au

# local imports need to be imported in a special way

model = dipsy.utils.remote_import(Path(__file__).absolute().parent / 'model_6_0_randomgrid_sebastian.py')
bumpmodel_result = model.bumpmodel_result

au = dipsy.cgs_constants.au
M_sun = dipsy.cgs_constants.M_sun

# %% define a second parser that processes analyis-specific options
RTHF = argparse.RawTextHelpFormatter
PARSER = argparse.ArgumentParser(description=__doc__, formatter_class=RTHF)
PARSER.add_argument('file', help='HDF5 file with the simulation data', type=str)
PARSER.add_argument('-l', '--lam', help='wavelength in cm', type=float, default=0.088174252)
PARSER.add_argument('-l2', '--lam2', help='second wavelength for alpha in cm', type=float, default=0.13324109)
PARSER.add_argument('-l3', '--lam3', help='third wavelength for alpha in cm', type=float, default=0.20675342)
PARSER.add_argument('-l4', '--lam4', help='fourth wavelength for alpha in cm', type=float, default=0.2725386)
PARSER.add_argument('-l5', '--lam5', help='fifth wavelength for alpha in cm', type=float, default=0.33310273)
PARSER.add_argument('-l6', '--lam6', help='sixth wavelength for alpha in cm', type=float, default=0.69719176)
PARSER.add_argument('-l7', '--lam7', help='sixth wavelength for alpha in cm', type=float, default=0.81024989)
PARSER.add_argument('-l8', '--lam8', help='sixth wavelength for alpha in cm', type=float, default=0.99930819)
PARSER.add_argument('-q', '--q', help='size distribution slope', type=float, default=3.5)
PARSER.add_argument('-qd', '--qd', help='size distribution slope in the drift limit, if none, uses value of q', type=float, default=None)
PARSER.add_argument('--no-scattering', dest='scattering', help='turn off scattering', default=True, action='store_false')
PARSER.add_argument('--flux-fraction', help='flux fraction to determine disk radius', type=float, default=0.68)
PARSER.add_argument('-o', '--opacity', help='opacity file or keyword', type=str, default='ricci_compact.npz')


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
    opac = ARGS.opacity

    lam = ARGS.lam
    lam2 = ARGS.lam2
    lam3 = ARGS.lam3
    lam4 = ARGS.lam4
    lam5 = ARGS.lam5
    lam6 = ARGS.lam6
    lam7 = ARGS.lam7
    lam8 = ARGS.lam8

    # if just q is given, use it for qd as well
    q = ARGS.q
    qd = ARGS.qd
    if qd is None:
        qd = q
        q_string = f'{q:0.1f}'
    else:
        q_string = f'{q:0.1f}_{qd:0.1f}'
    # then convert to array
    q = [q, qd]

    scattering = ARGS.scattering
    flux_fraction = ARGS.flux_fraction
    fname_in = ARGS.file
    fname_out = Path(fname_in)
    fname_out = fname_out.with_name(f'{fname_out.stem }_analysis_8wave_lam{1e4 * lam:0.0f}_q{q_string}_f{100 * flux_fraction:.0f}_s{int(scattering)}{fname_out.suffix}')

    return {
        'lam': lam,
        'lam2': lam2,
        'lam3': lam3,
        'lam4': lam4,
        'lam5': lam5,
        'lam6': lam6,
        'lam7': lam7,
        'lam8': lam8,
        'q': q,
        'flux_fraction': flux_fraction,
        'fname_in': fname_in,
        'fname_out': fname_out,
        'opac': opac,
        'scattering': scattering,
    }


def get_random_opac_file(filename):
    filename = Path(filename)
    if filename.suffix == '.npz':
        opacity = dipsy.Opacity(filename)
    elif filename.suffix == '.parquet':
        table = pd.read_parquet(filename)
        p = table["score"] / table["score"].sum()
        rng = np.random.default_rng()
        ind = rng.choice(np.arange(table.shape[0]), p=p, size=1)
        entry = table.iloc[ind[0]]
        opacity = dipsy.Opacity(entry.path)

    return opacity


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
    lam3 = settings['lam3']
    lam4 = settings['lam4']
    lam5 = settings['lam5']
    lam6 = settings['lam6']
    lam7 = settings['lam7']
    lam8 = settings['lam8']
    q = settings['q']
    scattering = settings['scattering']
    flux_fraction = settings['flux_fraction']

    opacity = get_random_opac_file(opac)

    # get the data from file and do the processing

    with h5py.File(fname, 'r') as fid:
        group = fid[key]
        b = bumpmodel_result(*[group[f][()] for f in bumpmodel_result._fields])
        params = group['params'][()]
    lambdas = np.array([lam, lam2, lam3, lam4, lam5, lam6, lam7, lam8])
    rf_t, flux_t, *_ = dipsy.get_all_observables(b, opacity, lambdas, q=q, flux_fraction=flux_fraction, scattering=scattering)
    # alpha_mm = np.log(flux_t[:,4] / flux_t[:,2]) / np.log(lam3 / lam5)

    if len(params) > 7 and len(params) < 9:
        rp1 = params[5]  # 1 planet case
        mp1 = params[6]  # 1 planet case
        rp2 = 0.
        mp2 = 0.
        N_substr = 1
        d2g = params[7]
    elif len(params) > 8:
        rp1 = params[5]  # 1 planet case
        mp1 = params[7]  # 1 planet case
        rp2 = params[6]  # 2 planets case
        mp2 = params[8]  # 2 planets case
        N_substr = 2
        d2g = params[9]
    else:
        rp1 = 0.
        mp1 = 0.
        rp2 = 0.
        mp2 = 0.
        N_substr = 0
        d2g = params[5]

    # pick a random snapshot

    # time_snap = 1.*10**(random.uniform(5,6+np.log10(3)))
    # t_snap = random.uniform(1e5, 3e6)
    # i_snap = b.time.searchsorted(t_snap * year)

    # try to read in the file with the simulation<->snapshot index association
    fname = Path(fname)
    snapshot_indices = pd.read_parquet(fname.with_name(fname.stem + '_times.parquet'))
    i_snap = int(snapshot_indices[snapshot_indices.key == key].it)

    r0 = 0.05 * au
    r1 = 2000. * au
    nr = 400

    r = np.logspace(np.log10(r0), np.log10(r1), nr + 1)

    M_gas = (np.pi * ((r[1:])**2 - (r[:-1])**2) * b.sig_g[i_snap][:-1]).sum(-1) / M_sun
    M_dust = (np.pi * ((r[1:])**2 - (r[:-1])**2) * b.sig_d[i_snap][:-1]).sum(-1) / M_sun

    out = {
        'N_substr': N_substr,
        'L': b.L,
        #'alpha': params[0], #not useful for sebastian
        'Mdisk': params[1] * params[4],  # now Mdisk is simply in M_sun units
        'r_c': params[2],
        #'v_frag': params[3], #not useful for sebastian
        'M_star': params[4],
        'rp1': rp1,
        'mp1': mp1,
        'rp2': rp2,
        'mp2': mp2,
        'd2g': d2g,
        'filename': opacity._filename,
        f'flux_t({1e1 * lam :0.2f}mm)': flux_t[i_snap, 0],
        f'flux_t({1e1 * lam2:0.2f}mm)': flux_t[i_snap, 1],
        f'flux_t({1e1 * lam3:0.2f}mm)': flux_t[i_snap, 2],
        f'flux_t({1e1 * lam4:0.2f}mm)': flux_t[i_snap, 3],
        f'flux_t({1e1 * lam5:0.2f}mm)': flux_t[i_snap, 4],
        f'flux_t({1e1 * lam6:0.2f}mm)': flux_t[i_snap, 5],
        f'flux_t({1e1 * lam7:0.2f}mm)': flux_t[i_snap, 6],
        f'flux_t({1e1 * lam8:0.2f}mm)': flux_t[i_snap, 7],
        f'rf_t({1e1 * lam :0.2f}mm)': rf_t[i_snap, 0] / au,
        f'rf_t({1e1 * lam2:0.2f}mm)': rf_t[i_snap, 1] / au,
        f'rf_t({1e1 * lam3:0.2f}mm)': rf_t[i_snap, 2] / au,
        f'rf_t({1e1 * lam4:0.2f}mm)': rf_t[i_snap, 3] / au,
        f'rf_t({1e1 * lam5:0.2f}mm)': rf_t[i_snap, 4] / au,
        f'rf_t({1e1 * lam6:0.2f}mm)': rf_t[i_snap, 5] / au,
        f'rf_t({1e1 * lam7:0.2f}mm)': rf_t[i_snap, 6] / au,
        f'rf_t({1e1 * lam8:0.2f}mm)': rf_t[i_snap, 7] / au,
        'time': b.time[i_snap] / (1.e6 * year),
        'M_dust': M_dust,
        'M_gas': M_gas,
    }

    return out


def create_snapshot_file(hdf_file, t0=100_000, t1=3e6):
    """for a simulation hdf5 file, create a parquet file with a random snapshot index
    between t0 and t1 (in years).
    """
    t0 *= year
    t1 *= year

    hdf_file = Path(hdf_file)

    with h5py.File(hdf_file, 'r') as fi:
        keys = list(fi.keys())
        time = fi[keys[0]]['time'][()]

    rand = np.random.default_rng()

    # draw random uniform times between t0 and t1
    t_rand = t0 + (t1 - t0) * rand.random(len(keys))

    # pick the time index and store key + index in a new table
    it_rand = np.abs(time[:, None] - t_rand[None, :]).argmin(0)
    table = pd.DataFrame({'key': keys, 'it': it_rand})

    # Store the table
    table.to_parquet(hdf_file.with_name(hdf_file.stem + '_times.parquet'))
