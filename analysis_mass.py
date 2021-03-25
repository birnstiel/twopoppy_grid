"""
Analyze the grid simulations to derive the emission
profiles for a given time and then the mass applying
the Powell method
"""
import argparse
from pathlib import Path

import numpy as np

import dipsy
year = dipsy.cgs_constants.year

lams = np.array([0.087, 0.1, 0.13, 0.8])

# local imports need to be imported in a special way

estimator = dipsy.utils.remote_import(Path(__file__).absolute().parent / 'estimator.py')
model = dipsy.utils.remote_import(Path(__file__).absolute().parent / 'model.py')
bumpmodel_result = model.bumpmodel_result

au = dipsy.cgs_constants.au

# %% define a second parser that processes analyis-specific options
RTHF = argparse.RawTextHelpFormatter
PARSER = argparse.ArgumentParser(description=__doc__, formatter_class=RTHF)
PARSER.add_argument('file', help='HDF5 file with the simulation data', type=str)
PARSER.add_argument('-q', '--q', help='size distribution slope', type=float, default=3.5)
PARSER.add_argument('-f', '--function', help='which function to use', type=int, default=3)
PARSER.add_argument('--time', help='simulation time [yr]', type=float, default=1e6)
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

    q = ARGS.q
    flux_fraction = ARGS.flux_fraction
    time = ARGS.time * year
    fname_in = ARGS.file
    fname_out = Path(fname_in)
    fname_out = fname_out.with_name(f'{fname_out.stem }_mass_{ARGS.time:.1e}yr{fname_out.suffix}')

    return {
        'time': time,
        'q': q,
        'flux_fraction': flux_fraction,
        'fname_in': fname_in,
        'fname_out': fname_out,
        'opac': opac,
        'fct_nr': ARGS.function
    }


def parallel_analyze(key, settings=None, debug=False, **kwargs):
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
    try:
        if settings is None:
            raise ValueError('settings must be given as keyword')

        # get all settings
        fname = settings['fname_in']
        opac = settings['opac']
        time = settings['time']
        q = settings['q']
        fct_nr = settings['fct_nr']
        flux_fraction = settings['flux_fraction']

        # get the data from file and do the processing

        sim = dipsy.utils.read_from_hdf5(fname, key)
        params = sim['params']
        it = sim['time'].searchsorted(time)

        obs = dipsy.get_observables(
            sim['r'],
            sim['sig_g'][it],
            sim['sig_d'][it],
            sim['a_max'][it],
            sim['T'][it],
            opac,
            lams,
            flux_fraction=flux_fraction,
            q=q
        )

        # now fit for the dust lines

        dipsy.fortran.crop = 1e-10

        r_dust = []
        r_best = []
        discards = []
        samplers = []
        xs = []
        ys = []

        kwargs['progress'] = kwargs.get('progress', False)
        kwargs['n_steps'] = kwargs.get('n_steps', 1000)
        kwargs['n_burnin'] = kwargs.get('n_burnin', 100)

        for ilam in np.arange(len(lams)):
            x = sim['r'] / au
            y = obs.I_nu[ilam]  # + RMS * np.random.randn(len(x))

            _r_dust, _dr_dust, _r_best, discard, sampler = estimator.get_dust_line(
                x, y,
                fct_nr=fct_nr,
                **kwargs
            )

            # slice = sampler.lnprobability[:, discard:]
            # idx = np.unravel_index(slice.argmax(), slice.shape)
            # ln_best = slice[idx[0], idx[1]]
            # p_best = sampler.chain[:, discard:, :][idx[0], idx[1], :]

            r_dust += [_r_dust]
            r_best += [_r_best]
            discards += [discard]
            samplers += [sampler]
            xs += [x]
            ys += [y]

        # now given the r_dust, apply powel method to get surface density

        M_star = sim['M_star']
        sig_g = estimator.sigma_estimator(r_dust, lams, time, M_star)

        # FIRST APPROACH: estimate the disk mass from those few surface densities

        r_dust_s = np.array(r_dust)
        sig_g = np.array(sig_g)
        idx = r_dust_s.argsort()
        M_d_est = np.trapz(
            2 * np.pi * r_dust_s[idx] * au * sig_g[idx],
            x=r_dust_s[idx] * au)

        M_d = np.trapz(2 * np.pi * sim['r'] * sim['sig_g'][it], x=sim['r'])

        # SECOND APPROACH: fit a LBP profile and integrate that.

        p_best, M_lbp, masses, sampler_lbp = estimator.fit_lbp(r_dust_s[idx] * au, sig_g[idx])
        # store the relevant results in a dict

        out = {
            'alpha': params[0],
            'Mdisk': params[1],
            'r_c': params[2],
            'v_frag': params[3],
            'M_star': params[4],
            'r_dust': r_best,
            'sig_g': sig_g,
            'M_est': M_d_est,
            'M_gas': M_d,
            'M_lbp': M_lbp,
            'M_lbp_med': np.median(masses),
            'N_lbp': p_best[0],
            'rc_lbp': p_best[1],
            'p_lbp': p_best[2],
        }

        if debug:
            sig_lbp = dipsy.fortran.lbp_profile(p_best, sim['r'])

            out['x'] = xs
            out['y'] = ys
            out['obs'] = obs
            out['sim'] = sim
            out['samplers'] = samplers
            out['sampler_lbp'] = sampler_lbp
            out['sig_lbp'] = sig_lbp
            out['discards'] = discards

        return out
    except Exception as err:
        print(err)
        return False
