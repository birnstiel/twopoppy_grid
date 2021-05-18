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

_lams = np.array([0.087, 0.1, 0.13, 0.3])

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
PARSER.add_argument('-l', '--lams', help='which wavelengths to use [cm]', type=float, nargs='+' ,default=list(_lams))


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
    lams = np.array(ARGS.lams)
    time = ARGS.time * year
    fname_in = ARGS.file
    fname_out = Path(fname_in)
    fname_out = fname_out.with_name(f'{fname_out.stem }_mass_{ARGS.time:.1e}yr_f{ARGS.function}{fname_out.suffix}')

    return {
        'time': time,
        'q': q,
        'flux_fraction': flux_fraction,
        'lams': lams,
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
        lams = settings['lams']

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
        sig_g, sig_d = estimator.sigma_estimator(r_dust, lams, time, M_star)

        # FIRST APPROACH: estimate the disk mass from those few surface densities

        r_dust = np.array(r_dust)
        sig_g = np.array(sig_g)
        sig_d = np.array(sig_d)

        idx = r_dust.argsort()

        M_g_est = np.trapz(
            2 * np.pi * r_dust[idx] * au * sig_g[idx],
            x=r_dust[idx] * au)

        M_d_est = np.trapz(
            2 * np.pi * r_dust[idx] * au * sig_d[idx],
            x=r_dust[idx] * au)

        M_dust = np.trapz(2 * np.pi * sim['r'] * sim['sig_d'][it], x=sim['r'])
        M_gas = np.trapz(2 * np.pi * sim['r'] * sim['sig_g'][it], x=sim['r'])

        # SECOND APPROACH: fit a LBP profile and integrate that.

        p_gas, M_g_lbp, masses_g, sampler_g = estimator.fit_lbp(
            r_dust[idx] * au, sig_g[idx])

        p_dust, M_d_lbp, masses_d, sampler_d = estimator.fit_lbp(
            r_dust[idx] * au, sig_d[idx])
        # store the relevant results in a dict

        out = {
            'alpha': params[0],
            'Mdisk': params[1],
            'r_c': params[2],
            'v_frag': params[3],
            'M_star': params[4],
            'r_dust': r_best,
            'sig_g': sig_g,
            'sig_d': sig_d,
            'lams': lams,
            'M_g_est': M_g_est,
            'M_d_est': M_d_est,
            'M_gas': M_gas,
            'M_dust': M_dust,
            'M_g_lbp': M_g_lbp,
            'M_d_lbp': M_d_lbp,
            'M_g_med': np.median(masses_g),
            'M_d_med': np.median(masses_d),
            'N_g': p_gas[0],
            'rc_g': p_gas[1],
            'p_g': p_gas[2],
            'N_d': p_dust[0],
            'rc_d': p_dust[1],
            'p_d': p_dust[2],
        }

        if debug:
            sig_g_lbp = dipsy.fortran.lbp_profile(p_gas, sim['r'])
            sig_d_lbp = dipsy.fortran.lbp_profile(p_dust, sim['r'])

            out['x'] = xs
            out['y'] = ys
            out['obs'] = obs
            out['sim'] = sim
            out['samplers'] = samplers
            out['sampler_g'] = sampler_g
            out['sampler_d'] = sampler_d
            out['sig_g_lbp'] = sig_g_lbp
            out['sig_d_lbp'] = sig_d_lbp
            out['masses_g'] = masses_g
            out['masses_d'] = masses_d
            out['discards'] = discards

        return out
    except Exception as err:
        print(err)
        return False
