import dipsy
# import pandas as pd
import numpy as np

au = dipsy.cgs_constants.au

opac = dipsy.Opacity(input='ricci_compact.npz')
# opac  = dipsy.Opacity(input='/home/moon/birnstiel/DATA/zormpas/dsharp_p0.90_smooth_extrapol.npz')
# opac  = dipsy.Opacity(input='/home/moon/birnstiel/DATA/zormpas/dsharp_p0.00_smooth_extrapol.npz')
# opac  = dipsy.Opacity(input='ricci_compact.npz')

lam = 0.085
q = 3.5
flux_fraction = 0.99


def parallel_analyze(param, res):

    rf_t, flux_t, *_ = dipsy.get_all_observables(res, opac, lam, q=q, flux_fraction=flux_fraction)

    out = {
        'alpha': param[0],
        'Mdisk': param[1],
        'r_c': param[2],
        'v_frag': param[3],
        'M_star': param[4],
        'rf_t': np.squeeze(rf_t / au).tolist(),
        'flux_t': np.squeeze(flux_t).tolist(),
        'sig_g': res.sig_g,
        'sig_d': res.sig_d
    }

    return out
