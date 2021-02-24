from multiprocessing import Pool

import numpy as np
import emcee

import dipsy

au = dipsy.cgs_constants.au
year = dipsy.cgs_constants.year
M_sun = dipsy.cgs_constants.M_sun
sig_sb = dipsy.cgs_constants.sigma_sb
mu = dipsy.cgs_constants.mu
m_p = dipsy.cgs_constants.m_p
k_b = dipsy.cgs_constants.k_B
G = dipsy.cgs_constants.Grav


def sigma_estimator(r_dust, lams, t, M_star):

    L_star, R_star, T_star = dipsy.tracks.get_stellar_properties(M_star, t)

    phi = 0.05
    rho_s = 1.2

    sig_g = []

    for r_d, lam in zip(r_dust, lams):
        r_d *= au

        # calculate the temperature

        Temp = ((phi * L_star / (4 * np.pi * sig_sb * r_d**2)) + 1e4)**0.25

        cs = np.sqrt(k_b * Temp / (mu * m_p))
        vk = np.sqrt(G * M_star / r_d)
        v0 = cs**2 / (2 * vk)

        sig_g += [t * v0 * rho_s * lam / r_d]

    return sig_g


def logp_pwr_wrapper(p, x, y):
    return dipsy._fortran_module.fmodule.lnp_pwr(p, x, y)


def logp_pwr2_wrapper(p, x, y):
    return dipsy._fortran_module.fmodule.lnp_pwr2(p, x, y)


def fit_pwr(x, y, nthreads=0, n_burnin=50, n_steps=500, n_walker=15, fct_nr=1, **kwargs):
    """fit power law

    Parameters
    ----------
    x : array
        x array in units of au
    y : array
        intensity
    nthreads : int, optional
        number of threads, by default 0
    n_burnin : int, optional
        number of burn-in steps, by default 50
    n_steps : int, optional
        number of sampling iterations, by default 500

    other keywords are passed to run_mcmc

    Returns
    -------
    sampler
        emcee sampler object
    """

    # we first use only emission outside 1 au which is a roughly
    # realistic best-case beam size

    mask = x > 1
    x = x[mask]
    y = y[mask]

    # set up problem size

    if fct_nr == 1:
        n_dim = 3

        #  initial parameters
        p0 = np.array([
            y[0] * 10.**(-3 + 6 * np.random.rand(n_walker)),
            -4 + 8 * np.random.rand(n_walker),
            -2 + 5 * np.random.rand(n_walker),
        ]).T

        fct = logp_pwr_wrapper
    elif fct_nr == 2:
        n_dim = 6

        #  initial parameters
        p0 = np.array([
            y[0] * 10.**(-3 + 6 * np.random.rand(n_walker)),
            -4 + 8 * np.random.rand(n_walker),
            10.**(2.5 * np.random.rand(n_walker)),
            np.random.rand(n_walker),
            -4 + 8 * np.random.rand(n_walker),
            10.**(2.3 * np.random.rand(n_walker)),
        ]).T

        fct = logp_pwr2_wrapper
    else:
        raise NameError('unknown function number')

    # initialize pool

    if nthreads > 0:
        pool = Pool(nthreads)
    else:
        pool = None

    sampler = emcee.EnsembleSampler(n_walker, n_dim, fct, pool=pool, args=[x, y])

    try:
        # run the burn-in: we take only the best and then pick samples around it

        _ = sampler.run_mcmc(p0, n_burnin, **kwargs)
        i_best = sampler.lnprobability[:, -1].argmax()
        p_best = sampler.chain[i_best, -1, :]
        p1 = get_valid_walker_cloud(p_best, n_walker, fct, x, y)

        # now we run again with that new cloud of points

        sampler.reset()
        sampler.run_mcmc(p1, n_steps, **kwargs)
    except Exception:
        pass
    finally:
        if pool is not None:
            pool.close()

    return sampler


def get_valid_walker_cloud(p0, n_walkers, fct, *args, delta=0.3, threshold=0.1):

    logP0 = fct(p0, *args)

    def get_p_new(p0, delta):
        return [p * (1 - delta) + 2 * delta * np.random.rand() for p in p0]

    result = []

    for i in range(n_walkers):
        p_new = get_p_new(p0, delta)
        while fct(p_new, *args) > threshold * logP0:
            p_new = get_p_new(p0, delta)
        result.append(p_new)

    return result


def get_sigma_area(sampler, fct, x, X=20, f=0.68):
    """get all samples from the last X iterations and flatten, then sort

     Parameters
     ----------
     sampler : sampler
         the samples from emcee
     fct : function
         callable function returning the model given parameters and
         arguments like fct(x, *p)
     X : int, optional
         number of last iterations to include, by default 20
     f : float, optional
         fraction of best samples to plot, by default 0.68

     Returns
     -------
     y_min, y_max : arrays
         returns the minimum and maximum values
     """

    lastP = sampler.chain[:, -X:, :]
    lastL = sampler.lnprobability[:, -X:]

    flatP = lastP.reshape([np.product(lastL.shape[:2]), -1])
    flatL = lastL.ravel()

    sortP = flatP[flatL.argsort()[::-1], :]

    n_sigma = int(f * len(flatL))

    y_max = -np.inf * np.ones_like(x)
    y_min = np.inf * np.ones_like(x)

    for p in sortP[:n_sigma]:
        _y = fct(x, *p)
        y_max = np.maximum(y_max, _y)
        y_min = np.minimum(y_min, _y)

    return y_min, y_max


def get_dust_line(x, y, nthreads=20, n_steps=2500, **kwargs):
    """determine the dust line for the given simulation

    Parameters
    ----------
    x : array
        x array in au
    y : array
        intensity array
    nthreads : int, optional
        number of threads, by default 20
    n_steps : int, optional
        number of iterations, by default 2500

    other keywords are passed to fit_pwr

    Returns
    -------
    r_dust, dr_dust
        float: dust radius and uncertainty

    sampler

    Raises
    ------
    ValueError
        if too many sample are discarded
    """
    sampler = fit_pwr(x, y, nthreads=nthreads, n_steps=n_steps, **kwargs)

    act = sampler.get_autocorr_time(quiet=True, tol=10)
    discard = int(2 * act.max())
    if discard > 0.5 * sampler.chain.shape[1]:
        raise ValueError('too many samples to discard')

    r_dust = sampler.chain[:, discard:, -1].mean()
    dr_dust = sampler.chain[:, discard:, -1].std()

    return r_dust, dr_dust, discard, sampler
