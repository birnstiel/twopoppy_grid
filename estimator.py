from multiprocessing import Pool
import warnings

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

# These wrappers are needed such that the functions can be pickeled
# as fortran functions cannot be pickled but the python wrapper can.


def lnp_pwr1_wrapper(p, x, y):
    return dipsy._fortran_module.fmodule.lnp_pwr1(p, x, y)


def lnp_pwr2_wrapper(p, x, y):
    return dipsy._fortran_module.fmodule.lnp_pwr2(p, x, y)


def lnp_pwr2_logit_wrapper(p, x, y):
    return dipsy._fortran_module.fmodule.lnp_pwr2_logit(p, x, y)


def fit_intensity(x, y, n_threads=1, n_burnin=50, n_steps=500, n_walker=None, fct_nr=1, **kwargs):
    """fit power law

    Parameters
    ----------
    x : array
        x array in units of au
    y : array
        intensity
    n_threads : int, optional
        number of threads, by default 1
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

    # we use only emission outside 1 au which is a roughly
    # realistic best-case beam size

    mask = x > 1
    x = x[mask]
    y = y[mask]

    # set up problem size

    if fct_nr == 1:
        n_dim = 3
        if n_walker is None:
            n_walker = 15

        #  initial parameters
        def get_p0():
            return np.array([
                y[0] * 10.**(-3 + 6 * np.random.rand(n_walker)),
                -4 + 8 * np.random.rand(n_walker),
                10.**(1 + 2 * np.random.rand(n_walker)),
            ]).T

        fct = lnp_pwr1_wrapper
    elif fct_nr == 2:
        n_dim = 6
        if n_walker is None:
            n_walker = 15

        #  initial parameters
        def get_p0():
            return np.array([
                y[0] * 10.**(-3 + 6 * np.random.rand(n_walker)),
                -4 + 8 * np.random.rand(n_walker),
                -4 + 8 * np.random.rand(n_walker),
                10.**(2 + 2 * np.random.rand(n_walker)),
                np.random.rand(n_walker),
                10.**(2 * np.random.rand(n_walker)),
            ]).T

        fct = lnp_pwr2_wrapper
    elif fct_nr == 3:
        n_dim = 7
        if n_walker is None:
            n_walker = 40

        #  initial parameters
        # def get_p0():
        #     return np.array([
        #         y[0] * 10.**(-6 + 6 * np.random.rand(n_walker)),
        #         np.random.rand(n_walker),
        #         0 + 3 * np.random.rand(n_walker),
        #         0 + 3 * np.random.rand(n_walker),
        #         10.**(2 + np.random.rand(n_walker)),
        #         10.**(2 * np.random.rand(n_walker)),
        #         10.**(1 + 1.3 * np.random.rand(n_walker)),
        #     ]).T
        fct = lnp_pwr2_logit_wrapper

        def get_p0():
            p0 = guess(x, y, n_walker)
            lnps = [lnp_pwr2_logit_wrapper(_p, x, y) for _p in p0]
            p_best = p0[np.argmax(lnps)]
            return get_valid_walker_cloud(p_best, n_walker, fct, x, y, delta=0.1, threshold=0.05)

    else:
        raise NameError('unknown function number')

    # initialize pool

    if n_threads > 1:
        pool = Pool(n_threads)
    else:
        pool = None

    sampler = emcee.EnsembleSampler(n_walker, n_dim, fct, pool=pool, args=[x, y])

    try:
        # run the burn-in: we take only the best and then pick samples around it
        logp = -np.inf

        while np.isneginf(logp):
            p0 = get_p0()
            sampler.reset()
            _ = sampler.run_mcmc(p0, n_burnin, **kwargs)
            i_best = sampler.lnprobability[:, -1].argmax()
            logp = sampler.lnprobability[i_best, -1]
            p_best = sampler.chain[i_best, -1, :]

        p1 = get_valid_walker_cloud(p_best, n_walker, fct, x, y, delta=0.1, threshold=0.05)

        # now we run again with that new cloud of points

        sampler.reset()
        sampler.run_mcmc(p1, n_steps, **kwargs)
    except Exception:
        pass
    finally:
        if pool is not None:
            pool.close()

    return sampler


def get_valid_walker_cloud(p0, n_walkers, fct, *args, delta=0.3, threshold=0.1, n_iter=1000):

    logP0 = fct(p0, *args)

    def get_p_new(p0, delta):
        return [p * (1 - delta + 2 * delta * np.random.randn()) for p in p0]

    result = []

    for i in range(n_walkers):
        p_new = get_p_new(p0, delta)
        counter = 0
        while fct(p_new, *args) < threshold * logP0 and counter < n_iter:
            p_new = get_p_new(p0, delta)
            counter += 1
        if counter == n_iter:
            warnings.warn('exceeded max. number of iterations')
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


def get_dust_line(x, y, n_threads=1, n_steps=2500, **kwargs):
    """determine the dust line for the given simulation

    Parameters
    ----------
    x : array
        x array in au
    y : array
        intensity array
    n_threads : int, optional
        number of threads, by default 1
    n_steps : int, optional
        number of iterations, by default 2500

    other keywords are passed to fit_intensity

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
    sampler = fit_intensity(x, y, n_threads=n_threads, n_steps=n_steps, **kwargs)

    act = sampler.get_autocorr_time(quiet=True, tol=10, discard=300)
    try:
        discard = int(2 * np.nanmax(act))
    except ValueError:
        discard = 300

    if discard > 0.5 * sampler.chain.shape[1]:
        raise ValueError('too many samples to discard')

    slice = sampler.lnprobability[:, discard:]
    idx = np.unravel_index(slice.argmax(), slice.shape)
    r_best = sampler.chain[:, discard:, -1][idx[0], idx[1]]

    r_dust = sampler.chain[:, discard:, -1].mean()
    dr_dust = sampler.chain[:, discard:, -1].std()

    return r_dust, dr_dust, r_best, discard, sampler


def findLocalMaximaMinima(arr):
    """
    Function to find all the local maxima
    and minima in the given array arr[]
    """
    n = len(arr)
    # Empty lists to store points of
    # local maxima and minima
    mx = []
    mn = []

    # Checking whether the first point is
    # local maxima or minima or neither
    if(arr[0] > arr[1]):
        mx.append(0)
    elif(arr[0] < arr[1]):
        mn.append(0)

    # Iterating over all points to check
    # local maxima and local minima
    for i in range(1, n - 1):

        # Condition for local minima
        if(arr[i - 1] > arr[i] < arr[i + 1]):
            mn.append(i)

        # Condition for local maxima
        elif(arr[i - 1] < arr[i] > arr[i + 1]):
            mx.append(i)

    # Checking whether the last point is
    # local maxima or minima or neither
    if(arr[-1] > arr[-2]):
        mx.append(n - 1)
    elif(arr[-1] < arr[-2]):
        mn.append(n - 1)

    return mx, mn


def guess(x, y, n, n_smooth=5, debug=False):
    """return a list of n guesses for the 7-parameter function.

    Parameters
    ----------
    x : array
        radius in au
    y : arra
        intensity on x-grid
    n : int
        number of guesses
    n_smooth : int, optional
        over how many cells to smooth the exponent, by default 5

    Returns
    -------
    array
        a list of guesses for all 7 parameters
    """
    mask = x > 1
    x = x[mask]
    y = y[mask] + 1e-100

    # get the exponent

    exponent = (x / y)[1:-1] * (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    x = x[1:-1]
    y = y[1:-1]

    # smooth the exponent

    exponent2 = np.convolve(exponent, np.ones(n_smooth) / n_smooth, mode='same')

    # guess the outer truncation

    try:
        r_out = x[np.where(exponent2 < -20)[0][0]]
    except IndexError:
        try:
            r_out = x[np.where(y < dipsy.fortran.crop)[0][0]]
        except IndexError:
            r_out = x[y.argmin()]

    # find the two most common slopes

    bins = np.arange(-5.25, 0.25, 0.5)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(exponent, bins=bins)

    i = counts.argsort()[::-1]
    slope1 = centers[i[0]]
    slope2 = centers[i[1]]

    # find which slope is inner/outer

    inner = exponent[:100].mean()

    if abs(inner - slope1) < abs(inner - slope2):
        slope_i = slope1
        slope_o = slope2
    else:
        slope_i = slope2
        slope_o = slope1

    # guesstimate the slope transition

    mask = x < r_out
    i1 = np.abs(exponent2 - slope_i)[mask].argmin()
    i2 = np.abs(exponent2 - slope_o)[mask].argmin()
    r_t = 0.5 * (x[mask][i1] + x[mask][i2])

    # find local minima in the slope
    # as candidate transition radii

    _, mn = findLocalMaximaMinima(exponent2)
    mn = np.array(mn)
    if len(mn) == 0:
        mn = np.array([exponent2.argmin()])
    mask2 = x[mn] < 0.8 * r_out
    if mask2.sum() > 0:
        mn = mn[mask2]
    else:
        r_out = np.hstack((r_out, x[mn])).max()
    r_list = x[mn]

    r_dust = r_list[exponent2[mn].argmin()]

    # construct the list of guesses
    r_t_array = np.random.choice(np.hstack(([r_t, r_list])), size=n)

    p = np.array([
        np.interp(r_t_array, x, y) * 10.**(-1 + 2 * np.ones(n)),
        np.random.rand(n),
        -slope_o * np.ones(n),
        -slope_i * np.ones(n),
        r_out * (0.9 + 0.2 * np.random.rand(n)),
        r_t_array,
        r_dust * np.ones(n)
    ])

    if debug:
        return p.T, {
            'x': x,
            'exponent2': exponent2,
            'r_t': r_t,
            'r_out': r_out,
            'r_dust': r_dust,
            'r_list': r_list,
            'x': x,
            'y': y,
        }
    else:
        return p.T
