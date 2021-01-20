from collections import namedtuple

import numpy as np
import astropy.units as u

import twopoppy2
import dipsy

k_b = dipsy.cgs_constants.k_B
Grav = dipsy.cgs_constants.Grav
pc = dipsy.cgs_constants.pc
au = dipsy.cgs_constants.au
mu = dipsy.cgs_constants.mu
m_p = dipsy.cgs_constants.m_p
c_light = dipsy.cgs_constants.c_light
year = dipsy.cgs_constants.year
M_sun = dipsy.cgs_constants.M_sun

bumpmodel_result = namedtuple('bumpmodel_result', ['r', 'time', 'M_star', 'sig_d', 'sig_g', 'T', 'a_dr', 'a_fr', 'a_gr', 'a_df', 'a_max', 'alpha', 'alpha_gas'])

Kelvin = u.Kelvin
arcsec_sq = (u.arcsec**2).to(u.sr)  # arcsec**2 in steradian


def run_bump_model2(alpha, M_disk, r_c, v_frag, M_star, r, T, time, r_bumps, a_bumps, mass_ratio, get_model=False, gap_profile='kanagawa'):
    """
    run a model with the given parameters.

    ALPHA : float
        turbulence parameter

    MD : float
        disk mass [g]

    rc : float
        characteristic radius [cm]

    v_frag : float
        fragmentation velocity [cm / s]

    M_star : float
        stellar mass [g]

    r : array
        radial grid [cm]

    T : array
        temperature array [K]

    Returns:
    --------

    dict with keys:
    - r
    - t
    - M_star
    - sig_d
    - sig_g
    - T_out
    - a_dr
    - a_fr
    - a_gr
    - a_df
    - a_t

    if get_model is True: return only the model
    """

    ri = np.logspace(np.log10(r[0]), np.log10(r[-1]), len(r) + 1)
    grid = twopoppy2.Grid(ri)
    T = 10.**np.interp(np.log10(grid.r), np.log10(r), np.log10(T))

    values = model_setup(alpha, M_disk, r_c, v_frag, M_star, grid.r, T, r_bumps, a_bumps, mass_ratio, gap_profile)

    m = twopoppy2.Twopoppy(grid=grid, a_0=values['a_0'])

    m.sigma_g = values['sig_g']
    m.sigma_d = values['sig_d']
    m.set_all_alpha(values['alpha'])
    m.alpha_gas = values['alpha_gas']
    m.M_star = values['M_star']
    m.v_frag = values['v_frag']
    m.rho_s = values['rho_s']
    m.T_gas = T

    m.stokesregime = 0
    m.snapshots = time
    m.initialize()
    m.run()

    if get_model:
        return m
    else:
        return bumpmodel_result(
            m.r, m.snapshots, m.M_star, m.data['sigma_d'], m.data['sigma_g'],
            m.data['T_gas'], m.data['a_dr'], m.data['a_fr'], m.data['a_1'],
            m.data['a_df'], m.data['a_1'], alpha, m.alpha_gas)


def kanagawa_profile(x, alpha, aspect_ratio, mass_ratio, smooth=2.5):
    """Kanagawa planetary gap profile.

    Returns the Kanagawa profile for a planetary gap on the array
    x where x = r / R_p.

    Arguments:
    ----------

    x : array
        radial coordinate in terms of the planet position

    alpha : float
        tubrulence parameter

    aspect_ratio : float
        h / r at the position of the planet

    mass_ratio : float
        planet to star mass ratio

    Output:
    -------
    surface density profile in units of the original surface density.
    """

    K_prime = mass_ratio**2 * aspect_ratio**-3 / alpha
    K = K_prime / (aspect_ratio**2)
    fact_min_0 = 1 / (1 + 0.04 * K)
    R1 = (fact_min_0 / 4 + 0.08) * K_prime**(1 / 4)
    R2 = 0.33 * K_prime**(1 / 4)
    fact = np.ones_like(x)
    mask = np.abs(x - 1) < R2
    fact[mask] = 4.0 * K_prime**(-1 / 4) * np.abs(x[mask] - 1) - 0.32
    fact[np.abs(x - 1) < R1] = fact_min_0

    # smoothing
    """
    smooth = 2.5
    x_h    = (mass_ratio / 3.)**(1. / 3.)
    x_s    = smooth * x_h
    fact   = np.exp(np.log(fact) * np.exp(-0.5 * (x - 1)**4 / x_s**4))
    #fact   = np.exp(np.log(fact) * np.exp(-(np.abs(x - 1)/2.3)**3 / R1**4))
    """
    return {'fact': fact, 'R1': R1, 'R2': R2, 'K': K, 'K_prime': K_prime}


def model_setup(ALPHA, M_disk, r_c, v_frag, M_star, r, Temp, r_bumps, a_bumps, masses, gap_profile='kanagawa'):
    """
    Calculates values that are used in the model.property

    Arguments
    ---------
    ALPHA : float
        alpha turbulence parameters

    M_disk : float
        disk mass in stellar masses

    r_c: float
        characteristic radius in au

    v_frag : float
        fragmentation velocity [cm / s]

    M_star: float
        stellar mass in solar masses

    r : array
        radial grid [cm]

    T : array
        temperature grid [cm]

    r_bumps : float or array
        where the bump(s) are located in units of r_c (eg. 1./3.)

    a_bumps : float or array
        the amplitude of the bump(s)

    masses  : float or array
        the mass ratio of planets to star

    Returns:
    --------
    a_0         : initial particle size

    sig_g       : initial condition : gas surface desnity [g / cm^2]

    sig_d       : initial condition : gas surface desnity [g / cm^2]

    v_gas       : initial condition : gas velocity

    alpha       : alpha parameter for dust

    M_star      : stellar mass [g]

    M_disk      : disk mass [g]

    v_frag      : fragmentation velocity [cm/s]

    rho_s       : material density

    alpha_gas   : alpha parameter for gas
    """

    # convert input to CGS

    if type(M_star) == list:
        M_star = [i * M_sun for i in M_star]
    else:
        M_star *= M_sun

    if type(M_disk) == list:
        M_disk = [i * M_star for i in M_disk]
    else:
        M_disk *= M_star

    if type(r_c) == list:
        r_c = [i * au for i in r_c]
    else:
        r_c *= au

    rho_s = 1.2              # material density of in Drazkowska
    d2g = 0.01             # dust-to-gas ratio [-]
    a_0 = 1e-5             # initial particle size [cm]

    a_bumps = np.array(a_bumps, ndmin=1)
    r_bumps = np.array(r_bumps, ndmin=1)
    masses = np.array(masses, ndmin=1)
    # Count the number of planets
    num_planets = np.count_nonzero(r_bumps)
    # Calculate the pressure scale height at the position of the bump
    cs = np.sqrt(k_b * Temp / mu / m_p)
    o_k = np.sqrt(Grav * M_star / r**3)
    H = cs / o_k

    H_bumps = np.interp(r_bumps * r_c, r, H)

    # initial conditions: gas and dust surface densities [g cm^-2]
    sig_g = (r / r_c)**-1 * np.exp(-(r / r_c))
    sig_g = sig_g / np.trapz(2 * np.pi * r * sig_g, x=r) * M_disk
    sig_g = np.maximum(sig_g, 1e-100)
    sig_d = d2g * sig_g

    # alpha profile for the dust
    alpha = ALPHA * np.ones_like(r)

    # Calculate the values a and b for the gaussian bumps
    amp = a_bumps * ALPHA

    R1 = []
    R2 = []
    # alpha profile for the gas with two possible gaps using the local H_rc**2 as variance
    # and the location of the bump as a function of r_c

    # zip the parameters needed for kanagawa_profile
    mapped = list(zip(r_bumps, H_bumps, masses))

    if gap_profile == 'kanagawa':
        factor = np.ones_like(r)
        for i in range(num_planets):
            r_p, h, mass = mapped[i][0], mapped[i][1], mapped[i][2]
            kanagawa = kanagawa_profile(r / (r_p * r_c), ALPHA, h / (r_p * r_c), mass)
            factor *= kanagawa['fact']
            R1 += [kanagawa['R1']]
            R2 += [kanagawa['R2']]
        alpha_gas = alpha / factor
    elif gap_profile == 'gap':
        alpha_gas = alpha + (amp[None, :] * np.exp(- (r[:, None] - r_bumps[None, :] * r_c)**2 / (2 * H_bumps[None, :]**2))).sum(-1)
    elif gap_profile == 'bump':
        alpha_gas = alpha - (amp[None, :] * np.exp(- (r[:, None] - r_bumps[None, :] * r_c)**2 / (2 * H_bumps[None, :]**2))).sum(-1)
    elif gap_profile is None:
        alpha_gas = alpha
    else:
        raise NameError(f'unknown model {gap_profile}')

    # v_gas is only for non-evolving gas disks, where you still
    # want the dust to feel a gas velocity (e.g. a steady-state disk)
    v_gas = np.zeros_like(r)

    return {
        'a_0': a_0,
        'sig_g': sig_g,
        'sig_d': sig_d,
        'v_gas': v_gas,
        'alpha': alpha,
        'alpha_gas': alpha_gas,
        'M_star': M_star,
        'M_disk': M_disk,
        'v_frag': v_frag,
        'rho_s': rho_s,
        'r_c': r_c}
