"""
external_input
==============

Mean (mu) and standard deviation (sigma) of the external Poisson input of
every population, in mV:

    mu    = K_ext * J_ex * tau_mem * rate_ext * factor
    sigma = sqrt(K_ext * J_ex**2 * tau_mem * rate_ext * factor)

K_ext is the external indegree from visuomotor_indegree_data.json,
J_ex the PSP of one external spike (mV), tau_mem the membrane time constant
(ms, converted to s) and rate_ext the rate of one external input (spikes/s).
factor is fac_nu_ext_5E / fac_nu_ext_6E for the 5E / 6E populations and 1
otherwise.

build_mean_input_file writes mu to mu_values.json as
mu[area][pop], the format that MesoscopicModel reads.
"""
import json
import os

import numpy as np

MU_PARAMS_FILE = 'mu_values_params.json'


def external_input_factors(indegree_dict, tau_mem, J_ex, fac_nu_ext_5E,
                           fac_nu_ext_6E, area_list, pop_list):
    """Drift and diffusion factors of the external input, [area][pop]."""
    drift, diffusion = {}, {}
    for area in area_list:
        drift[area], diffusion[area] = {}, {}
        for pop in pop_list:
            K_ext = indegree_dict[area][pop]['external']['external']
            fac = {'5E': fac_nu_ext_5E, '6E': fac_nu_ext_6E}.get(pop, 1.)
            drift[area][pop] = K_ext * J_ex * tau_mem * 1e-3 * fac
            diffusion[area][pop] = K_ext * J_ex ** 2 * tau_mem * 1e-3 * fac
    return drift, diffusion


def mu_sigma_external(indegree_dict, tau_mem, J_ex, rate_ext, fac_nu_ext_5E,
                      fac_nu_ext_6E, area_list, pop_list):
    """Returns mu[area][pop] and sigma[area][pop] in mV."""
    drift, diffusion = external_input_factors(
        indegree_dict, tau_mem, J_ex, fac_nu_ext_5E, fac_nu_ext_6E,
        area_list, pop_list)
    mu = {a: {p: drift[a][p] * rate_ext for p in pop_list} for a in area_list}
    sigma = {a: {p: float(np.sqrt(diffusion[a][p] * rate_ext))
                 for p in pop_list} for a in area_list}
    return mu, sigma


def build_mean_input_file(data_dir, data_files, input_params, area_list,
                          pop_list):
    """
    Compute mu from the indegree file and save it to data_files['mean_input'].
    The input parameters are saved next to it (mu_values_params.json) so that
    a change of the parameters can be detected later.
    """
    with open(os.path.join(data_dir, data_files['ext_indegrees']), 'r') as f:
        indegree_dict = json.load(f)
    mu, _ = mu_sigma_external(
        indegree_dict, input_params['tau_mem'], input_params['J_ex'],
        input_params['rate_ext'], input_params['fac_nu_ext_5E'],
        input_params['fac_nu_ext_6E'], area_list, pop_list)
    with open(os.path.join(data_dir, data_files['mean_input']), 'w') as f:
        json.dump(mu, f, indent=2)
    with open(os.path.join(data_dir, MU_PARAMS_FILE), 'w') as f:
        json.dump(input_params, f, indent=2, sort_keys=True)
    return mu
