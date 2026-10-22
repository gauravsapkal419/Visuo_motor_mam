"""
theory
======

Fixed points of the mean-field rate equations of the mesoscopic model.

All populations of all areas are stacked into one vector A (spikes/s) of
length n_areas * n_populations. The stationary rates satisfy

    A = hazard(mu + tau_m * W A),   hazard(u) = c * exp((u - V_T) / Delta_V)

with W the matrix of synaptic weights and mu the constant input. Starting
from random initial rates, fsolve searches for solutions; the distinct
solutions are the fixed points.

Note: W (pA) and, by default, mu (I_e in pA) are used with the neuron
parameters (mV, ms) as in the analysis notebook. Use mu_in_mV=True to pass the
mean input in mV instead.
"""
import numpy as np
from scipy.optimize import fsolve


def flat_index(area, pop, area_list, population_list):
    return area_list.index(area) * len(population_list) + population_list.index(pop)


def build_flat_weight_matrix(weights, area_list, population_list):
    """W_flat[target, source] from weights[t_area][t_pop][s_area][s_pop]."""
    n_total = len(area_list) * len(population_list)
    W_flat = np.zeros((n_total, n_total))
    for t_area in area_list:
        for t_pop in population_list:
            t_idx = flat_index(t_area, t_pop, area_list, population_list)
            for s_area in area_list:
                for s_pop in population_list:
                    s_idx = flat_index(s_area, s_pop, area_list, population_list)
                    W_flat[t_idx, s_idx] = weights[t_area][t_pop][s_area][s_pop]
    return W_flat


def build_flat_mu_vector(mean_input, area_list, population_list):
    """mu_flat from mean_input[area][pop]."""
    mu_flat = np.zeros(len(area_list) * len(population_list))
    for area in area_list:
        for pop in population_list:
            mu_flat[flat_index(area, pop, area_list, population_list)] = \
                mean_input[area][pop]
    return mu_flat


def hazard(u, c_hazard, DeltaV, vth):
    return c_hazard * np.exp((u - vth) / DeltaV)


def stationary_eqs(A, W_flat, mu_flat, taum, c_hazard, DeltaV, vth):
    u = mu_flat + taum * (W_flat @ A)
    return hazard(u, c_hazard, DeltaV, vth) - A


def find_fixed_points(W_flat, mu_flat, c_hazard=10., DeltaV=5., vth=15.,
                      taum=10., n_starts=300, A_max=60., seed=0, maxfev=2000,
                      decimals=3):
    """
    Search fixed points from n_starts random initial rates in [0, A_max].

    Returns
    -------
    unique_fps : array, shape (n_fixed_points, n_units)
        Distinct fixed points (rounded to `decimals`). Only converged
        solutions with non-negative rates are kept.
    """
    n = len(mu_flat)
    rng = np.random.default_rng(seed)
    fixed_points = []
    for _ in range(n_starts):
        A0 = rng.uniform(0, A_max, size=n)
        sol, info, ier, msg = fsolve(
            stationary_eqs, A0,
            args=(W_flat, mu_flat, taum, c_hazard, DeltaV, vth),
            full_output=True, maxfev=maxfev)
        if ier == 1 and np.all(sol >= -1e-3):
            fixed_points.append(np.round(sol, decimals))
    if not fixed_points:
        return np.empty((0, n))
    return np.unique(np.array(fixed_points), axis=0)
