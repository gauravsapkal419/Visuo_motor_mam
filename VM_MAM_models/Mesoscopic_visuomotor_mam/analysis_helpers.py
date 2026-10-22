"""
analysis_helpers
================

Functions for the analysis of the population rates of the mesoscopic model.
They work on the results dictionary results[area][pop] (see simulation.py)
and do not need NEST.

1. Burst detection and bistability tests
2. Band power and power spectra
3. Activity matrices
4. Log of experiments for several values of g and seeds
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch


# ============================================================
# 1. Burst detection and bistability tests
# ============================================================

# ============================================================
# 1. Burst detection (threshold-based, applied to rate traces)
# ============================================================
def detect_bursts(x, thresh_percentile=75, min_dur=1):
    """Detect burst episodes where rate exceeds a percentile threshold."""
    thresh = np.percentile(x, thresh_percentile)
    above = x > thresh
    bursts, start = [], None
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start >= min_dur:
                bursts.append((start, i))
            start = None
    if start is not None:
        bursts.append((start, len(above)))
    durations = [e - s for s, e in bursts]
    inter_burst = [bursts[i+1][0] - bursts[i][1] for i in range(len(bursts) - 1)]
    duty_cycle = above.sum() / len(x)
    return {'threshold': thresh, 'n_bursts': len(bursts), 'burst_durations': durations,
            'inter_burst_intervals': inter_burst, 'duty_cycle': duty_cycle, 'bursts': bursts}


# ============================================================
# 2. Burstiness coefficient B (Goh & Barabasi normalized index)
#    B -> 1: bursty, B ~ 0: Poisson-like, B -> -1: periodic
# ============================================================
def burstiness_coefficient(intervals):
    intervals = np.asarray(intervals, dtype=float)
    if len(intervals) < 2:
        return np.nan
    mu, sigma = intervals.mean(), intervals.std()
    return (sigma - mu) / (sigma + mu)


# ============================================================
# 3. Sliding-window mean/variance (non-stationarity check)
# ============================================================
def sliding_stats(x, window=20, step=5):
    means, stds, centers = [], [], []
    for start in range(0, len(x) - window, step):
        seg = x[start:start + window]
        means.append(seg.mean())
        stds.append(seg.std())
        centers.append(start + window // 2)
    return np.array(centers), np.array(means), np.array(stds)


# ============================================================
# 4. Autocorrelation function + power spectral density
# ============================================================
def acf(x, max_lag=50):
    x = x - x.mean()
    n = len(x)
    result = np.correlate(x, x, mode='full')[n - 1:]
    result /= result[0]
    return result[:max_lag]

def power_spectrum(x, dt=1.0):
    x = x - x.mean()
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=dt)
    psd = np.abs(np.fft.rfft(x)) ** 2 / n
    return freqs, psd


# ============================================================
# 5. Hidden Markov Model state fitting (optional: needs hmmlearn)
#    pip install hmmlearn
# ============================================================
def fit_hmm(x, n_states=2, n_iter=200, random_state=0):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return None, None
    X = x.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type='diag',
                         n_iter=n_iter, random_state=random_state)
    model.fit(X)
    states = model.predict(X)
    means = model.means_.flatten()
    return model, {'states': states, 'means': means, 'logL': model.score(X)}


# ============================================================
# 6. Hartigan's dip test for bimodality (optional: needs diptest)
#    pip install diptest
# ============================================================
def hartigan_dip_test(data):
    try:
        import diptest
    except ImportError:
        return None
    dip, pval = diptest.diptest(np.asarray(data))
    return {'dip_statistic': dip, 'p_value': pval}


# ============================================================
# 7. Master function: run full battery on one node's trace
# ============================================================
def analyze_node(x, label='', thresh_percentile=75, hmm_states=2):
    x = np.asarray(x, dtype=float)
    bstats = detect_bursts(x, thresh_percentile=thresh_percentile)
    B = burstiness_coefficient(bstats['inter_burst_intervals'])
    centers, means, stds = sliding_stats(x)
    acf_vals = acf(x, max_lag=min(50, len(x)//2))
    freqs, psd = power_spectrum(x)
    hmm_model, hmm_res = fit_hmm(x, n_states=hmm_states)

    summary = {
        'label': label,
        'mean_rate': x.mean(),
        'std_rate': x.std(),
        'max_rate': x.max(),
        'n_bursts': bstats['n_bursts'],
        'duty_cycle': bstats['duty_cycle'],
        'burstiness_B': B,
        'mean_burst_duration': np.mean(bstats['burst_durations']) if bstats['burst_durations'] else np.nan,
        'sliding_var_range': stds.max() - stds.min() if len(stds) else np.nan,
        'hmm_state_means': hmm_res['means'].tolist() if hmm_res else None,
        'hmm_logL': hmm_res['logL'] if hmm_res else None,
    }
    detail = {'burst_info': bstats, 'sliding': (centers, means, stds),
              'acf': acf_vals, 'psd': (freqs, psd), 'hmm': hmm_res}
    return summary, detail


# ============================================================
# 8. Batch runner across the full 344-node dictionary
# ============================================================
def analyze_all_nodes(results, area_list, population_list, transient_cut=50,
                       thresh_percentile=75, hmm_states=2):
    """
    results: dict[area][pop]['rate_from_spikes'] -> np.array
    transient_cut: number of initial ms/bins to discard as transient
    Returns a tidy DataFrame, one row per (area, population).
    """
    rows = []
    for area in area_list:
        for pop in population_list:
            try:
                trace = results[area][pop]['rate_from_spikes']
            except KeyError:
                continue
            trace = np.asarray(trace, dtype=float)[transient_cut:]
            if len(trace) < 20:
                continue
            summary, _ = analyze_node(trace, label=f'{area}_{pop}',
                                       thresh_percentile=thresh_percentile,
                                       hmm_states=hmm_states)
            summary['area'] = area
            summary['population'] = pop
            rows.append(summary)
    return pd.DataFrame(rows)


# ============================================================
# 9. Bistability test across seeds (network-mean rate per seed)
# ============================================================
def bistability_dip_test(seed_mean_rates):
    """seed_mean_rates: array-like of one scalar mean rate per seed."""
    data = np.log10(np.asarray(seed_mean_rates) + 1e-3)  # log-scale separates decades
    return hartigan_dip_test(data)

# ============================================================
# 2. Band power and power spectra
# ============================================================
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def band_power(freqs, psd, fmin, fmax):
    """Integral of the PSD between fmin and fmax."""
    idx = (freqs >= fmin) & (freqs <= fmax)
    return _trapezoid(psd[idx], freqs[idx])


def beta_power_matrix(results, area_list, population_list, fs, nperseg=200,
                      band=(13, 30)):
    """
    Band power of the rate_from_spikes trace of every population.
    Rows: populations, columns: areas.
    """
    power = np.zeros((len(population_list), len(area_list)))
    for i, area in enumerate(area_list):
        for j, pop in enumerate(population_list):
            r = results[area][pop]['rate_from_spikes']
            f, Pxx = welch(r, fs=fs, nperseg=nperseg)
            power[j, i] = band_power(f, Pxx, band[0], band[1])
    return power


def area_band_power(results, area_list, population_list, fs, nperseg=300,
                    band=(13, 30)):
    """Band power of the summed rate of all populations of each area."""
    power = np.zeros(len(area_list))
    for i, area in enumerate(area_list):
        combined_rate = np.sum(
            [results[area][pop]['rate_from_spikes'] for pop in population_list],
            axis=0)
        f, Pxx = welch(combined_rate, fs=fs, nperseg=nperseg)
        power[i] = band_power(f, Pxx, band[0], band[1])
    return power


# ============================================================
# 3. Activity matrices
# ============================================================
def activity_matrix(results, area_list, population_list, key='rate_from_spikes'):
    """
    Matrix (units x time bins) of the rates. The rows are ordered as
    area 1: 23E ... 6I, area 2: 23E ... 6I, and so on.
    """
    return np.array([results[area][pop][key]
                     for area in area_list for pop in population_list])


def population_statistics(rate_matrix):
    """
    Mean, standard deviation and median over the areas for each population.
    rate_matrix has populations as rows and areas as columns.
    """
    return {'mean': np.array([np.mean(r) for r in rate_matrix]),
            'std': np.array([np.std(r) for r in rate_matrix]),
            'median': np.array([np.median(r) for r in rate_matrix])}


# ============================================================
# 4. Log of experiments for several g and seeds
# ============================================================


def load_experiment_log(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    return data


def save_experiment_log(data, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def log_experiment(filepath, g_value, rng_seed, mean_network_activity,
                    activity_per_pop, std_per_pop, median_per_pop,
                    pop_list, overwrite=False):
    """
    Log a single (g, rng_seed) experiment result into the JSON file.
    Per-population values are stored as {pop_name: value} dictionaries.

    Parameters
    ----------
    filepath : str
    g_value : float
    rng_seed : int
    mean_network_activity : float
    activity_per_pop, std_per_pop, median_per_pop : array-like (length = n_pops)
        Values ordered to match pop_list.
    pop_list : list of str
        Population names, e.g. ['23E','23I','4E','4I','5E','5I','6E','6I'].
        Required now, since it defines the keys of the per-pop dicts.
    overwrite : bool
        If False (default), existing (g, seed) entries are not overwritten.
    """
    data = load_experiment_log(filepath)

    g_key = str(g_value)
    seed_key = str(rng_seed)

    if g_key not in data:
        data[g_key] = {}

    if seed_key in data[g_key] and not overwrite:
        print(f"[SKIP] Entry for g={g_value}, seed={rng_seed} already exists. "
              f"Use overwrite=True to replace it.")
        return data

    activity_arr = np.asarray(activity_per_pop, dtype=float)
    std_arr = np.asarray(std_per_pop, dtype=float)
    median_arr = np.asarray(median_per_pop, dtype=float)

    entry = {
        "mean_network_activity": float(mean_network_activity),
        "Mean_activity_per_pop": {pop: float(val) for pop, val in zip(pop_list, activity_arr)},
        "std_per_pop": {pop: float(val) for pop, val in zip(pop_list, std_arr)},
        "median_per_pop": {pop: float(val) for pop, val in zip(pop_list, median_arr)},
    }

    data[g_key][seed_key] = entry

    save_experiment_log(data, filepath)
    print(f"[SAVED] g={g_value}, seed={rng_seed}")
    return data


def list_logged_seeds(filepath, g_value):
    data = load_experiment_log(filepath)
    g_key = str(g_value)
    return sorted(data.get(g_key, {}).keys(), key=int)

