"""
mesoscopic_helpers
==================

Small helper functions shared by the model, simulation and analysis:
loading json files, hashing parameter dictionaries into labels, saving and
loading simulation results and converting results into matrices.
"""
import hashlib
import json
import os

import numpy as np


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path, indent=2):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def dict_hash(d):
    """Short hash of a dictionary, used as label of models and simulations."""
    text = json.dumps(d, sort_keys=True, default=_json_default)
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def save_results(results, path):
    """Save results[area][pop][field] (numbers and arrays) to one npz file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    flat = {}
    for area in results:
        for pop in results[area]:
            for field, value in results[area][pop].items():
                flat['::'.join((area, pop, field))] = np.asarray(value)
    np.savez_compressed(path, **flat)


def load_results(path):
    """Inverse of save_results."""
    results = {}
    with np.load(path) as data:
        for key in data.files:
            area, pop, field = key.split('::')
            value = data[key]
            if value.ndim == 0:
                value = value.item()
            results.setdefault(area, {}).setdefault(pop, {})[field] = value
    return results


def results_summary(results, fields=('mean_rate_spikes_Hz', 'mean_rate_mean_Hz')):
    """Scalar entries of the results (mean rates), for json output."""
    return {area: {pop: {f: float(results[area][pop][f]) for f in fields}
                   for pop in results[area]} for area in results}


def mean_rate_matrix(results, area_list, population_list,
                     key='mean_rate_spikes_Hz'):
    """
    Matrix of mean rates with populations as rows and areas as columns.
    Missing entries are 0.
    """
    matrix = np.zeros((len(population_list), len(area_list)))
    for i, pop in enumerate(population_list):
        for j, area in enumerate(area_list):
            try:
                matrix[i, j] = results[area][pop][key]
            except KeyError:
                matrix[i, j] = 0.0
    return matrix
