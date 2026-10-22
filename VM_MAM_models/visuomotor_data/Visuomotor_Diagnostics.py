"""
VisuomotorDiagnostics
=====================

Plots, diagnostics and area-level degrees for the visuomotor model.
It uses the files written by Visuomotor_Model.py and
Visuomotor_Data_Processing.py:

- visuomotor_Data_Model.json
- visuomotor_connection_prob_data.json
- visuomotor_indegree_data.json
- visuomotor_processed_data.json

Output
------
visuomotor_area_level_degrees.json
    Area-level indegrees (per neuron of the target area, including the
    external input) and outdegrees (per neuron of the source area).
figures/
    External synapses per area, external indegrees, fraction of inputs
    per source area, SLN matrix and population-level connection
    probability matrix (png files).

Diagnostics (printed to the console)
------------------------------------
- populations with N_syn / (N_i * N_j) above a threshold
- minimum and maximum connection probability
- difference between the approximated and the full expression for the
  connection probability
"""
import json
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator
from matplotlib.patches import Patch, Rectangle

from nested_dict import nested_dict


BASEPATH = os.path.abspath(os.path.dirname(__file__))
FIGPATH = os.path.join(BASEPATH, 'figures')


def load_json(name):
    with open(os.path.join(BASEPATH, name), 'r') as f:
        return json.load(f)


def _finish(fig, filename, show):
    """Save the figure to figures/ and optionally show it."""
    os.makedirs(FIGPATH, exist_ok=True)
    fig.savefig(os.path.join(FIGPATH, filename), bbox_inches='tight', dpi=200)
    if show:
        plt.show()
    plt.close(fig)


"""
Area-level in- and outdegrees
-----------------------------
"""


def compute_outdegrees(synapses, neuron_numbers, area_list, population_list):
    """
    Outdegree of each connection: number of synapses divided by the size of
    the source population. Outdegrees from empty populations are 0.
    """
    outdegrees = nested_dict()
    for target_area, target_pop, source_area, source_pop in product(
            area_list, population_list, area_list, population_list):
        numS = neuron_numbers[source_area][source_pop]
        if numS > 0.0:
            outdegrees[target_area][target_pop][source_area][source_pop] = (
                synapses[target_area][target_pop][source_area][source_pop] / numS)
        else:
            outdegrees[target_area][target_pop][source_area][source_pop] = 0.0
    return outdegrees.to_dict()


def area_level_dict(dic, num, area_list, population_list, degree='indegree'):
    """
    Convert a connectivity dictionary from population-specific
    connectivity to area-level connectivity.

    Parameters
    ----------
    dic : dict
        Dictionary to transform, dic[target_area][target_pop][source_area]
        [source_pop].
    num : dict
        Population sizes, num[area][population] and num[area]['total'].
    area_list : list
        List of areas.
    population_list : list
        List of populations.
    degree : {'indegree', 'outdegree'}
        Whether dic contains in- or outdegrees. Defaults to 'indegree'.
    """
    area_level_dic = nested_dict()
    for target_area, source_area in product(area_list, area_list):
        conns = 0.0
        for target_pop, source_pop in product(population_list, repeat=2):
            if degree == 'indegree':
                conns += (dic[target_area][target_pop][source_area][source_pop]
                          * num[target_area][target_pop])
            elif degree == 'outdegree':
                conns += (dic[target_area][target_pop][source_area][source_pop]
                          * num[source_area][source_pop])
        if degree == 'indegree':
            area_level_dic[target_area][source_area] = (
                conns / num[target_area]['total'])
        elif degree == 'outdegree':
            area_level_dic[target_area][source_area] = (
                conns / num[source_area]['total'])

    # External input (indegrees only)
    if (degree == 'indegree' and
            'external' in dic[area_list[0]][population_list[0]]):
        for target_area in area_list:
            conns = 0.
            for target_pop in population_list:
                conns += (dic[target_area][target_pop]['external']['external']
                          * num[target_area][target_pop])
            area_level_dic[target_area]['external'] = (
                conns / num[target_area]['total'])

    return area_level_dic.to_dict()


def compute_area_level_degrees(model_data, indegrees, out_label=''):
    """
    Compute area-level in- and outdegrees and write them to
    visuomotor_area_level_degrees<out_label>.json.
    """
    syn = model_data['synapses']
    num = model_data['neuron_numbers']
    area_list = model_data['area_list']
    population_list = model_data['population_list']

    outdegrees = compute_outdegrees(syn, num, area_list, population_list)
    indegrees_areas = area_level_dict(indegrees, num, area_list,
                                      population_list, degree='indegree')
    outdegrees_areas = area_level_dict(outdegrees, num, area_list,
                                       population_list, degree='outdegree')

    with open(os.path.join(BASEPATH, ''.join(
            ('visuomotor_area_level_degrees', out_label, '.json'))), 'w') as f:
        json.dump({'indegrees_areas': indegrees_areas,
                   'outdegrees_areas': outdegrees_areas}, f, indent=3)

    return indegrees_areas, outdegrees_areas


"""
Diagnostics
-----------
"""


def connection_probabilities(synapses, num, area_list, population_list,
                             use_approximation=False):
    """
    Connection probability C = 1 - (1 - 1/(N_i*N_j))**N_syn, or the
    approximation N_syn / (N_i*N_j). Same definition as in
    Visuomotor_Model.py.
    """
    C = {}
    for ta, tp, sa, sp in product(area_list, population_list,
                                  area_list, population_list):
        N_i = num[ta].get(tp, 0)
        N_j = num[sa].get(sp, 0)
        N_syn = synapses[ta][tp][sa][sp]
        if N_i == 0 or N_j == 0 or N_syn == 0:
            val = 0.0
        elif use_approximation:
            val = round(N_syn / (N_i * N_j), 4)
        else:
            val = round(1.0 - np.exp(N_syn * np.log(1.0 - 1.0 / (N_i * N_j))), 4)
        C.setdefault(ta, {}).setdefault(tp, {}).setdefault(sa, {})[sp] = val
    return C


def check_synapse_ratio(model_data, threshold=0.3):
    """
    Find connections with N_syn / (N_i * N_j) above the threshold. Values
    close to or above 1 mean that more synapses than possible connections
    were assigned.

    Returns
    -------
    list of tuples (target_area, target_pop, source_area, source_pop,
    ratio, N_i, N_j, N_syn)
    """
    syn = model_data['synapses']
    num = model_data['neuron_numbers']
    hits = []
    for ta, tp, sa, sp in product(model_data['area_list'],
                                  model_data['population_list'],
                                  model_data['area_list'],
                                  model_data['population_list']):
        N_i = num[ta][tp]
        N_j = num[sa][sp]
        N_syn = syn[ta][tp][sa][sp]
        if N_i > 0 and N_j > 0:
            ratio = N_syn / (N_i * N_j)
            if ratio > threshold:
                hits.append((ta, tp, sa, sp, ratio, N_i, N_j, N_syn))
    print(f"{len(hits)} connections with N_syn / (N_i * N_j) > {threshold}")
    for ta, tp, sa, sp, ratio, N_i, N_j, N_syn in hits:
        print(f"{ta} {tp} <- {sa} {sp} | ratio = {ratio:.4f} | "
              f"N_i = {N_i:.1f} | N_j = {N_j:.1f} | N_syn = {N_syn:.1f}")
    return hits


def check_connection_probabilities(model_data, diff_threshold=1.0):
    """
    Print the range of the connection probabilities and compare the
    approximated with the full expression.

    Returns
    -------
    dict with min/max of both versions and the mean difference
    """
    syn = model_data['synapses']
    num = model_data['neuron_numbers']
    areas = model_data['area_list']
    pops = model_data['population_list']

    C_approx = connection_probabilities(syn, num, areas, pops, True)
    C_full = connection_probabilities(syn, num, areas, pops, False)

    v_approx, v_full, diffs = [], [], []
    for ta, tp, sa, sp in product(areas, pops, areas, pops):
        a = C_approx[ta][tp][sa][sp]
        b = C_full[ta][tp][sa][sp]
        v_approx.append(a)
        v_full.append(b)
        diffs.append(a - b)
        if a - b >= diff_threshold:
            print(f"Difference at [{ta}][{tp}] <- [{sa}][{sp}] : {a} != {b}")

    result = {'min_approx': min(v_approx), 'max_approx': max(v_approx),
              'min_full': min(v_full), 'max_full': max(v_full),
              'mean_difference': float(np.mean(diffs))}
    print(f"Approximation: min = {result['min_approx']}, "
          f"max = {result['max_approx']}")
    print(f"Full expression: min = {result['min_full']}, "
          f"max = {result['max_full']}")
    print(f"Mean difference (approximation - full): "
          f"{result['mean_difference']:.6f}")
    return result


"""
Plots
-----
"""


def external_synapses_per_area(model_data):
    """
    Total number of external synapses per area, obtained by summing the
    external synapses of all populations.
    """
    syn = model_data['synapses']
    ext = {}
    for area in model_data['area_list']:
        ext[area] = sum(syn[area][pop]['external']['external']
                        for pop in model_data['structure'][area])
    return ext


def plot_external_synapses(model_data, show=False):
    """Bar plot of the external synapses per area, sorted increasingly."""
    external = external_synapses_per_area(model_data)
    sorted_areas = sorted(external.keys(), key=lambda x: external[x])
    sorted_synapses = [external[a] for a in sorted_areas]

    fig, ax = plt.subplots(figsize=(14, 8))
    x_positions = np.arange(len(sorted_areas))
    ax.bar(x_positions, sorted_synapses, width=0.8, edgecolor='none',
           color='steelblue', alpha=0.7)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    ax.set_ylabel('External synapses (millions)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Areas (sorted by increasing synapses)', fontsize=12,
                  fontweight='bold')
    ax.set_title('External synapses per area', fontsize=14, fontweight='bold',
                 pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_areas, rotation=45, ha='right', fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f'{v / 1e6:.0f}'))
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Value labels for the first, the last and every 10th bar
    for i, synapses in enumerate(sorted_synapses):
        if i == 0 or i == len(sorted_areas) - 1 or i % 10 == 0:
            ax.text(i, synapses + max(sorted_synapses) * 0.01,
                    f'{synapses / 1e6:.0f}M', ha='center', va='bottom',
                    fontsize=7, rotation=90)

    stats_text = (f'Min: {sorted_areas[0]} ({sorted_synapses[0] / 1e6:.1f}M)\n'
                  f'Max: {sorted_areas[-1]} ({sorted_synapses[-1] / 1e6:.1f}M)\n'
                  f'Mean: {np.mean(sorted_synapses) / 1e6:.1f}M\n'
                  f'Median: {np.median(sorted_synapses) / 1e6:.1f}M')
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    _finish(fig, 'external_synapses_per_area.png', show)
    return sorted_areas, sorted_synapses


def plot_external_indegrees(indegrees, area_list, population_list,
                            areas=None, show=False):
    """
    Bar plot of the external indegree of each population.

    Parameters
    ----------
    areas : list, optional
        Areas to plot. Defaults to all areas in area_list.
    """
    if areas is None:
        areas = area_list

    values = []
    population_labels = []
    for area in areas:
        for pop in population_list:
            values.append(indegrees[area][pop]['external']['external'])
            population_labels.append(pop)

    n_areas = len(areas)
    n_pops = len(population_list)
    # one empty position between the areas
    x = np.array([i * (n_pops + 1) + j for i in range(n_areas)
                  for j in range(n_pops)])

    fig, ax = plt.subplots(figsize=(25, 8))
    ax.bar(x, values, width=0.6, color='skyblue')
    ax.set_xticks(x)
    ax.set_xticklabels(population_labels, rotation=90, fontsize=8)

    # Area names below the population labels
    ax2 = ax.twiny()
    area_ticks = [i * (n_pops + 1) + (n_pops - 1) / 2 for i in range(n_areas)]
    ax2.hlines(np.mean(values), x.min(), x.max(), linestyles='--', color='k')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(area_ticks)
    ax2.set_xticklabels(areas, rotation=45, ha='center', fontsize=10)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 40))

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylabel('Indegree')
    ax.set_title('External indegree by area and population')

    fig.tight_layout()
    _finish(fig, 'external_indegrees.png', show)


def plot_input_fractions(indegrees_areas, area_list, show=False):
    """
    Matrix of the fraction of the total inputs (including external input)
    that a target area receives from each source area.
    """
    n = len(area_list)
    conn_matrix = np.zeros((n, n))
    for i, area1 in enumerate(area_list[::-1]):  # target areas (rows)
        total = np.sum(list(indegrees_areas[area1].values()))
        for j, area2 in enumerate(area_list):  # source areas (columns)
            conn_matrix[i][j] = indegrees_areas[area1][area2] / total

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("none")
    ax.set_aspect(1.)

    masked_matrix = np.ma.masked_values(conn_matrix, 0.0)
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad('white', 1.0)

    x = np.arange(0, n + 1)
    y = np.arange(0, n + 1)
    X, Y = np.meshgrid(x, y)

    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels(area_list, rotation=90, size=10., ha='center')
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels(area_list[::-1], size=10., va='center')
    ax.set_ylabel('Target area', fontsize=16)
    ax.set_xlabel('Source area', fontsize=16)

    im = ax.pcolormesh(X, Y, masked_matrix, cmap=cmap, edgecolors='None',
                       norm=LogNorm(vmin=1e-6, vmax=1.))
    cbar = fig.colorbar(im, ticks=FixedLocator([1e-6, 1e-4, 1e-2, 1]),
                        fraction=0.046, pad=0.02, ax=ax)
    cbar.set_label('Fraction of total inputs', fontsize=10)
    ax.set_title('Fraction of inputs from source to target areas', fontsize=12,
                 pad=20)

    fig.tight_layout()
    _finish(fig, 'input_fractions.png', show)
    return conn_matrix


def plot_sln_matrix(processed_data, area_list, show=False):
    """Heatmap of the completed SLN values (rows: target, columns: source)."""
    SLN = processed_data['SLN_completed']
    n = len(area_list)
    matrix = np.full((n, n), np.nan)
    for i, target in enumerate(area_list):
        for j, source in enumerate(area_list):
            if target != source and source in SLN.get(target, {}):
                matrix[i, j] = SLN[target][source]

    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    img = ax.imshow(matrix, cmap='RdBu')
    # self-connections
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True,
                                   color='yellow'))
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(area_list, rotation=90, size=10)
    ax.set_yticklabels(area_list, size=10)
    ax.set_xlabel('Source')
    ax.set_ylabel('Target')
    ax.set_title('SLN completed data')
    fig.colorbar(img, ax=ax, label='SLN')

    _finish(fig, 'sln_completed.png', show)
    return matrix


def dict_to_matrix(d, area_list, structure):
    """
    Convert a dictionary d[target_area][target_pop][source_area][source_pop]
    into a matrix. The order of the rows and columns is given by area_list
    and by the populations in structure.
    """
    dim = sum(len(structure[area]) for area in area_list)
    M = np.zeros((dim, dim))
    i = 0
    for target_area in area_list:
        for target_pop in structure[target_area]:
            j = 0
            for source_area in area_list:
                for source_pop in structure[source_area]:
                    M[i][j] = d[target_area][target_pop][source_area][source_pop]
                    j += 1
            i += 1
    return M


def plot_connection_probability_matrix(conn_prob, structure, area_list,
                                       show=False):
    """
    Population-level matrix of the connection probabilities
    (rows: target populations, columns: source populations).
    """
    M = dict_to_matrix(conn_prob, area_list, structure)

    fig, ax = plt.subplots(figsize=(10, 9))
    masked_matrix = np.ma.masked_values(M, 0.0)
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad('w', 1.0)

    im = ax.pcolormesh(masked_matrix, cmap=cmap, edgecolors='None',
                       norm=LogNorm(vmin=1e-6, vmax=1.))
    ax.invert_yaxis()
    ax.set_aspect(1.)

    # area labels at the centre of each block of populations
    positions = []
    start = 0
    for area in area_list:
        positions.append(start + len(structure[area]) / 2.)
        start += len(structure[area])
    ax.set_xticks(positions)
    ax.set_xticklabels(area_list, rotation=90, size=10.)
    ax.set_yticks(positions)
    ax.set_yticklabels(area_list, size=10.)
    ax.set_xlabel('Source area')
    ax.set_ylabel('Target area')

    cbar = fig.colorbar(im, ticks=FixedLocator([1e-6, 1e-4, 1e-2, 1]),
                        fraction=0.046, ax=ax)
    cbar.set_label('Connection probability')

    fig.tight_layout()
    _finish(fig, 'connection_probability_matrix.png', show)
    return M

from matplotlib import colors as mcolors


def load_diagnostics_data(out_label=''):
    """
    Load the files that the diagnostics and connectivity plots need.

    Returns a dictionary with model_data, conn_prob, indegrees,
    processed_data, area_list, population_list and structure.
    """
    model_data = load_json(''.join(('visuomotor_Data_Model', out_label, '.json')))
    return {
        'model_data': model_data,
        'conn_prob': load_json(''.join(
            ('visuomotor_connection_prob_data', out_label, '.json'))),
        'indegrees': load_json(''.join(
            ('visuomotor_indegree_data', out_label, '.json'))),
        'processed_data': load_json('visuomotor_processed_data.json'),
        'area_list': model_data['area_list'],
        'population_list': model_data['population_list'],
        'structure': model_data['structure'],
    }


def area_matrix(data, area_list):
    """
    Matrix [target, source] from data[target][source]. Missing entries and
    NaN are NaN.
    """
    n = len(area_list)
    mat = np.full((n, n), np.nan)
    for i, target in enumerate(area_list):
        for j, source in enumerate(area_list):
            try:
                value = data[target][source]
            except (KeyError, TypeError):
                continue
            if value is not None and not np.isnan(value):
                mat[i, j] = value
    return mat


def area_level_connection_probability(conn_prob, area_list, population_list):
    """
    Mean connection probability over all population pairs of each pair of
    areas, shape (n_areas, n_areas), rows: targets. Zero becomes NaN.
    """
    n = len(area_list)
    mat = np.zeros((n, n))
    for i, ta in enumerate(area_list):
        for j, sa in enumerate(area_list):
            mat[i, j] = np.mean([conn_prob[ta][tp][sa][sp]
                                 for tp in population_list
                                 for sp in population_list])
    mat[mat <= 0] = np.nan
    return mat


def _draw_area_matrix(ax, mat, area_list, title, cmap, norm=None, vmin=None,
                      vmax=None, cbar_label=None, diagonal_color='green'):
    """Draw an area-by-area matrix. NaN entries are white, the diagonal
    (self-connections) is drawn in diagonal_color. Use diagonal_color=None
    to switch it off."""
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad('white')
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, norm=norm,
                   vmin=None if norm is not None else vmin,
                   vmax=None if norm is not None else vmax,
                   interpolation='nearest', aspect='equal')
    n = len(area_list)

    if diagonal_color is not None:
        for i in range(n):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor=diagonal_color,
                                   edgecolor='none', zorder=3))
        ax.legend(handles=[Patch(facecolor=diagonal_color, label='Self-connection')],
                  loc='upper right', bbox_to_anchor=(1.0, -0.18), fontsize=7,
                  frameon=False)

    ax.set_xticks(range(n))
    ax.set_xticklabels(area_list, rotation=90, fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(area_list, fontsize=10)
    ax.set_xlabel('Source area')
    ax.set_ylabel('Target area')
    ax.set_title(title, fontsize=10)
    if cbar_label is not None:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                           label=cbar_label)
    return im


def _log_norm(mat):
    positive = mat[np.isfinite(mat) & (mat > 0)]
    return mcolors.LogNorm(vmin=positive.min(), vmax=positive.max())


def plot_bin_sym_connectivity(processed_data, area_list, show=False):
    """Binary symmetric connectivity (rows: target, columns: source)."""
    mat = area_matrix(processed_data['bin_sym_conn_data'], area_list)
    fig, ax = plt.subplots(figsize=(9, 8))
    _draw_area_matrix(ax, mat, area_list, 'Binary symmetric connectivity',
                      'binary', vmin=0, vmax=1)
    fig.tight_layout()
    _finish(fig, 'binary_symmetric_connectivity.png', show)
    return mat


def plot_fln_matrix(processed_data, area_list, show=False):
    """FLN_completed (log scale, rows: target, columns: source)."""
    mat = area_matrix(processed_data['FLN_completed'], area_list)
    mat = np.where(mat > 0, mat, np.nan)
    fig, ax = plt.subplots(figsize=(9, 8))
    _draw_area_matrix(ax, mat, area_list, 'FLN completed data', 'inferno',
                      norm=_log_norm(mat), cbar_label='FLN')
    fig.tight_layout()
    _finish(fig, 'fln_completed.png', show)
    return mat


def plot_connectivity_overview(data, show=False):
    """
    Four panels: binary symmetric connectivity, FLN, SLN and the area-level
    connection probability. data is the output of load_diagnostics_data.
    """
    area_list = data['area_list']
    processed = data['processed_data']
    bin_mat = area_matrix(processed['bin_sym_conn_data'], area_list)
    fln_mat = area_matrix(processed['FLN_completed'], area_list)
    fln_mat = np.where(fln_mat > 0, fln_mat, np.nan)
    sln_mat = area_matrix(processed['SLN_completed'], area_list)
    prob_mat = area_level_connection_probability(
        data['conn_prob'], area_list, data['population_list'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    _draw_area_matrix(axes[0, 0], bin_mat, area_list,
                      'A  Binary symmetric connectivity Matrix', 'binary',
                      vmin=0, vmax=1)
    _draw_area_matrix(axes[0, 1], fln_mat, area_list, 'B  Completed FLN Matrix', 'inferno',
                      norm=_log_norm(fln_mat), cbar_label='FLN')
    _draw_area_matrix(axes[1, 0], sln_mat, area_list, 'C  Completed SLN Matrix', 'RdBu',
                      vmin=0, vmax=1, cbar_label='SLN')
    _draw_area_matrix(axes[1, 1], prob_mat, area_list,
                      'D  Connection probability (mean over population pairs)',
                      'magma', norm=_log_norm(prob_mat), cbar_label='p')
    fig.tight_layout()
    _finish(fig, 'connectivity_overview.png', show)
    return {'bin_sym': bin_mat, 'FLN': fln_mat, 'SLN': sln_mat,
            'connection_probability': prob_mat}



def run_diagnostics(out_label='', show=False, ratio_threshold=0.3):
    """
    Run diagnostics, compute area-level degrees and create all plots.

    Parameters
    ----------
    out_label : str
        Label of the input and output files.
    show : bool
        Whether to show the figures in addition to saving them.
    ratio_threshold : float
        Threshold for check_synapse_ratio.
    """
    model_data = load_json(''.join(('visuomotor_Data_Model', out_label, '.json')))
    conn_prob = load_json(''.join(
        ('visuomotor_connection_prob_data', out_label, '.json')))
    indegrees = load_json(''.join(
        ('visuomotor_indegree_data', out_label, '.json')))
    processed_data = load_json('visuomotor_processed_data.json')

    area_list = model_data['area_list']
    population_list = model_data['population_list']
    structure = model_data['structure']

    print("Synapse ratio check")
    check_synapse_ratio(model_data, threshold=ratio_threshold)
    print("\nConnection probability check")
    check_connection_probabilities(model_data)

    indegrees_areas, outdegrees_areas = compute_area_level_degrees(
        model_data, indegrees, out_label=out_label)

    sorted_areas, sorted_synapses = plot_external_synapses(model_data, show)
    print("\nAreas sorted by increasing external synapses")
    for i, (area, s) in enumerate(zip(sorted_areas, sorted_synapses), 1):
        print(f"{i:2d}. {area:6s}: {s / 1e6:8.2f} million synapses")

    plot_external_indegrees(indegrees, area_list, population_list, show=show)
    plot_input_fractions(indegrees_areas, area_list, show=show)
    plot_sln_matrix(processed_data, area_list, show=show)
    plot_connection_probability_matrix(conn_prob, structure, area_list,
                                       show=show)
    plot_bin_sym_connectivity(processed_data, area_list, show=show)
    plot_fln_matrix(processed_data, area_list, show=show)
    plot_connectivity_overview(load_diagnostics_data(out_label), show=show)    


if __name__ == '__main__':
    run_diagnostics()
