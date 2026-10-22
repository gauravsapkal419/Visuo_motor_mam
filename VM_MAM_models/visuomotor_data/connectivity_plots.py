"""
connectivity_plots
==================

Plots of the connectivity data of the visuomotor model:

- FLN            fraction of labelled neurons (FLN_completed, processed data)
- SLN            fraction of supragranular labelled neurons (SLN_completed)
- binary matrix  binary symmetric connectivity (bin_sym_conn_data)
- connection probability between populations (visuomotor_connection_prob_data.json)

In all matrices rows are target areas and columns are source areas.
Missing connections are white.

    from connectivity_plots import plot_connectivity_overview
    plot_connectivity_overview(M.params['data_dir'], M.params['data_files'],
                               area_list=M.area_list)
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

PROCESSED_FILE = 'visuomotor_processed_data.json'


def _load(path):
    with open(path, 'r') as f:
        return json.load(f)


def area_matrix(data, area_list):
    """
    Matrix [target, source] from data[target][source].
    Entries that are missing or NaN are NaN.
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


def population_matrix(conn_probs, area_list, population_list):
    """
    Matrix of connection probabilities, shape (n_areas*n_pops, n_areas*n_pops),
    ordered area 1: 23E ... 6I, area 2: 23E ... 6I, and so on.
    Rows: targets, columns: sources.
    """
    n = len(area_list) * len(population_list)
    mat = np.zeros((n, n))
    for ta, target_area in enumerate(area_list):
        for tp, target_pop in enumerate(population_list):
            for sa, source_area in enumerate(area_list):
                for sp, source_pop in enumerate(population_list):
                    mat[ta * len(population_list) + tp,
                        sa * len(population_list) + sp] = \
                        conn_probs[target_area][target_pop][source_area][source_pop]
    return mat


def _draw(ax, mat, area_list, title, cmap, norm=None, vmin=None, vmax=None,
          cbar_label=None):
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad('white')
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, norm=norm,
                   vmin=None if norm else vmin, vmax=None if norm else vmax,
                   interpolation='nearest', aspect='equal')
    n = len(area_list)
    ax.set_xticks(range(n))
    ax.set_xticklabels(area_list, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(area_list, fontsize=6)
    ax.set_xlabel('Source area')
    ax.set_ylabel('Target area')
    ax.set_title(title, fontsize=10)
    if cbar_label is not None:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=cbar_label)
    return im


def load_connectivity(data_dir, data_files, processed_file=PROCESSED_FILE):
    """Load the json files needed for the plots."""
    processed = _load(os.path.join(data_dir, processed_file))
    conn_probs = _load(os.path.join(data_dir, data_files['connection_probs']))
    return processed, conn_probs


def plot_connectivity_overview(data_dir, data_files, area_list,
                               population_list=None, save_path=None,
                               show=True):
    """
    Four panels: binary symmetric connectivity, FLN (log scale), SLN and the
    area-level connection probability (mean over all population pairs, log
    scale).
    """
    if population_list is None:
        population_list = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']
    processed, conn_probs = load_connectivity(data_dir, data_files)

    bin_mat = area_matrix(processed['bin_sym_conn_data'], area_list)
    fln_mat = area_matrix(processed['FLN_completed'], area_list)
    sln_mat = area_matrix(processed['SLN_completed'], area_list)

    pop_mat = population_matrix(conn_probs, area_list, population_list)
    k = len(population_list)
    n = len(area_list)
    prob_area = pop_mat.reshape(n, k, n, k).mean(axis=(1, 3))
    prob_area[prob_area <= 0] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    _draw(axes[0, 0], bin_mat, area_list, 'A  Binary symmetric connectivity',
          'binary', vmin=0, vmax=1)
    fln_pos = fln_mat[fln_mat > 0]
    _draw(axes[0, 1], np.where(fln_mat > 0, fln_mat, np.nan), area_list,
          'B  FLN', 'viridis',
          norm=mcolors.LogNorm(vmin=fln_pos.min(), vmax=fln_pos.max()),
          cbar_label='FLN')
    _draw(axes[1, 0], sln_mat, area_list, 'C  SLN', 'coolwarm', vmin=0, vmax=1,
          cbar_label='SLN')
    p_pos = prob_area[np.isfinite(prob_area)]
    _draw(axes[1, 1], prob_area, area_list,
          'D  Connection probability (mean over population pairs)', 'magma',
          norm=mcolors.LogNorm(vmin=p_pos.min(), vmax=p_pos.max()),
          cbar_label='p')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_connection_probability(data_dir, data_files, area_list,
                                population_list=None, areas=None,
                                save_path=None, show=True):
    """
    Population-level connection probability.

    areas=None : full matrix of all populations of all areas (log scale).
    areas=[...]: the given areas only, for example ['V1', 'F1'], so that
                 single 8 x 8 blocks are readable.
    """
    if population_list is None:
        population_list = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']
    conn_probs = _load(os.path.join(data_dir, data_files['connection_probs']))
    sel = list(areas) if areas is not None else list(area_list)
    mat = population_matrix(conn_probs, sel, population_list)
    mat = np.where(mat > 0, mat, np.nan)
    k = len(population_list)

    fig, ax = plt.subplots(figsize=(12, 11) if len(sel) > 6 else (8, 7))
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('white')
    pos = mat[np.isfinite(mat)]
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, aspect='equal',
                   norm=mcolors.LogNorm(vmin=pos.min(), vmax=pos.max()),
                   interpolation='nearest')
    for i in range(1, len(sel)):
        ax.axhline(i * k - 0.5, color='k', lw=0.4)
        ax.axvline(i * k - 0.5, color='k', lw=0.4)
    if len(sel) <= 6:
        labels = [f'{a} {p}' for a in sel for p in population_list]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
    else:
        centres = [i * k + k / 2 - 0.5 for i in range(len(sel))]
        ax.set_xticks(centres)
        ax.set_xticklabels(sel, rotation=90, fontsize=6)
        ax.set_yticks(centres)
        ax.set_yticklabels(sel, fontsize=6)
    ax.set_xlabel('Source population')
    ax.set_ylabel('Target population')
    ax.set_title('Connection probability')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='p')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
