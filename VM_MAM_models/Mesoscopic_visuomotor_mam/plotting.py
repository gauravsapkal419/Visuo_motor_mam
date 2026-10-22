"""
plotting
========

Plots for the results of the mesoscopic visuomotor model. All functions
work on the results dictionary results[area][pop] (see simulation.py) or on
matrices derived from it.

configure(save_dir, show) sets where the figures are saved and whether they
are displayed. Figures are saved as png in save_dir if it is not None.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from scipy.signal import welch

_SETTINGS = {'save_dir': None, 'show': True}


def configure(save_dir=None, show=True):
    """Set the folder for saved figures (None: do not save) and display mode."""
    _SETTINGS['save_dir'] = save_dir
    _SETTINGS['show'] = show
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)


def _output(name):
    """Save the current figure and show or close it."""
    fig = plt.gcf()
    if _SETTINGS['save_dir'] is not None and name:
        fig.savefig(os.path.join(_SETTINGS['save_dir'], name + '.png'),
                    dpi=200, bbox_inches='tight')
    if _SETTINGS['show']:
        plt.show()
    else:
        plt.close(fig)


def plot_mean_rate_heatmap(matrix, area_list, population_list, title=None,
                           annotate=True):
    """Heatmap of the mean rates (populations x areas) on a log scale."""
    import seaborn as sns

    matrix_no_zeros = np.where(matrix <= 0, 1e-5, matrix)
    plt.figure(figsize=(22, 6))
    norm = mcolors.LogNorm(vmin=matrix_no_zeros.min(), vmax=matrix_no_zeros.max())
    ax = sns.heatmap(matrix_no_zeros, cmap=plt.cm.RdYlBu, norm=norm,
                     annot=annotate, fmt=".2f", xticklabels=area_list,
                     yticklabels=population_list, linewidths=0.5,
                     linecolor='black')
    plt.yticks(rotation=0)
    plt.xlabel("\nBrain Areas", fontsize=15)
    plt.ylabel("Populations", fontsize=15)
    plt.title(title or "Mean firing rates (Hz)", fontsize=15)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Rate (Hz)")
    cbar.set_ticks([1e-2, 1e-1, 1, 10, 100])
    cbar.ax.set_yticklabels(['0.01', '0.1', '1', '10', '100'])
    plt.tight_layout()
    _output('mean_rate_heatmap')


def plot_population_rates(rate_matrix, population_list, plot_type='box',
                          log_scale=False, values=False):
    """
    Mean firing rate of each population across areas.

    Parameters
    ----------
    rate_matrix : array, populations x areas
    plot_type : {'box', 'bar'}
        'box': distribution across areas, diamond = mean.
        'bar': mean and standard deviation across areas.
    values : bool
        Print mean and standard deviation of every population.
    """
    n = len(population_list)
    means = np.array([np.mean(rate_matrix[i]) for i in range(n)])
    stds = np.array([np.std(rate_matrix[i]) for i in range(n)])

    if values:
        print(f"{'Population':<12}{'Mean (sp/s)':<15}{'SD (sp/s)':<15}")
        print("-" * 42)
        for i in range(n):
            print(f"{population_list[i]:<12}{means[i]:<15.4f}{stds[i]:<15.4f}")

    fig, ax = plt.subplots(figsize=(12, 7))
    index = np.arange(n)
    is_exc = np.array(['E' in pop for pop in population_list])
    colors = np.where(is_exc, 'steelblue', 'lightcoral')

    if plot_type == 'box':
        bp = ax.boxplot([rate_matrix[i] for i in range(n)], positions=index,
                        widths=0.5, patch_artist=True, showmeans=True,
                        meanprops={'marker': 'D', 'markerfacecolor': 'Yellow',
                                   'markeredgecolor': 'black', 'markersize': 6},
                        medianprops={'color': 'black', 'linewidth': 1.2},
                        flierprops={'marker': 'x', 'markersize': 5,
                                    'markerfacecolor': 'gray', 'alpha': 0.6})
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        handles = [Patch(color='steelblue', label='Excitatory'),
                   Patch(color='lightcoral', label='Inhibitory'),
                   plt.Line2D([0], [0], marker='D', color='w',
                              markerfacecolor='Yellow', markeredgecolor='black',
                              markersize=6, label='Mean')]
        ax.legend(handles=handles, frameon=True, fontsize=11, loc='upper left')
    else:
        bars = ax.bar(index, means, 0.35, color=colors, yerr=stds, capsize=4,
                      error_kw={'elinewidth': 1, 'alpha': 0.7})
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.legend(handles=[Patch(color='steelblue', label='Excitatory'),
                           Patch(color='lightcoral', label='Inhibitory')],
                  frameon=False)

    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Firing rate (spikes/s, log scale)', fontsize=14)
    else:
        ax.set_ylabel('Firing rate (spikes/s)', fontsize=14)
    ax.set_xlabel('Population', fontsize=14, labelpad=20)
    ax.set_xticks(index)
    ax.set_xticklabels(population_list)
    ax.grid(alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    plt.tight_layout()
    _output('population_rates_' + plot_type)


def plot_rate_dynamics(results, areas, pops, key='rate_from_spikes',
                       title='Rate dynamics', figure_size=(18, 6)):
    """Rate traces of the chosen populations (key: rate_from_spikes or rate_from_mean)."""
    if isinstance(areas, str):
        areas = [areas]
    if isinstance(pops, str):
        pops = [pops]
    time_key = 'time_bins' if key == 'rate_from_spikes' else 'times_mm'

    plt.figure(figsize=figure_size)
    for area in areas:
        for pop in pops:
            if area in results and pop in results[area]:
                r = results[area][pop]
                plt.plot(r[time_key], r[key], linewidth=0.7, label=f'{area}:{pop}')
            else:
                print(f"Warning: {area}:{pop} not found in data.")
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing rate (spikes/s)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    _output('rate_dynamics')


def plot_phase_space(results, area, pop_x, pop_y, key='rate_from_mean'):
    """Rate of pop_x against rate of pop_y, with the time encoded in the alpha value."""
    r1 = np.asarray(results[area][pop_x][key])
    r2 = np.asarray(results[area][pop_y][key])
    t = np.arange(len(r1))
    timecolors = np.array([(1, 0, 0, i) for i in t / max(t[-1], 1)])
    plt.figure(figsize=(6, 6))
    plt.scatter(r1, r2, color=timecolors, s=6)
    plt.xlabel(f'{area} {pop_x} (spikes/s)')
    plt.ylabel(f'{area} {pop_y} (spikes/s)')
    plt.tight_layout()
    _output('phase_space')


def plot_beta_power_map(power_matrix, area_list, population_list, title=None):
    """Map of the beta power (populations x areas) on a log scale."""
    plt.figure(figsize=(14, 6))
    plt.imshow(power_matrix, aspect='auto', origin='upper',
               norm=mcolors.LogNorm(vmin=power_matrix[power_matrix > 0].min(),
                                    vmax=power_matrix.max()),
               cmap='cividis')
    plt.colorbar(label=r'Beta power $(spikes/s)^2$')
    plt.xticks(range(len(area_list)), area_list, rotation=90)
    plt.yticks(range(len(population_list)), population_list)
    plt.xlabel("Brain areas")
    plt.ylabel("Populations")
    plt.title(title or "Spatial map of beta power")
    plt.tight_layout()
    _output('beta_power_map')


def plot_area_beta_power(area_power, area_list, band=(13, 30)):
    """Beta power of every area, sorted in decreasing order."""
    n_areas = len(area_list)
    order = np.argsort(area_power)[::-1]
    sorted_areas = [area_list[i] for i in order]
    sorted_power = area_power[order]

    fig, ax = plt.subplots(figsize=(13, 5))
    span = sorted_power.max() - sorted_power.min()
    colors_bar = plt.cm.inferno((sorted_power - sorted_power.min())
                                / (span if span > 0 else 1.))
    ax.bar(range(n_areas), sorted_power, color=colors_bar)
    ax.set_xticks(range(n_areas))
    ax.set_xticklabels(sorted_areas, rotation=90, fontsize=9)
    ax.set_ylabel(f'Beta band power ({band[0]}-{band[1]} Hz)', fontsize=12)
    ax.set_xlabel('Area', fontsize=12)
    ax.set_title('Area-level beta power across areas', fontsize=13)
    ax.set_yscale('log')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    _output('area_beta_power')


def plot_activity_heatmap(activity, area_list, population_list):
    """Heatmap of the rate traces of all populations (rows) over time."""
    import seaborn as sns

    n_pops = len(population_list)
    n_areas = len(area_list)
    plt.figure(figsize=(14, 12))
    sns.heatmap(activity, cmap="coolwarm", xticklabels=False,
                yticklabels=False, cbar=True)
    for i in range(1, n_areas):
        plt.axhline(i * n_pops, color='black', linewidth=1)
    y_ticks = [i * n_pops + n_pops / 2 for i in range(n_areas)]
    plt.yticks(y_ticks, area_list, rotation=0, fontsize=9)
    plt.xlabel("Time bins")
    plt.ylabel("Areas")
    plt.title(f"Firing rate dynamics ({n_areas} areas x {n_pops} populations)")
    plt.tight_layout()
    _output('activity_heatmap')


# The following three functions are taken unchanged from the notebook
# (only the final plt.show() is replaced by _output).


def plot_population_activity(
    results,
    areas,
    pops,
    transient_ms=0.0,
    show_spikes=True,
    show_mean=True,
    smooth=False,
    smoothing_window=5,
    figure_size=(18,6),
    title="Population Activity",
    show_individual=True,      # NEW: toggle individual pop traces on/off
    show_area_average=True,    # NEW: toggle the population-averaged trace
    show_std_band=True         # NEW: shaded band = std across populations
):
    if isinstance(areas, str):
        areas = [areas]
    if isinstance(pops, str):
        pops = [pops]

    plt.figure(figsize=figure_size)

    for area in areas:

        # --------------------------------------------
        # Collect all population traces first (needed for averaging)
        # --------------------------------------------
        spk_traces, mean_traces = [], []
        t_spk_ref, t_mm_ref = None, None

        for pop in pops:
            data = results[area][pop]

            if show_spikes:
                t_spk = data["time_bins"]
                A_spk = data["rate_from_spikes"]
                mask = t_spk >= transient_ms
                t_spk_masked = t_spk[mask]
                A_spk_masked = A_spk[mask]

                if smooth:
                    kernel = np.ones(smoothing_window) / smoothing_window
                    A_spk_masked = np.convolve(A_spk_masked, kernel, mode='same')

                spk_traces.append(A_spk_masked)
                t_spk_ref = t_spk_masked

                if show_individual:
                    plt.plot(t_spk_masked, A_spk_masked, linewidth=0.7, alpha=0.35,
                              color='darkblue', label=f"{area}:{pop}  $A_N(t)$" if len(pops)==1 else None)

            if show_mean:
                t_mm = data["times_mm"]
                A_mean = data["rate_from_mean"]
                mask = t_mm >= transient_ms
                t_mm_masked = t_mm[mask]
                A_mean_masked = A_mean[mask]

                if smooth:
                    kernel = np.ones(smoothing_window) / smoothing_window
                    A_mean_masked = np.convolve(A_mean_masked, kernel, mode='same')

                mean_traces.append(A_mean_masked)
                t_mm_ref = t_mm_masked

                if show_individual:
                    plt.plot(t_mm_masked, A_mean_masked, linewidth=0.3, alpha=0.3,
                              color='skyblue', label=f"{area}:{pop}  $\\bar A(t)$" if len(pops)==1 else None)

        # --------------------------------------------
        # Area-averaged activity: mean across populations at each timepoint
        # --------------------------------------------
        if show_area_average and show_spikes and spk_traces:
            spk_stack = np.array(spk_traces)              # shape: (n_pops, n_timepoints)
            avg_spk = spk_stack.mean(axis=0)
            std_spk = spk_stack.std(axis=0)

            plt.plot(t_spk_ref, avg_spk, linewidth=2.2, color='navy',
                      label=f"{area} population-averaged $A_N(t)$")
            if show_std_band:
                plt.fill_between(t_spk_ref, avg_spk - std_spk, avg_spk + std_spk,
                                  color='navy', alpha=0.15)

        if show_area_average and show_mean and mean_traces:
            mean_stack = np.array(mean_traces)
            avg_mean = mean_stack.mean(axis=0)
            std_mean = mean_stack.std(axis=0)

            plt.plot(t_mm_ref, avg_mean, linewidth=2.2, color='deepskyblue',
                      label=f"{area} population-averaged $\\bar A(t)$")
            if show_std_band:
                plt.fill_between(t_mm_ref, avg_mean - std_mean, avg_mean + std_mean,
                                  color='deepskyblue', alpha=0.15)

    plt.xlabel("Time (ms)")
    plt.ylabel("Average activity (spikes/s)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _output('population_activity')


def plot_burstiness_relationships(df, area_col='area', pop_col='population',
                                    highlight_thresh_B=0.3, highlight_thresh_rate=None,
                                    annotate_top_n=10, figsize=(10, 6), save_path=None):
    """
    Plot burstiness (B) vs mean_rate and B vs duty_cycle side by side.

    Parameters
    ----------
    df : pd.DataFrame
        Output of analyze_all_nodes(), must contain columns:
        'burstiness_B', 'mean_rate', 'duty_cycle', area_col, pop_col
    highlight_thresh_B : float
        B values above this are highlighted as candidate "driver" nodes
    highlight_thresh_rate : float or None
        Optional mean_rate threshold for a second highlight criterion
    annotate_top_n : int
        Number of most-bursty nodes to label with area_pop names
    figsize : tuple
        Figure size
    save_path : str or None
        If provided, saves the figure to this path (e.g. 'burst_scatter.png')
    """
    fig, axes = plt.subplots(1, 1, figsize=figsize)

    # Color by population type if E/I is encoded in population names
    is_excitatory = df[pop_col].astype(str).str.contains('E', case=False)
    colors = np.where(is_excitatory, '#d62728', '#1f77b4')  # red=E, blue=I

    driver_mask = df['burstiness_B'] >= highlight_thresh_B
    if highlight_thresh_rate is not None:
        driver_mask &= df['mean_rate'] >= highlight_thresh_rate

    # --- Panel 1: B vs mean_rate ---
    ax = axes
    ax.scatter(df.loc[~driver_mask, 'mean_rate'], df.loc[~driver_mask, 'burstiness_B'],
               c=colors[~driver_mask.values], alpha=0.5, s=35, edgecolors='none', label='_nolegend_')
    ax.scatter(df.loc[driver_mask, 'mean_rate'], df.loc[driver_mask, 'burstiness_B'],
               c=colors[driver_mask.values], alpha=0.9, s=70, edgecolors='black', linewidths=0.8,
               label='Candidate drivers')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axhline(highlight_thresh_B, color='red', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Mean rate (spikes/s)')
    ax.set_ylabel('Burstiness coefficient (B)')
    ax.set_title('Burstiness vs. Mean Firing Rate')
    ax.set_xscale('log')



    # Annotate the top-N most bursty nodes on both panels
    top_nodes = df.nlargest(annotate_top_n, 'burstiness_B')
    for _, row in top_nodes.iterrows():
        label = f"{row[area_col]}_{row[pop_col]}"
        axes.annotate(label, (row['mean_rate'], row['burstiness_B']),
                          fontsize=7, xytext=(4, 4), textcoords='offset points')

    # Legend for E/I coloring
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=8, label='Excitatory'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=8, label='Inhibitory'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black',
               markersize=9, label=f'B ≥ {highlight_thresh_B} (candidate driver)')
    ]
    fig.legend(handles=legend_elems, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05), frameon=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    _output('burstiness_relationships')
    return fig, axes


def plot_power_spectrum(results, area, pops, fs, colors=None, nperseg=300,
                         beta_band=(13, 30), xlim=None, title=None,
                         rho_0=None, Delta_V=None, tau_syn_ex_ms=None,
                         tau_syn_in_ms=None, seed_val=None, log_x = False, log_y=False):
    """
    Plot Welch PSD for one area across multiple populations.

    Parameters
    ----------
    results : dict
        Simulation results dict, results[area][pop]['rate_from_spikes'].
    area : str
        Area name to plot.
    pops : list of str
        Populations to plot for this area, e.g. ['23E','23I','4E','4I'].
    fs : float
        Sampling frequency in Hz (1000/record_interval_ms).
    colors : dict, optional
        Mapping {pop_name: color}. If None, default matplotlib cycle is used.
    nperseg : int
        Segment length for Welch's method.
    beta_band : tuple
        (fmin, fmax) to shade as beta band.
    xlim : tuple, optional
        x-axis limits.
    title : str, optional
        Custom title; if None, builds one from provided parameters.
    log_y : bool
        Whether to use log scale on y-axis.
    """
    plt.figure(figsize=(9, 5))

    default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, pop in enumerate(pops):
        r = results[area][pop]['rate_from_spikes'][200:]
        f, Pxx = welch(r, fs=fs, nperseg=nperseg)

        color = colors[pop] if (colors is not None and pop in colors) else default_cycle[i % len(default_cycle)]
        plt.plot(f, Pxx, label=pop, color=color, linewidth=1.5)

    plt.axvspan(beta_band[0], beta_band[1], color='gray', alpha=0.2, label='Beta band')

    if xlim is not None:
        plt.xlim(xlim)
    if log_y:
        plt.yscale('log')
    if log_x:
        plt.xscale('log')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'PSD $((\mathrm{spikes/s})^2/\mathrm{Hz})$')

    if title is None:
        title = f'Power Spectral Density — {area}'
        params_str = []
        if rho_0 is not None: params_str.append(f'$\\rho_0$={rho_0}')
        if Delta_V is not None: params_str.append(f'$\\Delta_V$={Delta_V}')
        if tau_syn_ex_ms is not None: params_str.append(f'$\\tau_e$={tau_syn_ex_ms}')
        if tau_syn_in_ms is not None: params_str.append(f'$\\tau_i$={tau_syn_in_ms}')
        if seed_val is not None: params_str.append(f'RNG={seed_val}')
        if params_str:
            title += ', ' + ', '.join(params_str)

    plt.title(title, size=10)
    plt.legend()
    plt.tight_layout()
    #plt.savefig('output/power_spectrum_demo.png', dpi=200)
    #plt.close()
    _output('power_spectrum')
