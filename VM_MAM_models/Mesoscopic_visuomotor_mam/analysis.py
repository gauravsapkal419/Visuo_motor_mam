"""
analysis
========

Analysis class of the mesoscopic visuomotor model. It loads the results of a
simulation (or of a previous run identified by its label) and provides the
analyses and plots:

- mean rates of all populations (heatmap, statistics across areas)
- rate traces and population activity
- burstiness and bimodality of the rate traces
- power spectra and beta power
- fixed points of the mean-field equations

Figures are saved in data_path/<label>/figures/.
"""
import os
from copy import deepcopy

import numpy as np

import analysis_helpers as ah
import plotting
import theory
from config import data_path
from default_params import analysis_params, check_custom_params, nested_update
from mesoscopic_helpers import load_results, mean_rate_matrix


class Analysis:
    """
    Parameters
    ----------
    network : MesoscopicModel, optional
        Provides area and population lists, parameters and weights.
    simulation : Simulation, optional
        Results are taken from the simulation or loaded from its folder.
    label : str, optional
        Label of a previous simulation (folder in data_path). Use instead of
        `simulation`.
    ana_spec : dict, optional
        Custom analysis parameters, merged into default_params.analysis_params.
    show : bool
        Display the figures (they are always saved).
    """

    def __init__(self, network=None, simulation=None, label=None,
                 ana_spec=None, show=True):
        self.network = network
        self.simulation = simulation
        self.params = deepcopy(analysis_params)
        if ana_spec is not None:
            check_custom_params(ana_spec, analysis_params)
            nested_update(self.params, ana_spec)

        if simulation is not None:
            self.data_dir = simulation.data_dir
            self.results = simulation.results
            if self.results is None:
                self.results = load_results(simulation.results_file)
        elif label is not None:
            self.data_dir = os.path.join(data_path, label)
            self.results = load_results(os.path.join(self.data_dir, 'results.npz'))
        else:
            raise ValueError("Give a simulation or the label of a simulation.")

        if network is not None:
            self.area_list = network.area_list
            self.population_list = network.population_list
        else:
            self.area_list = list(self.results.keys())
            self.population_list = list(self.results[self.area_list[0]].keys())

        self.figure_dir = os.path.join(self.data_dir, 'figures')
        plotting.configure(save_dir=self.figure_dir, show=show)

    @property
    def fs(self):
        """Sampling frequency (Hz) of the rate traces."""
        if self.params['fs'] is not None:
            return self.params['fs']
        t = self.results[self.area_list[0]][self.population_list[0]]['time_bins']
        return 1000.0 / float(np.mean(np.diff(t)))

    """
    Mean rates
    """

    def mean_rate_matrix(self, key='mean_rate_spikes_Hz'):
        """Mean rates, populations x areas."""
        return mean_rate_matrix(self.results, self.area_list,
                                self.population_list, key=key)

    def population_statistics(self, key='mean_rate_spikes_Hz'):
        """Mean, standard deviation and median across areas per population."""
        return ah.population_statistics(self.mean_rate_matrix(key))

    def plot_mean_rates(self, annotate=True):
        title = "Mean firing rates (Hz), mesoscopic model"
        if self.network is not None:
            gp = self.network.params['gif_params']
            title += f", rho_0 = {gp['rho_0']}, Delta_V = {gp['Delta_V']}"
        plotting.plot_mean_rate_heatmap(self.mean_rate_matrix(), self.area_list,
                                        self.population_list, title=title,
                                        annotate=annotate)

    def plot_population_rates(self, plot_type='box', log_scale=False, values=True):
        plotting.plot_population_rates(self.mean_rate_matrix(),
                                       self.population_list,
                                       plot_type=plot_type,
                                       log_scale=log_scale, values=values)

    def log_experiment(self, filepath, g_value, rng_seed, overwrite=True):
        """Add the mean rate statistics of this run to an experiment log (json)."""
        stats = self.population_statistics()
        matrix = self.mean_rate_matrix()
        return ah.log_experiment(
            filepath=filepath, g_value=g_value, rng_seed=rng_seed,
            mean_network_activity=matrix.mean(),
            activity_per_pop=stats['mean'], std_per_pop=stats['std'],
            median_per_pop=stats['median'], pop_list=self.population_list,
            overwrite=overwrite)

    """
    Time series
    """

    def plot_rate_dynamics(self, areas, pops, key='rate_from_spikes', **kwargs):
        plotting.plot_rate_dynamics(self.results, areas, pops, key=key, **kwargs)

    def plot_population_activity(self, areas, pops=None, **kwargs):
        pops = pops if pops is not None else self.population_list
        plotting.plot_population_activity(self.results, areas, pops, **kwargs)

    def plot_phase_space(self, area, pop_x='4E', pop_y='4I', key='rate_from_mean'):
        plotting.plot_phase_space(self.results, area, pop_x, pop_y, key=key)

    def plot_activity_heatmap(self):
        act = ah.activity_matrix(self.results, self.area_list, self.population_list)
        plotting.plot_activity_heatmap(act, self.area_list, self.population_list)

    """
    Burstiness
    """

    def burstiness(self, areas=None, pops=None):
        """DataFrame with burst statistics of the chosen populations."""
        areas = areas if areas is not None else self.area_list
        pops = pops if pops is not None else self.population_list
        return ah.analyze_all_nodes(
            self.results, areas, pops,
            transient_cut=self.params['transient_cut'],
            thresh_percentile=self.params['burst_percentile'],
            hmm_states=self.params['hmm_states'])

    def plot_burstiness(self, df=None, **kwargs):
        df = df if df is not None else self.burstiness()
        return plotting.plot_burstiness_relationships(df, **kwargs)

    """
    Spectra
    """

    def plot_power_spectrum(self, area, pops=None, **kwargs):
        pops = pops if pops is not None else self.population_list
        kwargs.setdefault('nperseg', self.params['nperseg'])
        kwargs.setdefault('beta_band', self.params['beta_band'])
        if self.network is not None:
            gp = self.network.params['gif_params']
            npar = self.network.params['neuron_params']
            kwargs.setdefault('rho_0', gp['rho_0'])
            kwargs.setdefault('Delta_V', gp['Delta_V'])
            kwargs.setdefault('tau_syn_ex_ms', npar['tau_syn_E'])
            kwargs.setdefault('tau_syn_in_ms', npar['tau_syn_I'])
        if self.simulation is not None:
            kwargs.setdefault('seed_val', self.simulation.params['rng_seed'])
        plotting.plot_power_spectrum(self.results, area, pops, self.fs, **kwargs)

    def beta_power_matrix(self, new_area_list, nperseg=300):
        """Beta power of every population, populations x areas."""
        if new_area_list:
            area_list = new_area_list
        else:
            area_list = self.area_list

        return ah.beta_power_matrix(self.results, area_list,
                                    self.population_list, self.fs,
                                    nperseg=nperseg,
                                    band=self.params['beta_band'])

    def area_beta_power(self, new_area_list):
        """Beta power of the summed rate of each area."""
        if new_area_list:
            area_list = new_area_list
        else:
            area_list = self.area_list

        
        return ah.area_band_power(self.results, area_list,
                                  self.population_list, self.fs,
                                  nperseg=self.params['nperseg'],
                                  band=self.params['beta_band'])

    def plot_beta_power(self, new_area_list, type = None):

        if new_area_list:
            area_list = new_area_list
        else:
            area_list = self.area_list


        if type == 'beta_power_map':
            plotting.plot_beta_power_map(self.beta_power_matrix(area_list), area_list,
                                         self.population_list,
                                     title="Spatial map of beta power")
        if type == 'area_level_beta_power':
            plotting.plot_area_beta_power(self.area_beta_power(area_list), area_list,
                                          band=self.params['beta_band'])

    """
    Fixed points
    """

    def fixed_points(self, g=None, n_starts=300, seed=0, mu_in_mV=False, **kwargs):
        """
        Fixed points of the mean-field rate equations (see theory.py).
        g: relative inhibitory weight (default: the value of the model).
        """
        if self.network is None:
            raise ValueError("fixed_points needs the network (weights and input).")
        net = self.network
        weights = net.compute_synaptic_weights(g=g)
        mean_input = net.mean_input_mV if mu_in_mV else net.mean_input_pA
        W_flat = theory.build_flat_weight_matrix(weights, self.area_list,
                                                 self.population_list)
        mu_flat = theory.build_flat_mu_vector(mean_input, self.area_list,
                                              self.population_list)
        gp = net.params['gif_params']
        npar = net.params['neuron_params']
        kwargs.setdefault('c_hazard', gp['rho_0'])
        kwargs.setdefault('DeltaV', gp['Delta_V'])
        kwargs.setdefault('vth', npar['v_thresh'] - npar['v_rest'])
        kwargs.setdefault('taum', npar['tau_m'])
        return theory.find_fixed_points(W_flat, mu_flat, n_starts=n_starts,
                                        seed=seed, **kwargs)
