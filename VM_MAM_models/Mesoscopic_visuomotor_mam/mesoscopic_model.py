"""
mesoscopic_model
================

Mesoscopic_Visuomotor_Model collects everything that defines the mesoscopic visuomotor
model before it is simulated:

- population sizes (1 mm^2 patch) from visuomotor_Data_Model.json
- connection probabilities from visuomotor_connection_prob_data.json
- mean input of each population from the Siegert mean-field model
- synaptic weights (pA) and delays (ms) for every pair of populations

Every area consists of eight populations (23E, 23I, 4E, 4I, 5E, 5I, 6E, 6I).
Each population is one gif_pop_psc_exp node in NEST, so a connection between
two populations is a single connection with a weight that already contains
the connection probability (mean-field connectivity).

Usage
-----
    M = Mesoscopic_Visuomotor_Model(network_spec={'connection_params': {'g': -14.}},
                        simulation=True,
                        sim_spec={'sim_time_ms': 700., 'rng_seed': 4})
    M.simulation.simulate()
"""
import os
import pprint
from copy import deepcopy

from default_params import (check_custom_params, complete_area_list,
                            nested_update, network_params, population_list)
from mesoscopic_helpers import dict_hash, load_json
from prepare_data import ensure_model_data


class Mesoscopic_Visuomotor_Model:
    """
    Mesoscopic visuomotor model.

    Parameters
    ----------
    network_spec : dict or str, optional
        Custom network parameters, either as (partial) dictionary or as path
        to a json file. Merged into default_params.network_params.
    simulation : bool
        If True, a Simulation instance is created (M.simulation).
    sim_spec : dict, optional
        Custom simulation parameters, merged into default_params.sim_params.
    """

    def __init__(self, network_spec=None, simulation=False, sim_spec=None,build_missing_data=True):

        self.params = deepcopy(network_params)
        if network_spec is not None:
            if isinstance(network_spec, str):
                network_spec = load_json(network_spec)
            check_custom_params(network_spec, network_params)
            nested_update(self.params, network_spec)

        areas = self.params['areas_simulated']
        self.area_list = list(areas) if areas else list(complete_area_list)
        self.population_list = list(population_list)
        self.label = dict_hash(self.params)

        # This ensure the raw data is already loaded and processed
        if build_missing_data:
            ensure_model_data(self.params['data_dir'], self.params['data_files'], self.params['ext_input_params'])
        self.load_data()

        self.compute_input()
        self.synaptic_weights = self.compute_synaptic_weights()
        self.delays = self.compute_delays()

        self.simulation = None
        if simulation:
            self.init_simulation(sim_spec)

    def __str__(self):
        s = "Mesoscopic visuomotor model\n"
        s += f"label: {self.label}\n"
        s += f"areas: {len(self.area_list)}, populations per area: " \
             f"{len(self.population_list)}\n"
        s += pprint.pformat({k: v for k, v in self.params.items()
                             if k not in ('data_dir', 'data_files')})
        return s

    def __eq__(self, other):
        return self.label == other.label

    def __hash__(self):
        return hash(self.label)

    def init_simulation(self, sim_spec=None):
        from simulation import Simulation
        self.simulation = Simulation(self, sim_spec)

    """
    Data
    """

    def load_data(self):
        """Load the json files listed in params['data_files']."""
        data_dir = self.params['data_dir']
        files = self.params['data_files']

        model_data = load_json(os.path.join(data_dir, files['model_data']))
        self.connection_probs = load_json(
            os.path.join(data_dir, files['connection_probs']))
        self.mean_input_mV = load_json(os.path.join(data_dir, files['mean_input']))

        # Neuron numbers in a 1 mm^2 patch
        self.neuron_numbers = model_data['neuron_numbers']
        self.distances = model_data['distances']

    def compute_input(self):
        """
        Constant input current I_e (pA) of each population from the mean
        input mu (mV): I_e = mu * C_m / tau_m.
        """
        npar = self.params['neuron_params']
        self.mean_input_pA = {
            area: {pop: self.mean_input_mV[area][pop] * npar['cm'] / npar['tau_m']
                   for pop in self.population_list}
            for area in self.area_list}

    """
    Synaptic weights and delays
    """

    def compute_synaptic_weights(self, g=None, **overrides):
        """
        Synaptic weights (pA) for all pairs of populations,
        weights[target_area][target_pop][source_area][source_pop].

        Parameters
        ----------
        g : float, optional
            Relative inhibitory weight. Defaults to the value in the
            connection parameters.
        overrides : optional
            Other entries of connection_params (for example
            cc_weights_factor) to use instead of the stored values.
        """
        cp = dict(self.params['connection_params'])
        cp.update(overrides)
        if g is not None:
            cp['g'] = g
        npar = self.params['neuron_params']
        C_m = npar['cm']

        weights = {}
        for target_area in self.area_list:
            weights[target_area] = {}
            for target_pop in self.population_list:
                weights[target_area][target_pop] = {}
                for source_area in self.area_list:
                    weights[target_area][target_pop][source_area] = {}
                    for source_pop in self.population_list:
                        p = self.connection_probs[target_area][target_pop][
                            source_area][source_pop]

                        # PSC of the source population and its synaptic time constant
                        if 'E' in source_pop:
                            J_psc = cp['w_mean']
                            tau_syn = npar['tau_syn_E']
                        else:
                            J_psc = cp['g'] * cp['w_mean']
                            tau_syn = npar['tau_syn_I']

                        # PSC -> PSP (mV)
                        J_psp = J_psc * tau_syn / C_m
                        J_psp /= cp['scaling_factor']

                        # Mean-field connectivity: weight times connection probability
                        if cp['meanfield_connectivity']:
                            J_psp *= p

                        # PSP -> PSC in NEST units (pA)
                        J_pA = J_psp * (C_m / tau_syn)

                        # Cortico-cortical scaling
                        if source_area != target_area:
                            J_pA *= cp['cc_weights_factor']
                            if 'I' in target_pop:
                                J_pA *= cp['cc_weights_I_factor']

                        # Intra-area L4E -> L23E
                        if (source_area == target_area and
                                source_pop == '4E' and target_pop == '23E'):
                            J_pA *= cp['L4E_L23E_factor']

                        weights[target_area][target_pop][source_area][
                            source_pop] = J_pA
        return weights

    def compute_delays(self):
        """
        Delays (ms), delays[target_area][target_pop][source_area][source_pop].

        Intra-area: fixed delay for excitatory and inhibitory sources.
        Inter-area: distance divided by the conduction speed.
        """
        cp = self.params['connection_params']
        delays = {}
        for target_area in self.area_list:
            delays[target_area] = {}
            for target_pop in self.population_list:
                delays[target_area][target_pop] = {}
                for source_area in self.area_list:
                    delays[target_area][target_pop][source_area] = {}
                    for source_pop in self.population_list:
                        if target_area == source_area:
                            d = cp['delay_E'] if 'E' in source_pop else cp['delay_I']
                        else:
                            d = (self.distances[target_area][source_area]
                                 / cp['interarea_speed'])
                        delays[target_area][target_pop][source_area][
                            source_pop] = d
        return delays
