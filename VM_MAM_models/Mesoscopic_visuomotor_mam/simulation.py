"""
simulation
==========

Simulation of the mesoscopic visuomotor model with NEST.

The Simulation class creates one gif_pop_psc_exp node per population,
connects all populations, records the population activity, runs the
simulation and stores the results in data_path/<label>/:

    params.json    network and simulation parameters
    results.npz    time series and mean rates of every population
    summary.json   mean rates (spikes and multimeter) of every population

results[area][pop] contains
    mean_rate_spikes_Hz  mean rate from the spike recorder after the transient
    mean_rate_mean_Hz    mean rate from the multimeter ('mean') after the transient
    time_bins            bin centres (ms) of rate_from_spikes
    rate_from_spikes     rate (spikes/s per neuron) from the spike recorder
    times_mm             sampling times (ms) of the multimeter
    rate_from_mean       rate (spikes/s per neuron) from the multimeter ('mean')
"""
import os
import shutil
from copy import deepcopy

import numpy as np

from config import data_path
from default_params import check_custom_params, nested_update, sim_params
from mesoscopic_helpers import (dict_hash, load_json, load_results,
                                results_summary, save_json, save_results)


class Simulation:
    """
    Simulation of a Mesoscopic_Visuomotor_Model.

    Parameters
    ----------
    network : Mesoscopic_Visuomotor_Model
    sim_spec : dict or str, optional
        Custom simulation parameters (dictionary or json file), merged into
        default_params.sim_params.
    """

    def __init__(self, network, sim_spec=None):
        self.network = network
        self.params = deepcopy(sim_params)
        if sim_spec is not None:
            if isinstance(sim_spec, str):
                sim_spec = load_json(sim_spec)
            check_custom_params(sim_spec, sim_params)
            nested_update(self.params, sim_spec)

        self.label = dict_hash({'network': network.label, 'sim': self.params})
        self.data_dir = os.path.join(data_path, self.label)
        self.results = None

        self.populations = None
        self.multimeters = None
        self.spike_recorders = None

    def __str__(self):
        return f"Simulation {self.label} (model {self.network.label})"

    def __eq__(self, other):
        return self.label == other.label

    def __hash__(self):
        return hash(self.label)

    @property
    def results_file(self):
        return os.path.join(self.data_dir, 'results.npz')

    def save_parameters(self):
        save_json({'network': self.network.params, 'simulation': self.params,
                   'network_label': self.network.label, 'label': self.label},
                  os.path.join(self.data_dir, 'params.json'))

    """
    NEST network
    """

    def prepare(self):
        """Reset and configure the NEST kernel."""
        import nest
        nest.set_verbosity("M_WARNING")
        nest.ResetKernel()
        nest.SetKernelStatus({
            "resolution": self.params['resolution'],
            "print_time": self.params['print_time'],
            "local_num_threads": self.params['local_num_threads'],
            "rng_seed": self.params['rng_seed'],
        })

    def create_populations(self):
        """Create one gif_pop_psc_exp node for every population."""
        import nest
        net = self.network
        npar = net.params['neuron_params']
        gpar = net.params['gif_params']

        V_reset = npar['v_reset'] - npar['v_rest']
        V_T_star = npar['v_thresh'] - npar['v_rest']

        self.populations = {}
        for area in net.area_list:
            self.populations[area] = {}
            for pop in net.population_list:
                N_pop = int(np.round(net.neuron_numbers[area][pop]))

                node = nest.Create("gif_pop_psc_exp", 1)
                nest.SetStatus(node, {
                    "C_m": npar['cm'],
                    "I_e": np.array([net.mean_input_pA[area][pop]], dtype=float),
                    "lambda_0": gpar['rho_0'],
                    "Delta_V": gpar['Delta_V'],
                    "tau_m": npar['tau_m'],
                    "V_T_star": V_T_star,
                    "V_reset": V_reset,
                    "len_kernel": gpar['len_kernel'],
                    "N": N_pop,
                    "t_ref": npar['tau_refrac'],
                    "tau_syn_ex": npar['tau_syn_E'],
                    "tau_syn_in": npar['tau_syn_I'],
                    "E_L": gpar['E_L'],
                })
                self.populations[area][pop] = node

    def connect_populations(self):
        """Connect all pairs of populations (one connection per pair)."""
        import nest
        net = self.network
        count = 0
        for ta, tp, sa, sp in self._population_pairs():
            weight = net.synaptic_weights[ta][tp][sa][sp]
            if self.params['skip_zero_weights'] and weight == 0:
                continue
            nest.Connect(self.populations[sa][sp], self.populations[ta][tp],
                         "all_to_all",
                         {"weight": float(weight),
                          "delay": float(net.delays[ta][tp][sa][sp]),
                          "synapse_model": "static_synapse"})
            count += 1
        print(f"Connections created: {count}")

    def create_recording_devices(self):
        """A multimeter and a spike recorder for every population."""
        import nest
        net = self.network
        self.multimeters = {}
        self.spike_recorders = {}
        for area in net.area_list:
            self.multimeters[area] = {}
            self.spike_recorders[area] = {}
            for pop in net.population_list:
                node = self.populations[area][pop]
                mm = nest.Create("multimeter", {
                    "record_from": ["n_events", "mean"],
                    "interval": self.params['record_interval_ms']})
                sr = nest.Create("spike_recorder")
                nest.Connect(mm, node)
                nest.Connect(node, sr)
                self.multimeters[area][pop] = mm
                self.spike_recorders[area][pop] = sr

    def _population_pairs(self):
        net = self.network
        for ta in net.area_list:
            for tp in net.population_list:
                for sa in net.area_list:
                    for sp in net.population_list:
                        yield ta, tp, sa, sp

    """
    Simulation and results
    """

    def simulate(self, force=False):
        """
        Build the network, simulate and store the results.

        If results of this simulation (same label) exist, they are loaded
        instead, unless force is True.
        """
        import nest

        if os.path.exists(self.results_file) and not force:
            print(f"Loading existing results from {self.data_dir}")
            self.results = load_results(self.results_file)
            return self.results

        self.prepare()
        print("NEST initialized")
        self.create_populations()
        print("Populations created")
        self.connect_populations()
        self.create_recording_devices()
        print("Recording devices connected")

        print(f"Simulating {self.params['sim_time_ms']} ms...")
        nest.Simulate(self.params['sim_time_ms'])

        self.results = self.extract_results()
        self.save()
        print(f"\n{'=' * 80} \n Simulation finished. \n Results in {self.data_dir}")
        return self.results

    def extract_results(self):
        """Rates of all populations from the recording devices."""
        import nest
        net = self.network
        p = self.params
        T = p['sim_time_ms']
        interval = p['record_interval_ms']
        transient = p['transient_ms']

        # gif_pop_psc_exp reports counts per simulation step, so the rates
        # from the multimeter are divided by the resolution
        dt_step_s = p['resolution'] / 1000.0
        dt_bin_s = interval / 1000.0
        bins = np.arange(0.0, T + interval, interval)
        time_bins = bins[:-1] + 0.5 * interval

        results = {}
        for area in net.area_list:
            results[area] = {}
            for pop in net.population_list:
                N_pop = net.neuron_numbers[area][pop]

                mm_events = nest.GetStatus(self.multimeters[area][pop])[0]["events"]
                times_mm = np.asarray(mm_events["times"], dtype=float)
                mean = np.asarray(mm_events["mean"], dtype=float)

                sr_events = nest.GetStatus(self.spike_recorders[area][pop])[0]["events"]
                spike_times = np.asarray(sr_events["times"], dtype=float)
                spike_counts, _ = np.histogram(spike_times, bins=bins)

                if N_pop > 0:
                    rate_from_mean = mean / N_pop / dt_step_s
                    rate_from_spikes = spike_counts / N_pop / dt_bin_s
                else:
                    # empty population
                    rate_from_mean = np.zeros_like(mean)
                    rate_from_spikes = np.zeros(len(time_bins))

                steady_mm = times_mm >= transient
                steady_spk = time_bins >= transient

                results[area][pop] = {
                    "mean_rate_spikes_Hz": rate_from_spikes[steady_spk].mean(),
                    "mean_rate_mean_Hz": rate_from_mean[steady_mm].mean(),
                    "times_mm": times_mm,
                    "rate_from_mean": rate_from_mean,
                    "time_bins": time_bins,
                    "rate_from_spikes": rate_from_spikes,
                }
        return results

    def save(self):
        """Write parameters, results and summary to data_dir."""
        os.makedirs(self.data_dir, exist_ok=True)
        self.save_parameters()
        save_results(self.results, self.results_file)
        save_json(results_summary(self.results),
                  os.path.join(self.data_dir, 'summary.json'))

    def load(self):
        """Load results of a previous run with the same label."""
        self.results = load_results(self.results_file)
        return self.results

    def delete(self):
        """Remove the data directory of this simulation."""
        shutil.rmtree(self.data_dir, ignore_errors=True)
