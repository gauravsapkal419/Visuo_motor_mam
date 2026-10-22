"""
default_params
==============

Default parameters of the mesoscopic visuomotor model.

network_params : areas, neuron and connection parameters
sim_params     : NEST settings, simulation time, recording
analysis_params: settings of the analysis (spectra, bursts)

Custom values are passed to MesoscopicModel / Simulation / Analysis as
(partial) dictionaries and merged into these defaults with nested_update.
"""
from copy import deepcopy

from config import model_data_path

complete_area_list = ['V1', 'V2', 'V3', 'V4', 'V3A', 'DP', 'V6', 'MT', 'V4t',
                      'PIP', 'TEO', 'V6A', 'TEOm', '8m', '8l', 'MIP', 'F1',
                      '45B', 'STPc', 'LIP', 'PBc', '8r', 'MST', 'F3', '7A', 'F5',
                      'F4', '46d', '9-46v', 'VIP', '9-46d', 'TEpd', '46v', 'STPi',
                      'F2', 'F7', 'PGa', 'F6', 'TEpv', 'STPr', 'IPa', 'TEad',
                      'TH-TF']

population_list = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']

motor_areas = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']

network_params = {
    # None: all areas of complete_area_list
    'areas_simulated': None,

    # input files (json), all inside data_dir
    'data_dir': model_data_path,
    'data_files': {
        'model_data': 'visuomotor_Data_Model.json',
        'connection_probs': 'visuomotor_connection_prob_data.json',
        # mean input (mV) of each population from the Siegert mean-field model
        'mean_input': 'mu_values.json',
        'ext_indegrees' : 'visuomotor_indegree_data.json'
    },

    'neuron_params': {
        'cm': 250.,         # pF
        'i_offset': 0.0,    # nA
        'tau_m': 10.0,      # ms
        'tau_refrac': 2.0,  # ms
        'tau_syn_E': 0.5,   # ms
        'tau_syn_I': 0.5,   # ms
        'v_reset': -65.0,   # mV
        'v_rest': -65.0,    # mV
        'v_thresh': -50.0,  # mV
    },

    # gif_pop_psc_exp
    'gif_params': {
        'rho_0': 10.0,      # baseline rate at threshold (Hz)
        'Delta_V': 5.0,     # softness of the threshold (mV)
        'len_kernel': -1,   # -1: automatic history size
        'E_L': 0.0,
    },

    'connection_params': {
        'w_mean': 87.8,               # mean PSC of excitatory synapses (pA)
        'g': -14.,                    # relative inhibitory weight
        'cc_weights_factor': 1.,      # scaling of cortico-cortical weights
        'cc_weights_I_factor': 1.,    # extra factor for cc weights onto I
        'scaling_factor': 1.,
        'meanfield_connectivity': True,   # weight *= connection probability
        'L4E_L23E_factor': 2.,        # intra-area 4E -> 23E weight factor
        'delay_E': 1.5,               # intra-area delay from E (ms)
        'delay_I': 0.75,              # intra-area delay from I (ms)
        'interarea_speed': 3.5,       # mm/ms, delay = distance / speed
    },

    'ext_input_params': {
    'tau_mem': 10.,        # ms
    'J_ex': 0.15,          # mV, PSP of one external spike
    'rate_ext': 10.,       # spikes/s per external input
    'fac_nu_ext_5E': 1.,
    'fac_nu_ext_6E': 1.,
    },
}

sim_params = {
    'sim_time_ms': 700.0,
    'resolution': 0.1,             # ms
    'record_interval_ms': 1.0,     # ms
    'transient_ms': 100.0,         # ms, excluded from the mean rates
    'rng_seed': 456123,
    'local_num_threads': 4,
    'print_time': True,
    'skip_zero_weights': False,
}

analysis_params = {
    # sampling frequency (Hz) of the rate traces. None: 1000 / record_interval_ms
    'fs': 100,
    'nperseg': 300,
    'beta_band': (13, 30),         # Hz
    'transient_cut': 50,           # bins removed before the burst analysis
    'burst_percentile': 75,
    'hmm_states': 2,
}


def nested_update(d, d_update):
    """Update dictionary d recursively with the entries of d_update."""
    for key, value in d_update.items():
        if isinstance(value, dict) and isinstance(d.get(key), dict):
            nested_update(d[key], value)
        else:
            d[key] = deepcopy(value)


def check_custom_params(custom, default, path=''):
    """Raise a KeyError if custom contains a key that is not in default."""
    for key, value in custom.items():
        if key not in default:
            raise KeyError(f"Unknown parameter: {path}{key}")
        if isinstance(value, dict) and isinstance(default[key], dict):
            check_custom_params(value, default[key], path=f"{path}{key}.")
