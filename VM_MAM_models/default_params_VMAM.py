from Visuo_motor_mam.config import base_path
import numpy as np
import json
import os

with open(base_path / "VM_MAM_models/visuomotor_data/visuomotor_raw_data.json", "r") as f:

    raw_data = json.load(f)


av_indegree_Cragg = raw_data['av_indegree_Cragg']
av_indegree_OKusky = raw_data['av_indegree_OKusky']



complete_area_list = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd',
                      'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd',
                      'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp',
                      'STPa', '46', 'AITd', 'TH']

population_list = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']



"""
Network parameters
"""
network_params = {
    # Surface area of each area in mm^2
    'surface': 1.0,
    # Scaling of population sizes
    'N_scaling': 1.,
    # Scaling of indegrees
    'K_scaling': 1.,
    # Absolute path to the file holding full-scale rates for scaling
    # synaptic weights
    'fullscale_rates': None,
    # Check whether NEST 2 or 3 is used. No straight way of checking this is
    # available. But PrintNetwork was removed in NEST 3, so checking for its
    # existence should suffice.
    #'USING_NEST_3': 'PrintNetwork' not in dir(nest)
}



connection_params = {
    
    # Relative inhibitory synaptic strength (in relative units).
    'g': -11.,

    # compute average indegree in V1 from data
    'av_indegree_V1': np.mean([av_indegree_Cragg, av_indegree_OKusky]),

    # synaptic volume density
    # area-specific --> conserves average in-degree
    # constant --> conserve syn. volume density
    'rho_syn': 'constant',

    # Increase the external Poisson indegree onto 5E and 6E
    'fac_nu_ext_5E': 1.125,
    'fac_nu_ext_6E': 1.41666667,
    # to increase the ext. input to 23E and 5E in area TH
    'fac_nu_ext_TH': 1.2,

    # synapse weight parameters for current-based neurons
    # excitatory intracortical synaptic weight (mV)
    'PSP_e': 0.15,
    'PSP_e_23_4': 0.3,
    # synaptic weight (mV) for external input
    'PSP_ext': 0.15,

    # scaling factor for cortico-cortical connections (chi)
    # Default value is 1.9 which reproduces the "Metastable" state
    # activity described in Schmidt et al. (2018). 
    # A weight factor of 1.0 produces Ground state activity.
    'cc_weights_factor': 1.9,
    # Factor to scale cortico-cortical inh. weights in relation
    # to exc. weights (chi_I). Default is 2.0 to reproduce metastable state
    # activity. For ground state activity, set to 1.0.
    'cc_weights_I_factor': 2.,

}


network_params.update({'connection_params': connection_params})