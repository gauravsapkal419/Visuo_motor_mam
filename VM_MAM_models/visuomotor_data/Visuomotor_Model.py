"""
VisuomotorModel
===============

This script defines the visuomotor multi-area model. It follows the
structure of Model.py of the multi-area model (Schmidt et al. 2018).
It loads the data prepared by Visuomotor_Data.py and
Visuomotor_Data_Processing.py (visuomotor_raw_data.json and
visuomotor_processed_data.json) and computes

1. neuronal numbers for each population (scaled to the model surface),
2. intrinsic (type I) and cortico-cortical (type III) synapse numbers,
3. the external inputs to each population,
4. connection probabilities and indegrees for every connection.

Output files (written next to this script)
------------------------------------------
visuomotor_Data_Model<out_label>.json
    Neuron numbers, synapse numbers and related data.
visuomotor_connection_prob_data<out_label>.json
    Connection probability for each pair of populations.
visuomotor_indegree_data<out_label>.json
    Indegree for each pair of populations, including external input.

Population sizes and synapse numbers refer to areas with the surface
defined in network_params['surface'].
"""
import json
import os
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
import re

import numpy as np
import scipy
import scipy.integrate
from nested_dict import nested_dict

# Add the project root to the Python path (as in the notebook)
try:
    sys.path.append(str(Path(__file__).resolve().parent.parents[2]))
except IndexError:
    pass

try:
    from Visuo_motor_MAM.VM_MAM_models.default_params_VMAM import network_params
except ImportError:
    from Visuo_motor_mam.VM_MAM_models.default_params_VMAM import network_params
    


def compute_Model_params(out_label=''):
    """
    Compute the parameters of the network: size of the populations,
    external inputs to them and number of synapses in every connection.

    Parameters
    ----------
    out_label : str
        Label that is appended to the output file name.

    Returns
    -------
    collected_data : dict
        The data that is written to visuomotor_Data_Model<out_label>.json.
    """
    basepath = os.path.abspath(os.path.dirname(__file__))

    raw_fn = os.path.join(basepath, 'visuomotor_raw_data.json')
    proc_fn = os.path.join(basepath, 'visuomotor_processed_data.json')

    """
    Load data
    """
    with open(raw_fn, 'r') as f:
        raw_data = json.load(f)
    with open(proc_fn, 'r') as f:
        processed_data = json.load(f)

    FLN_EDR_completed = processed_data['FLN_completed']
    SLN_Data = processed_data['SLN_completed']
    Distance_Data = raw_data['thom_distance_data_markov']
    Area_surfaces = raw_data['surface_data']
    Intra_areal = raw_data['Intrinsic_Connectivity']
    Intrinsic_FLN_Data = raw_data['Intrinsic_FLN_Data']
    num_V1 = raw_data['num_V1']
    binzegger_data = raw_data['Binzegger_Data']
    total_thicknesses = processed_data['total_thicknesses']
    laminar_thicknesses = processed_data['laminar_thicknesses']
    neuronal_numbers_fullscale = processed_data['realistic_neuronal_numbers']
    Binary_symmetric_connectivity = processed_data['bin_sym_conn_data']

    """
    Define area and population lists.
    Define termination and origin patterns according
    to Felleman and van Essen 91
    """
    area_list = ["V1", "V2", "V3", "V4", "V3A", "DP", "V6", "MT", "V4t",
                 "PIP", "TEO", "V6A", "TEOm", "8m", "8l", "MIP", "F1", "45B",
                 "STPc", "LIP", "PBc", "8r", "MST", "F3", "7A", "F5", "F4",
                 "46d", "9-46v", "VIP", "9-46d", "TEpd", "46v", "STPi",
                 "F2", "F7", "PGa", "F6", "TEpv", "STPr", "IPa", "TEad",
                 "TH-TF"]
    # Order of areas in the output files (motor areas last)
    output_area_list = ['V1', 'V2', 'V3', 'V4', 'V3A', 'DP', 'V6', 'MT', 'V4t',
                        'PIP', 'TEO', 'V6A', 'TEOm', '8m', '8l', 'MIP',
                        '45B', 'STPc', 'LIP', 'PBc', '8r', 'MST', '7A',
                        '46d', '9-46v', 'VIP', '9-46d', 'TEpd', '46v', 'STPi',
                        'PGa', 'TEpv', 'STPr', 'IPa', 'TEad', 'TH-TF',
                        'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
    population_list = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']

    origin_patterns = {'S': ['23E'], 'I': ['5E', '6E'], 'B': ['23E', '5E', '6E']}
    termination_layers = {'F': ['4'], 'M': ['1', '23', '5', '6'],
                          'C': ['1', '23', '4', '5', '6'], 'S': ['1', '23']}

    binzegger_pops = list(binzegger_data.keys())
    binzegger_I_pops = [binzegger_pops[i] for i in range(
        len(binzegger_pops)) if binzegger_pops[i].find('b') != -1]
    binzegger_E_pops = [binzegger_pops[i] for i in range(
        len(binzegger_pops)) if binzegger_pops[i].find('b') == -1]

    # Structure dictionary: for each area, the populations that
    # contain neurons
    structure = {}
    for area in area_list:
        structure[area] = []
        for pop in population_list:
            if neuronal_numbers_fullscale[area][pop] > 0.0:
                structure[area].append(pop)

    """
    Define parameter values
    """
    net_params = deepcopy(network_params)

    # surface area of each area in mm^2
    surface = net_params['surface']
    conn_params = net_params['connection_params']

    # average indegree in V1 to compute the synaptic volume density
    # (determined for V1 and taken to be constant across areas)
    av_indegree_V1 = conn_params['av_indegree_V1']

    # Parameters for the external input and the synaptic weights. They are
    # read here but only needed once the weights are computed.
    fac_nu_ext_5E = conn_params['fac_nu_ext_5E']
    fac_nu_ext_6E = conn_params['fac_nu_ext_6E']
    fac_nu_ext_TH = conn_params['fac_nu_ext_TH']
    PSP_e = conn_params['PSP_e']
    PSP_e_23_4 = conn_params['PSP_e_23_4']
    PSP_ext = conn_params['PSP_ext']
    g = conn_params['g']
    cc_weights_factor = conn_params['cc_weights_factor']
    cc_weights_I_factor = conn_params['cc_weights_I_factor']

    E_specificity = True

    """
    Neuronal numbers
    ----------------
    """
    # Synaptic volume density for each area. The total thickness includes
    # L1. Since L1 can be approximated as having no neurons, rho_syn is a
    # synapse density across all layers.
    rho_syn = {}
    for area in area_list:
        rho_syn[area] = (av_indegree_V1 * neuronal_numbers_fullscale['V1']['total']
                         / (Area_surfaces['V1'] * total_thicknesses['V1']))

    # Population sizes: scale the realistic sizes down to the model surface
    neuronal_numbers = {}
    for a in neuronal_numbers_fullscale:
        neuronal_numbers[a] = {}
        for pop in neuronal_numbers_fullscale[a]:
            neuronal_numbers[a][pop] = (neuronal_numbers_fullscale[a][pop]
                                        / Area_surfaces[a] * surface)

    """
    1. Radius-dependent connection probabilities of the microcircuit.
    """
    # constants for the connection probability transfer
    # from Potjans & Diesmann (2014) (PD14)
    sigma = 0.29653208289812366  # mm
    C0 = 0.1429914097112598

    # average connection probability with the method from PD14
    r_PD14 = np.sqrt(1. / np.pi)
    C_prime_mean_PD14 = (2. / (r_PD14 ** 2) * C0 * sigma ** 2
                         * (1. - np.exp(-r_PD14 ** 2 / (2 * sigma ** 2))))

    # Calculation based on Sheng (1985), The distance between two random
    # points in plane regions, Theorem 2.4 on the expectation value of
    # arbitrary functions of distance between points in disks.
    def integrand(r, R, sig):
        gauss = np.exp(-r ** 2 / (2 * sig ** 2))
        x1 = np.arctan(np.sqrt((2 * R - r) / (2 * R + r)))
        x2 = np.sin(4 * np.arctan(np.sqrt((2 * R - r) / (2 * R + r))))
        factor = 4 * x1 - x2
        return r * gauss * factor

    # Approximation of log(1 - x), needed for large areas
    def log_approx(x, limit):
        res = 0.
        for k in range(limit):
            res += x ** (k + 1) * (-1.) ** k / (k + 1)
        return res

    """
    To determine the conversion from the microcircuit model to the
    area-specific composition in the model, the intrinsic FLN has to be
    scaled down from 0.79 to a lower value. Therefore, the procedure is
    executed twice: first for realistic area sizes (Indegree_prime_fullscale)
    and then for areas with the model surface (Indegree_prime).

    Mean connection probability and indegrees for full-scale areas.
    """
    C_prime_fullscale_mean = {}
    for area in area_list:
        R_area = np.sqrt(Area_surfaces[area] / np.pi)  # radius of the disc
        C_prime_fullscale_mean[area] = (
            2 * C0 / Area_surfaces[area]
            * scipy.integrate.quad(integrand, 0, 2 * R_area,
                                   args=(R_area, sigma))[0])

    # Indegree_prime_fullscale[area][target_pop][source_pop] is the
    # indegree per population pair for the full-scale radius of the area.
    Indegree_prime_fullscale = nested_dict()
    for area, target_pop, source_pop in product(area_list, population_list,
                                                population_list):
        # Adjust each PD14 population-pair connection probability
        C = (Intra_areal[target_pop][source_pop]
             * C_prime_fullscale_mean[area] / C_prime_mean_PD14)

        n_t = num_V1[target_pop]['neurons']
        n_s = num_V1[source_pop]['neurons']
        # Limit to choose between np.log and log_approx
        if Area_surfaces[area] < 100.:
            K = int(round(np.log(1.0 - C) / np.log(
                1. - 1. / (n_t * n_s * Area_surfaces[area] ** 2)))) / (
                    n_t * Area_surfaces[area])
        else:
            K = int(round(log_approx(C, 20) / log_approx(
                1. / (n_t * n_s * Area_surfaces[area] ** 2), 20))) / (
                    n_t * Area_surfaces[area])
        Indegree_prime_fullscale[area][target_pop][source_pop] = K
    Indegree_prime_fullscale = Indegree_prime_fullscale.to_dict()

    # Assign the average intrinsic FLN to each area
    mean_Intrinsic_FLN = Intrinsic_FLN_Data['mean']['mean']
    mean_Intrinsic_error = Intrinsic_FLN_Data['mean']['error']
    Intrinsic_FLN_completed_fullscale = {}
    for area in area_list:
        Intrinsic_FLN_completed_fullscale[area] = {
            'mean': mean_Intrinsic_FLN, 'error': mean_Intrinsic_error}

    """
    Mean connection probability, indegrees and intrinsic FLN
    for areas with the model surface.
    """
    C_prime_mean = {}
    for area in area_list:
        R_area = np.sqrt(surface / np.pi)
        C_prime_mean[area] = (
            2 * C0 / surface
            * scipy.integrate.quad(integrand, 0, 2 * R_area,
                                   args=(R_area, sigma))[0])

    Indegree_prime = nested_dict()
    for area, target_pop, source_pop in product(area_list, population_list,
                                                population_list):
        C = (Intra_areal[target_pop][source_pop]
             * C_prime_mean[area] / C_prime_mean_PD14)
        n_t = num_V1[target_pop]['neurons']
        n_s = num_V1[source_pop]['neurons']
        if surface < 100.:
            K = int(round(np.log(1.0 - C) / np.log(
                1. - 1. / (n_t * n_s * surface ** 2)))) / (n_t * surface)
        else:
            K = int(round(log_approx(C, 20) / log_approx(
                1. / (n_t * n_s * surface ** 2), 20))) / (n_t * surface)
        Indegree_prime[area][target_pop][source_pop] = K
    Indegree_prime = Indegree_prime.to_dict()

    Intrinsic_FLN_completed = {}
    for area in area_list:
        average_relation_indegrees = []
        for pop in population_list:
            for pop2 in population_list:
                if Indegree_prime_fullscale[area][pop][pop2] > 0.:
                    average_relation_indegrees.append(
                        Indegree_prime[area][pop][pop2]
                        / Indegree_prime_fullscale[area][pop][pop2])
        Intrinsic_FLN_completed[area] = {
            'mean': mean_Intrinsic_FLN * np.mean(average_relation_indegrees),
            'error': mean_Intrinsic_error}

    """
    2. Conversion factors c_A(R) between the microcircuit and the
    multi-area model areas, for down-scaled and full-scale areas. The
    factor is chosen such that the scaling preserves the synapse budget
    of the area and the relative indegrees between connections.
    """
    conversion_factor = {}
    for area in area_list:
        Nsyn_int_prime = 0.0
        for target_pop in population_list:
            for source_pop in population_list:
                Nsyn_int_prime += (Indegree_prime[area][target_pop][source_pop]
                                   * neuronal_numbers[area][target_pop])
        conversion_factor[area] = (
            Intrinsic_FLN_completed[area]['mean'] * rho_syn[area] * surface
            * total_thicknesses[area] / Nsyn_int_prime)

    conversion_factor_fullscale = {}
    for area in area_list:
        Nsyn_int_prime = 0.0
        for target_pop in population_list:
            for source_pop in population_list:
                Nsyn_int_prime += (
                    Indegree_prime_fullscale[area][target_pop][source_pop]
                    * neuronal_numbers_fullscale[area][target_pop])
        conversion_factor_fullscale[area] = (
            Intrinsic_FLN_completed_fullscale[area]['mean'] * rho_syn[area]
            * Area_surfaces[area] * total_thicknesses[area] / Nsyn_int_prime)

    def num_IA_synapses(area, target_pop, source_pop, area_model='micro'):
        """
        Compute the number of intrinsic synapses from source_pop to
        target_pop in an area.

        Parameters
        ----------
        area : str
            Area for which to compute connectivity.
        target_pop : str
            Target population of the connection.
        source_pop : str
            Source population of the connection.
        area_model : str
            'real' for the area with realistic surface area, 'micro' for
            the area with the model surface. Defaults to 'micro'.

        Returns
        -------
        Nsyn : float
            Number of synapses.
        """
        if area_model == 'micro':
            c_area = conversion_factor[area]
            In_degree = Indegree_prime[area][target_pop][source_pop]
            num_source = neuronal_numbers[area][source_pop]
            num_target = neuronal_numbers[area][target_pop]
        if area_model == 'real':
            c_area = conversion_factor_fullscale[area]
            In_degree = Indegree_prime_fullscale[area][target_pop][source_pop]
            num_source = neuronal_numbers_fullscale[area][source_pop]
            num_target = neuronal_numbers_fullscale[area][target_pop]

        if num_source == 0 or num_target == 0:
            Nsyn = 0
        else:
            Nsyn = c_area * In_degree * num_target
        return Nsyn

    """
    3. Number of intrinsic (type I) synapses
    """
    synapse_numbers = nested_dict()
    for area, target_pop, source_pop in product(area_list, population_list,
                                                population_list):
        N_syn = num_IA_synapses(area, target_pop, source_pop)
        synapse_numbers[area][target_pop][area][source_pop] = N_syn

    # Total number of type I synapses for each area
    synapses_type_I = {}
    for area in area_list:
        N_syn_i = 0.0
        for target_pop in population_list:
            for source_pop in population_list:
                N_syn_i += num_IA_synapses(area, source_pop, target_pop)
        synapses_type_I[area] = N_syn_i

    """
    4. Number of type II synapses
    """
    synapses_type_II = {}
    for target_area in area_list:
        s_area = 0.0
        for target_pop in population_list:
            syn = 0.0
            if neuronal_numbers[target_area][target_pop] != 0.0:
                for source_pop in population_list:
                    micro_in_degree = (
                        num_IA_synapses(target_area, target_pop, source_pop)
                        / neuronal_numbers[target_area][target_pop])
                    real_in_degree = (
                        num_IA_synapses(target_area, target_pop, source_pop,
                                        area_model='real')
                        / neuronal_numbers_fullscale[target_area][target_pop])
                    syn += ((real_in_degree - micro_in_degree)
                            * neuronal_numbers[target_area][target_pop])
            s_area += syn
        synapses_type_II[target_area] = s_area

    """
    Cortico-cortical synapses
    -------------------------
    1. Normalize the FLN values of the cortico-cortical connections
       to (1 - FLN_i - 0.013).
       1.3%: subcortical inputs, data from Markov et al. (2011)
    """
    FLN_completed = {}
    for target_area in area_list:
        FLN_completed[target_area] = {}
        cc_proportion = (1. - Intrinsic_FLN_completed_fullscale[target_area]['mean']
                         - 0.013)
        norm_factor = cc_proportion / sum(FLN_EDR_completed[target_area].values())
        for source_area in FLN_EDR_completed[target_area]:
            FLN_completed[target_area][source_area] = (
                norm_factor * FLN_EDR_completed[target_area][source_area])

    """
    2. Process Binzegger data.
    The notation follows Eqs. (11-12 and following) in
    Schmidt et al. (2018):
    v : layer of the cortico-cortical synapse
    cb : cell type
    cell_layer : layer of the cell
    i : population in the model
    """
    layer_regex = r"\(.*\)"

    # Relative numbers of the 8 populations in Binzegger's data
    relative_numbers_binzegger = {pop: 0.0 for pop in population_list}
    s = 0.0
    for cb in binzegger_data:
        cell_layer = re.sub(r"\D", "", re.sub(layer_regex, "", cb))
        if cell_layer not in ['', '1']:
            s += binzegger_data[cb]['occurrence']

    for cb in binzegger_data:
        cell_layer = re.sub(r"\D", "", re.sub(layer_regex, "", cb))
        if cell_layer not in ['', '1']:
            if cb in binzegger_E_pops:
                relative_numbers_binzegger[cell_layer + 'E'] += (
                    binzegger_data[cb]['occurrence'] / s)
            if cb in binzegger_I_pops:
                relative_numbers_binzegger[cell_layer + 'I'] += (
                    binzegger_data[cb]['occurrence'] / s)

    # Relative numbers of the 8 populations in V1
    relative_numbers_model = {pop: 0.0 for pop in population_list}
    for pop in neuronal_numbers['V1']:
        relative_numbers_model[pop] = (neuronal_numbers['V1'][pop]
                                       / neuronal_numbers['V1']['total'])

    # Binzegger data as conditional probabilities: what is the probability
    # of a cell body in layer u if a cortico-cortical connection forms a
    # synapse in layer v?

    # Number of CC synapses formed in each layer
    num_cc_synapses = {'1': 0.0, '23': 0.0, '4': 0.0, '5': 0.0, '6': 0.0}
    for cb in binzegger_data:
        cell_layer = re.sub(r"\D", "", re.sub(layer_regex, "", cb))
        if cb in binzegger_E_pops:
            i = cell_layer + 'E'
        if cb in binzegger_I_pops:
            i = cell_layer + 'I'
        if i != '1I':
            for v in binzegger_data[cb]['syn_dict']:
                if v in num_cc_synapses:
                    num_ratio = (relative_numbers_model[i]
                                 / relative_numbers_binzegger[i])
                    cc_syn_num = (
                        binzegger_data[cb]['syn_dict'][v]['corticocortical'] / 100.0
                        * binzegger_data[cb]['syn_dict'][v][
                            'number of synapses per neuron']
                        * binzegger_data[cb]['occurrence'] / 100.0 * num_ratio)
                    num_cc_synapses[v] += cc_syn_num

    # Conditional probability
    synapse_to_cell_body_basis = {}
    for cb in binzegger_data:
        cell_layer = re.sub(r"\D", "", re.sub(layer_regex, "", cb))
        if cb in binzegger_E_pops:
            i = cell_layer + 'E'
        else:
            i = cell_layer + 'I'
        for v in binzegger_data[cb]['syn_dict']:
            if v in num_cc_synapses:
                if i != '1I':  # no cell types in layer 1
                    num_ratio = (relative_numbers_model[i]
                                 / relative_numbers_binzegger[i])
                    value = (
                        binzegger_data[cb]['syn_dict'][v]['corticocortical'] / 100.0
                        * binzegger_data[cb]['syn_dict'][v][
                            'number of synapses per neuron']
                        * binzegger_data[cb]['occurrence'] / 100.0 * num_ratio)
                    cond_prob = value / num_cc_synapses[v]
                    if v in synapse_to_cell_body_basis:
                        if i in synapse_to_cell_body_basis[v]:
                            synapse_to_cell_body_basis[v][i] += cond_prob
                        else:
                            synapse_to_cell_body_basis[v].update({i: cond_prob})
                    else:
                        synapse_to_cell_body_basis.update({v: {i: cond_prob}})

    # Make synapse_to_cell_body area-specific to account for
    # missing layers in some areas
    synapse_to_cell_body = {}
    for area in area_list:
        synapse_to_cell_body[area] = deepcopy(synapse_to_cell_body_basis)

    def num_CC_synapses(target_area, target_pop, source_area, source_pop):
        """
        Compute the number of cortico-cortical synapses from source_pop in
        source_area to target_pop in target_area. Laminar patterns from
        CoCoMac are not used.
        """
        Nsyn = 0.0

        # Test if the connection exists
        if (source_area in Binary_symmetric_connectivity[target_area] and
                source_pop not in ['4I', '4E'] and
                neuronal_numbers[target_area][target_pop] != 0 and
                source_pop not in ['23I', '4I', '5I', '6I']):

            num_source = neuronal_numbers_fullscale[source_area][source_pop]
            FLN_BA = FLN_completed[target_area][source_area]
            Nsyn_tot = (rho_syn[target_area] * Area_surfaces[target_area]
                        * total_thicknesses[target_area])

            # Source side: X is the fraction of the FLN from this source
            # (via SLN), Y the fraction contributed by this population
            if source_pop in origin_patterns['S']:  # supragranular (23E)
                X = SLN_Data[target_area][source_area]
                Y = 1.0
            elif source_pop in origin_patterns['I']:  # infragranular (5E, 6E)
                X = 1.0 - SLN_Data[target_area][source_area]
                infra_neurons = sum(neuronal_numbers_fullscale[source_area][pop]
                                    for pop in origin_patterns['I'])
                Y = num_source / infra_neurons
            else:
                return 0.0

            source_fraction = X * Y

            # Target side: termination layers depend on the SLN value
            if SLN_Data[target_area][source_area] > 0.65:
                T = termination_layers['F']  # feedforward: layer 4
            elif SLN_Data[target_area][source_area] < 0.35:
                T = termination_layers['M']  # feedback: layers 1, 23, 5, 6
            else:
                T = termination_layers['C']  # lateral: all layers

            # Total thickness of the termination layers (without layer 1)
            p_T = 0.0
            for layer in T:
                if layer != '1':
                    p_T += laminar_thicknesses[target_area][layer]

            Nsyn = 0.0
            for syn_layer in T:
                # Skip if the target population cannot receive synapses here
                if target_pop not in synapse_to_cell_body[target_area][syn_layer]:
                    continue

                # Fraction Z of the synapses that terminate in this layer
                if syn_layer == '1':
                    Z = 0.5
                else:
                    if '1' in T:
                        Z = 0.5 * laminar_thicknesses[target_area][syn_layer] / p_T
                    else:
                        Z = laminar_thicknesses[target_area][syn_layer] / p_T

                # Probability that a synapse in this layer lands on target_pop
                P_target = synapse_to_cell_body[target_area][syn_layer][target_pop]

                Nsyn += P_target * Nsyn_tot * FLN_BA * source_fraction * Z

        return Nsyn

    for target_area, target_pop, source_area, source_pop in product(
            area_list, population_list, area_list, population_list):
        if target_area != source_area:
            N_fullscale = neuronal_numbers_fullscale[target_area][target_pop]
            N = neuronal_numbers[target_area][target_pop]
            if N != 0:
                N_syn = num_CC_synapses(target_area, target_pop,
                                        source_area, source_pop) / N_fullscale * N
            else:
                N_syn = 0.0
            synapse_numbers[target_area][target_pop][source_area][source_pop] = N_syn

    """
    If E_specificity is True, redistribute the synapses of feedback
    connections to achieve the E_specific_factor of 0.93.
    """
    if E_specificity:
        E_specific_factor = 0.93
        for target_area in area_list:
            for source_area in area_list:
                if (target_area != source_area and
                        source_area in Binary_symmetric_connectivity[target_area] and
                        SLN_Data[target_area][source_area] < 0.35):
                    syn_I = 0.0
                    syn_E = 0.0
                    for target_pop in synapse_numbers[target_area]:
                        for source_pop in synapse_numbers[target_area][
                                target_pop][source_area]:
                            if target_pop.find('E') > -1:
                                syn_E += synapse_numbers[target_area][
                                    target_pop][source_area][source_pop]
                            else:
                                syn_I += synapse_numbers[target_area][
                                    target_pop][source_area][source_pop]
                    if syn_E > 0.0 or syn_I > 0.0:
                        alpha_E = syn_E / (syn_E + syn_I)
                        alpha_I = syn_I / (syn_E + syn_I)
                        if alpha_I != 0.0 and alpha_E != 0.0:
                            for target_pop in synapse_numbers[target_area]:
                                for source_pop in synapse_numbers[target_area][
                                        target_pop][source_area]:
                                    N_syn = synapse_numbers[target_area][
                                        target_pop][source_area][source_pop]
                                    if target_pop.find('E') > -1:
                                        synapse_numbers[target_area][target_pop][
                                            source_area][source_pop] = (
                                                E_specific_factor / alpha_E * N_syn)
                                    else:
                                        synapse_numbers[target_area][target_pop][
                                            source_area][source_pop] = (
                                                (1. - E_specific_factor)
                                                / alpha_I * N_syn)

    """
    External inputs
    ---------------
    To determine the number of external inputs to each population, we
    compute the total number of external synapses of an area and distribute
    them such that each population receives the same indegree from external
    Poisson sources.

    1. Total number of external synapses of each area: the total number of
       synapses minus the intrinsic (type I) and cortico-cortical
       (type III) synapses.
    """
    External_synapses = {}
    for target_area in area_list:
        N_syn_tot = surface * total_thicknesses[target_area] * rho_syn[target_area]
        CC_synapses = 0.0
        for target_pop, source_area, source_pop in product(
                population_list, area_list, population_list):
            if source_area != target_area:
                CC_synapses += synapse_numbers[target_area][target_pop][
                    source_area][source_pop]

        ext_syn = (N_syn_tot * (1. - Intrinsic_FLN_completed[target_area]['mean'])
                   - CC_synapses)
        External_synapses[target_area] = ext_syn

    """
    2. Distribute the Poisson sources among the populations such that each
       population receives the same Poisson indegree. This is a system of
       linear equations, solved with a least-squares algorithm
       (numpy.linalg.lstsq).
    """
    for area in area_list:
        nonvisual_fraction_matrix = np.zeros(
            (len(structure[area]) + 1, len(structure[area])))
        for i in range(len(structure[area])):
            nonvisual_fraction_matrix[i] = (
                1. / len(structure[area]) * np.ones(len(structure[area])))
            nonvisual_fraction_matrix[i][i] -= 1

        for i in range(len(structure[area])):
            nonvisual_fraction_matrix[-1][i] = neuronal_numbers[area][
                structure[area][i]]

        vector = np.zeros(len(structure[area]) + 1)
        vector[-1] = External_synapses[area]

        solution, residues, rank, sv = np.linalg.lstsq(
            nonvisual_fraction_matrix, vector, rcond=-1)
        for i, pop in enumerate(structure[area]):
            synapse_numbers[area][pop]['external'] = {
                'external': solution[i] * neuronal_numbers[area][pop]}

    """
    Output section
    --------------
    All data are saved to a json file named
    'visuomotor_Data_Model<out_label>.json'.
    """
    collected_data = {'area_list': output_area_list,
                      'av_indegree_V1': av_indegree_V1,
                      'population_list': population_list,
                      'structure': structure,
                      'synapses_orig': synapse_numbers,
                      'synapses': synapse_numbers,
                      'realistic_neuron_numbers': neuronal_numbers_fullscale,
                      'realistic_synapses': synapse_numbers,
                      'neuron_numbers': neuronal_numbers,
                      'synapses_type_I': synapses_type_I,
                      'synapses_type_II': synapses_type_II,
                      'distances': Distance_Data,
                      'binzegger_processed': synapse_to_cell_body,
                      'Intrinsic_FLN_completed': Intrinsic_FLN_completed}

    with open(os.path.join(basepath,
                           ''.join(('visuomotor_Data_Model', out_label, '.json'))),
              'w') as f:
        json.dump(collected_data, f, indent=3)

    return collected_data


def compute_connection_probabilities(synapses, neuron_numbers, area_list,
                                     population_list, use_approximation=False):
    """
    Compute the connection probabilities C_{iA, jB} for the model surface.

    Parameters
    ----------
    synapses : dict
        synapses[target_area][target_pop][source_area][source_pop]
        (type I and type III synapses, scaled to the model surface).
    neuron_numbers : dict
        neuron_numbers[area][population] for the model surface.
    area_list : list
        List of areas.
    population_list : list
        List of populations.
    use_approximation : bool
        If True, use C = N_syn / (N_i * N_j), which is accurate for small
        probabilities. Defaults to False.

    Returns
    -------
    C : dict
        C[target_area][target_pop][source_area][source_pop]
    """
    C = {}

    for target_area in area_list:
        C[target_area] = {}

        for target_pop in population_list:
            C[target_area][target_pop] = {}

            N_i = neuron_numbers[target_area].get(target_pop, 0)

            for source_area in area_list:
                C[target_area][target_pop][source_area] = {}

                for source_pop in population_list:
                    N_j = neuron_numbers[source_area].get(source_pop, 0)
                    N_syn = synapses[target_area][target_pop][source_area][
                        source_pop]

                    if N_i == 0 or N_j == 0 or N_syn == 0:
                        C_val = 0.0
                    elif use_approximation:
                        C_val = round(N_syn / (N_i * N_j), 40)
                    else:
                        C_val = round(1.0 - np.exp(
                            N_syn * np.log(1.0 - 1.0 / (N_i * N_j))), 40)
                            

                    C[target_area][target_pop][source_area][source_pop] = C_val

    return C


def compute_indegrees(synapses, neuron_numbers, area_list, population_list):
    """
    Compute the indegree of each connection (number of synapses divided by
    the size of the target population), including the external input.

    Returns
    -------
    indegrees : dict
        indegrees[target_area][target_pop][source_area][source_pop] and
        indegrees[target_area][target_pop]['external']['external']
    """
    indegrees = nested_dict()
    for target_area, target_pop, source_area, source_pop in product(
            area_list, population_list, area_list, population_list):
        numT = neuron_numbers[target_area][target_pop]
        if numT > 0.0:
            indegrees[target_area][target_pop][source_area][source_pop] = (
                synapses[target_area][target_pop][source_area][source_pop] / numT)
        else:
            # indegrees onto non-existing populations are 0
            indegrees[target_area][target_pop][source_area][source_pop] = 0.0

    for target_area, target_pop in product(area_list, population_list):
        numT = neuron_numbers[target_area][target_pop]
        if numT > 0.0:
            indegrees[target_area][target_pop]['external']['external'] = (
                synapses[target_area][target_pop]['external']['external'] / numT)
        else:
            indegrees[target_area][target_pop]['external']['external'] = 0.0

    return indegrees.to_dict()


def compute_connectivity_files(collected_data=None, out_label=''):
    """
    Compute connection probabilities and indegrees from the model data and
    write them to json files.

    Parameters
    ----------
    collected_data : dict, optional
        Output of compute_Model_params. If None, the data is loaded from
        visuomotor_Data_Model<out_label>.json.
    out_label : str
        Label that is appended to the output file names.
    """
    basepath = os.path.abspath(os.path.dirname(__file__))

    if collected_data is None:
        with open(os.path.join(basepath, ''.join(
                ('visuomotor_Data_Model', out_label, '.json'))), 'r') as f:
            collected_data = json.load(f)

    syn = collected_data['synapses']
    num = collected_data['neuron_numbers']
    complete_area_list = collected_data['area_list']
    population_list = collected_data['population_list']

    conn_prob = compute_connection_probabilities(
        syn, num, complete_area_list, population_list,
        use_approximation=False)

    with open(os.path.join(basepath, ''.join(
            ('visuomotor_connection_prob_data', out_label, '.json'))), 'w') as f:
        json.dump(conn_prob, f, indent=3)

    indegrees = compute_indegrees(syn, num, complete_area_list,
                                  population_list)

    with open(os.path.join(basepath, ''.join(
            ('visuomotor_indegree_data', out_label, '.json'))), 'w') as f:
        json.dump(indegrees, f, indent=3)


if __name__ == '__main__':
    model_data = compute_Model_params()
    compute_connectivity_files(model_data)
