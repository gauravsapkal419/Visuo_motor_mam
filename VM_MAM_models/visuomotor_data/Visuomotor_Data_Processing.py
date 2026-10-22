"""
VisuomotorDataProcessing
========================

This script provides the function process_data, which
1) loads the raw data stored in visuomotor_raw_data.json (created by
Visuomotor_Data.py) plus two laminar thickness files from raw_data/,
2) derives complete sets of neuronal densities, laminar thicknesses,
neuron numbers, FLN and SLN for the visuomotor areas, and
3) stores the results in visuomotor_processed_data.json.

Steps
-----
1. Neuronal densities: linear fit of Visual MAM vs. Beul densities,
   mapping to 4 layers, structural-type averages for missing areas.
2. Total and laminar thicknesses: log-density vs. thickness fit,
   relative layer thicknesses (Visual MAM, F1 for motor areas),
   L4 fraction from a log-density regression.
3. Neuron numbers per population (E/I ratio from cat V1).
4. FLN: exponential distance rule (EDR) fitted on the Markov et al. data.
5. SLN: sigmoidal fit vs. log density ratio (R script, hard-coded
   fallback if R is not available).
6. Binary symmetric connectivity, filling of missing FLN and SLN values.

Requires visuomotor_raw_data.json in the script folder and, in raw_data/:
vmam_laminar_thickness.json and F1_thickness_data.json.
"""
import copy
import json
import os
import subprocess

import numpy as np

from itertools import product
from nested_dict import nested_dict
from scipy import integrate
from scipy import stats


def process_data():
    """
    Process the raw visuomotor data and store the result as json.
    """

    """
    Set input and output paths
    """
    basepath = os.path.abspath(os.path.dirname(__file__))
    datapath = os.path.join(basepath, 'raw_data')
    out_label = ''
    out_path = basepath

    motor_areas = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
    layers = ['1', '23', '4', '5', '6']

    """
    Load raw data
    """
    with open(os.path.join(out_path,
                           'visuomotor_raw_data' + out_label + '.json'), 'r') as f:
        raw_data = json.load(f)

    area_list = raw_data['area_list']
    architecture_completed = raw_data['architecture_completed']
    surface_data = raw_data['surface_data']
    thom_distance_data_markov = raw_data['thom_distance_data_markov']
    FLN_Data = raw_data['FLN_Data']
    SLN_Data = raw_data['SLN_Data']
    num_V1 = raw_data['num_V1']
    total_thickness = raw_data['total_thickness_data_beul']
    Beul_neuronal_densities = raw_data['neuronal_density_data_beul']
    visual_MAM_neuronal_densities = raw_data['visual_MAM_neuronal_densities']
    raw_neuronal_numbers = raw_data['raw_neuronal_numbers']

    with open(os.path.join(datapath, 'vmam_laminar_thickness.json'), 'r') as f:
        visual_MAM_laminar_thicknesses = json.load(f)

    with open(os.path.join(datapath, 'F1_thickness_data.json'), 'r') as f:
        F1_laminar_fraction = json.load(f)

    """
    Neuronal densities
    ------------------

    1. Linear fit of Visual MAM vs. Beul overall neuronal densities
    """
    neuron_densities_vmam_val = []
    neuron_densities_beul_val = []

    for area in Beul_neuronal_densities:
        if area in visual_MAM_neuronal_densities:
            if np.isfinite(Beul_neuronal_densities[area]):
                neuron_densities_vmam_val.append(
                    visual_MAM_neuronal_densities[area]['overall'])
                neuron_densities_beul_val.append(Beul_neuronal_densities[area])

    gradient, intercept, r_value, p_value, std_err = stats.linregress(
        neuron_densities_vmam_val, neuron_densities_beul_val)

    """
    2. Map neuronal density data to 4 layers.
    Layer-specific densities are obtained by scaling the Visual MAM values.
    """
    neuronal_density_data_4layers = {}

    for area in Beul_neuronal_densities:
        if area in visual_MAM_neuronal_densities or area == '46v':
            neuronal_density_data_4layers.update(
                {area: {'23': 0., 'overall': 0.,
                        '4': {'value': 0.0, 'error': 0.0},
                        '5': 0., '6': 0.}})

    for i in list(neuronal_density_data_4layers.keys()):
        FV91_key = i
        if i == '46v':
            FV91_key = '46'  # translation from FV91 to M132

        for layer in ['23', '4', '5', '6']:
            if visual_MAM_neuronal_densities[FV91_key][layer] > 0.:
                # Layers 5 and 6 are scaled separately
                if layer in ['5', '6']:
                    neuronal_density_data_4layers[i]['5'] = (
                        visual_MAM_neuronal_densities[FV91_key]['5']
                        * gradient + intercept)
                    neuronal_density_data_4layers[i]['6'] = (
                        visual_MAM_neuronal_densities[FV91_key]['6']
                        * gradient + intercept)
                else:
                    neuronal_density_data_4layers[i][layer] = (
                        visual_MAM_neuronal_densities[FV91_key][layer]
                        * gradient + intercept)
            else:
                neuronal_density_data_4layers[i][layer] = 0.
        # Beul overall density if available, otherwise scaled Visual MAM value
        if i in Beul_neuronal_densities:
            if np.isfinite(Beul_neuronal_densities[i]):
                neuronal_density_data_4layers[i]['overall'] = \
                    Beul_neuronal_densities[i]
            else:
                neuronal_density_data_4layers[i]['overall'] = (
                    visual_MAM_neuronal_densities[FV91_key]['overall']
                    * gradient + intercept)

    """
    We build a dictionary containing neuron densities (overall and
    layer-specific) for each area. If there is no data for an area, we
    compute the densities in two steps:

    1. Assign an average neuronal density to each structural type, for
       each layer and for the overall density.
    2. Based on the structural type, assign a density to each area
       without direct data.
    """
    # Step 1: average density per structural type
    neuronal_density_list = [{'overall': [], '23': [], '4': [], '5': [],
                              '6': []} for arch_type in range(8)]

    for area in list(neuronal_density_data_4layers.keys()):
        category = architecture_completed[area]
        for key in list(neuronal_density_data_4layers[area].keys()):
            neuronal_density_list[category - 1][key].append(
                float(neuronal_density_data_4layers[area][key]))

    category_density = {}
    for arch_type in range(8):
        dict_ = {}
        for layer_key in list(neuronal_density_list[0].keys()):
            if len(neuronal_density_list[arch_type][layer_key]) == 0:
                dict_[layer_key] = np.nan
            else:
                dict_[layer_key] = np.mean(
                    neuronal_density_list[arch_type][layer_key])
        category_density[arch_type + 1] = dict_

    # Step 2: structural-type average for areas without direct data
    neuronal_densities = nested_dict()
    for area in list(architecture_completed.keys()):
        if architecture_completed[area] in range(1, 9, 1):
            if area in list(neuronal_density_data_4layers.keys()):
                for key in list(neuronal_density_data_4layers[area].keys()):
                    neuronal_densities[area][key] = \
                        neuronal_density_data_4layers[area][key]
            else:
                neuronal_densities[area] = \
                    category_density[architecture_completed[area]]
        else:
            neuronal_densities[area] = '?'
    neuronal_densities = neuronal_densities.to_dict()

    # Overall neuronal densities of the motor areas (Beul et al.)
    neuronal_densities.update(
        {area: {'overall': Beul_neuronal_densities[area]}
         for area in motor_areas})

    """
    Thicknesses
    -----------

    Total thicknesses: linear fit of thickness vs. logarithmic overall
    neuronal density.
    """
    log_densities_dict = {}
    thickness_dict = {}

    for area in area_list:
        if np.isfinite(neuronal_densities[area]['overall']):
            log_densities_dict[area] = np.log10(
                neuronal_densities[area]['overall'])
            if area in total_thickness:
                thickness_dict[area] = total_thickness[area]

    slope, intercept_th, r_value, p_value, std_err = stats.linregress(
        np.array(list(log_densities_dict.values())),
        np.array(list(thickness_dict.values())))

    """
    Laminar thicknesses

    1. Relative layer thicknesses for areas in the Visual MAM
       (TH-TF is taken from TH).
    2. Motor areas take the laminar fractions of F1.
    """
    relative_layer_thicknesses = nested_dict()
    for area, layer in product(area_list, layers):
        if area in visual_MAM_laminar_thicknesses:
            relative_layer_thicknesses[area][layer] = (
                visual_MAM_laminar_thicknesses[area][layer]
                / sum(visual_MAM_laminar_thicknesses[area].values()))
        if area == "TH-TF":
            relative_layer_thicknesses[area][layer] = (
                visual_MAM_laminar_thicknesses["TH"][layer]
                / sum(visual_MAM_laminar_thicknesses["TH"].values()))

    relative_layer_thicknesses = relative_layer_thicknesses.to_dict()

    relative_layer_thicknesses.update(
        {area: F1_laminar_fraction["laminar_fractions"].copy()
         for area in motor_areas if area in total_thickness})

    # For areas without data: mean across areas of the fractions of the
    # total thickness occupied by L1, L2/3, L5 and L6
    tmp1 = np.array([])
    tmp23 = np.array([])
    tmp5 = np.array([])
    tmp6 = np.array([])
    for area in list(relative_layer_thicknesses.keys()):
        tmp1 = np.append(tmp1, relative_layer_thicknesses[area]['1'])
        tmp23 = np.append(tmp23, relative_layer_thicknesses[area]['23'])
        tmp5 = np.append(tmp5, relative_layer_thicknesses[area]['5'])
        tmp6 = np.append(tmp6, relative_layer_thicknesses[area]['6'])

    mean_rel_L1_thickness = np.mean(tmp1[np.isfinite(tmp1)])
    mean_rel_L23_thickness = np.mean(tmp23[np.isfinite(tmp23)])
    mean_rel_L5_thickness = np.mean(tmp5[np.isfinite(tmp5)])
    mean_rel_L6_thickness = np.mean(tmp6[np.isfinite(tmp6)])

    updated_relative_layer_thicknesses = nested_dict()
    updated_relative_layer_thicknesses.update(relative_layer_thicknesses)
    for area in area_list:
        if area not in list(relative_layer_thicknesses.keys()):
            updated_relative_layer_thicknesses.update(
                {area: {'1': [], '23': [], '4': [], '5': [], '6': []}})
            updated_relative_layer_thicknesses[area]['1'] = \
                mean_rel_L1_thickness
            updated_relative_layer_thicknesses[area]['23'] = \
                mean_rel_L23_thickness
            updated_relative_layer_thicknesses[area]['4'] = np.nan
            updated_relative_layer_thicknesses[area]['5'] = \
                mean_rel_L5_thickness
            updated_relative_layer_thicknesses[area]['6'] = \
                mean_rel_L6_thickness

    updated_relative_layer_thicknesses = \
        updated_relative_layer_thicknesses.to_dict()

    # L4: regression of known relative L4 thickness vs. log overall density
    frac4_of_total = []
    log_densities = []

    for area in log_densities_dict.keys():
        if area in relative_layer_thicknesses:
            frac4_of_total.append(relative_layer_thicknesses[area]['4'])
            log_densities.append(log_densities_dict[area])

    gradient4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(
        np.array(log_densities), np.array(frac4_of_total))

    for area in list(updated_relative_layer_thicknesses.keys()):
        if np.isnan(updated_relative_layer_thicknesses[area]['4']):
            updated_relative_layer_thicknesses[area]['4'] = (
                intercept4 + gradient4 * np.log(
                    neuronal_densities[area]['overall']))

    # Convert relative thicknesses into absolute ones (mm). The result is
    # normalized so that the relative thicknesses sum to 1.
    laminar_thickness_completed = {}
    for area in list(updated_relative_layer_thicknesses.keys()):
        laminar_thickness_completed[area] = {}
        sum_rel_thick = sum(updated_relative_layer_thicknesses[area].values())
        for layer in list(updated_relative_layer_thicknesses[area].keys()):
            laminar_thickness_completed[area][layer] = (
                updated_relative_layer_thicknesses[area][layer]
                * total_thickness[area] / sum_rel_thick)

    """
    Update motor-area neuronal densities with the densities calculated
    from the neuron numbers, surface areas and laminar thicknesses.
    """
    for area in motor_areas:
        nd = {'overall': Beul_neuronal_densities[area]}
        for layer in ['23', '4', '5', '6']:
            nd[layer] = ((raw_neuronal_numbers[area][layer + 'E']
                          + raw_neuronal_numbers[area][layer + 'I'])
                         / (laminar_thickness_completed[area][layer]
                            * surface_data[area]))
        neuronal_densities.update({area: nd})

    """
    Neuron numbers
    --------------
    We assume a laminar-specific ratio of excitatory to inhibitory
    neurons that is constant across areas.
    """
    EI_ratio = {layer: num_V1[layer + 'E']['neurons']
                / (num_V1[layer + 'E']['neurons']
                   + num_V1[layer + 'I']['neurons'])
                for layer in ['23', '4', '5', '6']}

    """
    The number of neurons in population i in layer v_i of area A is
    N(A,i) = rho(A,v_i) * S(A) * D(A,v_i) * EI_ratio.
    """
    realistic_neuronal_numbers = nested_dict()
    for area in list(neuronal_densities.keys()):
        S = surface_data[area]
        ltc = laminar_thickness_completed[area]
        nd = neuronal_densities[area]
        for layer in ['23', '4', '5', '6']:
            realistic_neuronal_numbers[area][layer + 'E'] = (
                S * ltc[layer] * nd[layer] * EI_ratio[layer])
            realistic_neuronal_numbers[area][layer + 'I'] = (
                realistic_neuronal_numbers[area][layer + 'E']
                * (1. - EI_ratio[layer]) / EI_ratio[layer])
        realistic_neuronal_numbers[area]['total'] = sum(
            realistic_neuronal_numbers[area].values())
    realistic_neuronal_numbers = realistic_neuronal_numbers.to_dict()

    """
    Cortico-cortical connectivity
    -----------------------------

    FLN values

    1. Process the raw FLN data into the dictionary FLN_Data_vm.
    2. Fit the exponential distance rule (EDR) to the existing data.
    """
    FLN_Data_vm = {}

    for monkey in FLN_Data:
        vm_area = FLN_Data[monkey]['target_area']

        if vm_area in area_list:
            if vm_area in FLN_Data_vm:
                for source in FLN_Data[monkey]['source_areas']:
                    fln = FLN_Data[monkey]['source_areas'][source]['FLN']
                    if source in FLN_Data_vm[vm_area]:
                        FLN_Data_vm[vm_area][source] = np.append(
                            FLN_Data_vm[vm_area][source], fln)
                    else:
                        FLN_Data_vm[vm_area][source] = np.array([fln])
            else:
                FLN_Data_vm[vm_area] = {}
                for source in FLN_Data[monkey]['source_areas']:
                    FLN_Data_vm[vm_area][source] = np.array(
                        [FLN_Data[monkey]['source_areas'][source]['FLN']])
                    
    # Replace nan values with 0
    for target_area in FLN_Data_vm:
        for source_area in FLN_Data_vm[target_area]:
            arr = FLN_Data_vm[target_area][source_area]
            FLN_Data_vm[target_area][source_area] = np.nan_to_num(arr, nan=np.nan)


    # Arithmetic means; arrays are padded with zeros if necessary
    for target in FLN_Data_vm:
        dimension = 0
        for source in FLN_Data_vm[target]:
            if FLN_Data_vm[target][source].size > dimension:
                dimension = FLN_Data_vm[target][source].size
        for source in FLN_Data_vm[target]:
            array = np.append(FLN_Data_vm[target][source], np.zeros(
                dimension - FLN_Data_vm[target][source].size))
            FLN_Data_vm[target][source] = np.mean(array)

    # Remove intrinsic FLN
    for target in FLN_Data_vm:
        if target in FLN_Data_vm[target]:
            del FLN_Data_vm[target][target]

    # Add target areas without data
    for a in area_list:
        if a not in FLN_Data_vm:
            FLN_Data_vm.update({a: {}})

    # Exponential distance rule
    FLN_values_vm = []
    distances_vm = []

    for target_area in FLN_Data_vm:
        for source_area in FLN_Data_vm[target_area]:

            # Target and source area must exist in the distance matrix
            if (target_area in thom_distance_data_markov and
                    source_area in thom_distance_data_markov[target_area]):

                fln_val = FLN_Data_vm[target_area][source_area]
                dist_val = thom_distance_data_markov[target_area][source_area]

                # Remove NaN, zeros and invalid values
                if (fln_val is not None and not np.isnan(fln_val) and
                        fln_val > 0 and
                        dist_val is not None and not np.isnan(dist_val) and
                        dist_val > 0):
                    FLN_values_vm.append(fln_val)
                    distances_vm.append(dist_val)

    FLN_values_vm = np.array(FLN_values_vm, dtype=float)
    distances_vm = np.array(distances_vm, dtype=float)

    # Linear fit to log values
    gradient_edr, intercept_edr, r_value, p_value, std_err = stats.linregress(
        distances_vm, np.log(FLN_values_vm))
    EDR_params = [intercept_edr, gradient_edr]

    def EDR(target_area, source_area):
        return np.exp(EDR_params[0] + EDR_params[1] *
                      thom_distance_data_markov[target_area][source_area])

    """
    SLN values

    1. Process the raw SLN data into the dictionary SLN_Data_vm.
    2. Prepare SLN and log-density data for curve fitting.
    3. Sigmoidal fit of SLN vs. log neuronal density ratio (R).
    4. Fill missing data with fitted values.
    """
    SLN_Data_vm = {}
    for monkey in SLN_Data:
        vm_area = SLN_Data[monkey]['target_area']

        if vm_area in area_list:
            if vm_area in SLN_Data_vm:
                for source in SLN_Data[monkey]['source_areas']:
                    sln = SLN_Data[monkey]['source_areas'][source]['SLN']
                    if source in SLN_Data_vm[vm_area]:
                        SLN_Data_vm[vm_area][source] = np.append(
                            SLN_Data_vm[vm_area][source], sln)
                    else:
                        SLN_Data_vm[vm_area][source] = np.array([sln])
            else:
                SLN_Data_vm[vm_area] = {}
                for source in SLN_Data[monkey]['source_areas']:
                    SLN_Data_vm[vm_area][source] = np.array(
                        [SLN_Data[monkey]['source_areas'][source]['SLN']])

    for target in SLN_Data_vm:
        dimension = 0
        for source in SLN_Data_vm[target]:
            if SLN_Data_vm[target][source].size > dimension:
                dimension = SLN_Data_vm[target][source].size
        for source in SLN_Data_vm[target]:
            array = np.append(SLN_Data_vm[target][source], np.zeros(
                dimension - SLN_Data_vm[target][source].size))
            SLN_Data_vm[target][source] = np.mean(array)

    # Data file for the R fit
    rows = ["\t".join(["Monkey", "TO", "FROM", "S", "I", "DENS"])]

    for monkey, mdata in SLN_Data.items():
        target = mdata["target_area"]

        if target not in neuronal_densities:
            continue

        for source, vals in mdata["source_areas"].items():
            S = vals["S"]
            I = vals["I"]

            if source == target:
                continue
            if np.isnan(S) or np.isnan(I):
                continue
            if S + I == 0:
                continue
            if source not in neuronal_densities:
                continue

            dens = np.log(neuronal_densities[target]['overall'] /
                          neuronal_densities[source]['overall'])

            rows.append("\t".join([monkey, target, source,
                                   str(int(S)), str(int(I)), f"{dens:.8f}"]))

    with open(os.path.join(datapath, "SLN_logdensities_data.txt"), "w") as f:
        f.write("\n".join(rows))

    # Sigmoidal fit
    def integrand(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(
            -(x - mu) ** 2 / (2. * sigma ** 2))

    def probit(x):
        if isinstance(x, np.ndarray):
            res = [integrate.quad(integrand, -1000., i, args=(0., 1.))[0]
                   for i in x]
        else:
            res = integrate.quad(integrand, -1000., x, args=(0., 1.))[0]
        return res

    # Call R script to perform the SLN fit
    try:
        proc = subprocess.Popen(["Rscript",
                                 os.path.join(basepath, 'SLN_logdensities.R'),
                                 basepath],
                                stdout=subprocess.PIPE)
        out = proc.communicate()[0].decode('utf-8')
        R_fit = [float(out.split('\n')[1].split(' ')[1]),
                 float(out.split('\n')[1].split(' ')[3])]
    except (OSError, IndexError, ValueError):
        print("No R installation or IndexError/ValueError, "
              "taking hard-coded SLN fit parameters.")
        R_fit = [-0.2053786, -0.9090635]

    """
    Fill in missing FLN and SLN data
    --------------------------------

    1. Binary symmetric connectivity from FLN and SLN data.
    2. Fill missing FLN values with the EDR.
    3. Fill missing SLN values with the sigmoidal fit.
    """
    Bin_sym_conn_data = {area: {} for area in area_list}

    def is_valid(data, target, source):
        """Check if a connection is present and not NaN."""
        return (target in data and
                source in data[target] and
                not np.isnan(data[target][source]))

    for target in area_list:
        for source in area_list:

            if target == source:
                continue

            # connection exists in either dataset, in either direction
            exists = (is_valid(FLN_Data_vm, target, source) or
                      is_valid(FLN_Data_vm, source, target) or
                      is_valid(SLN_Data_vm, target, source) or
                      is_valid(SLN_Data_vm, source, target))

            if exists:
                Bin_sym_conn_data[target][source] = 1
                Bin_sym_conn_data[source][target] = 1

    # FLN: copy existing values, then fill missing connections with the EDR
    FLN_completed = {area: {} for area in area_list}

    for target_area in area_list:
        if target_area in FLN_Data_vm:
            FLN_completed[target_area] = copy.deepcopy(FLN_Data_vm[target_area])

    for target_area in area_list:
        for source_area in area_list:

            if source_area == target_area:
                continue
            if source_area in FLN_completed[target_area]:
                continue

            if (source_area in Bin_sym_conn_data[target_area] and
                    source_area in thom_distance_data_markov):
                FLN_completed[target_area][source_area] = EDR(
                    target_area, source_area)

    # Sum FLN from areas outside area_list into 'external'
    for target_area in FLN_completed:
        ext_FLN = 0.
        sources = list(FLN_completed[target_area].keys())
        for source_area in sources:
            if source_area not in area_list:
                ext_FLN += FLN_completed[target_area][source_area]
                del FLN_completed[target_area][source_area]
        FLN_completed[target_area]['external'] = ext_FLN

    # SLN: measured values where available, otherwise the sigmoidal fit
    SLN_completed = {}
    for target in area_list:
        SLN_completed[target] = {}
        for source in list(Bin_sym_conn_data[target].keys()):
            if (target in SLN_Data_vm and source in SLN_Data_vm[target] and
                    not np.isnan(SLN_Data_vm[target][source])):
                value = SLN_Data_vm[target][source]
            else:
                nd_target = neuronal_densities[target]['overall']
                nd_source = neuronal_densities[source]['overall']
                x = (R_fit[1] * (np.log(nd_target) - np.log(nd_source)) +
                     R_fit[0])
                value = probit(x)
            SLN_completed[target][source] = value

    """
    Write output files
    ------------------

    Store processed values to json file.
    """
    processed_data = {'bin_sym_conn_data': Bin_sym_conn_data,
                      'architecture_completed': architecture_completed,
                      'FLN_Data_vm' : FLN_Data_vm,
                      'SLN_Data_vm' : SLN_Data_vm,
                      'SLN_completed': SLN_completed,
                      'FLN_completed': FLN_completed,
                      'total_neuronal_densities': neuronal_densities,
                      'realistic_neuronal_numbers': realistic_neuronal_numbers,
                      'total_thicknesses': total_thickness,
                      'laminar_thicknesses': laminar_thickness_completed,
                      'category_density': category_density}

    with open(os.path.join(out_path,
                           ''.join(('visuomotor_processed_data', out_label,
                                    '.json'))),
              'w') as f:
        json.dump(processed_data, f)


if __name__ == '__main__':
    process_data()
