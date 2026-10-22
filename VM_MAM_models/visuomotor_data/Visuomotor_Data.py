"""
VisuomotorData
==============

This script provides the function process_raw_data, which loads the
experimental data from the raw data files stored in raw_data/ and stores
it to visuomotor_raw_data.json.

Input files (in raw_data/):
- m132_neuron_numbers_visuomotor_data.json
- M132_visual_areas_architectural_type.json
- midcortical_surface_areas_F99.json
- macaque_structural_data_Beul_Hilgetag_Barbas_Wagstyl.csv
- vmam_laminar_thickness.json, F1_thickness_data.json
- VMAM_neuronal_densities.json
- overlap_data.json
- Markov2014_FLN_rawdata.csv, SLN_Data.csv
- macaque_injection_data_Markov_Kennedy.csv
- Thom_Distances_MERetal12.csv
- WM_distances_MoralesGregorio_in_array_form.csv
- Intrinsic_FLN_Data.csv, Intrinsic_Connectivity.csv, Numbers_V1.csv
- BinzeggerData.csv
- SynapticDensity_Cragg.csv, SynapticDensity_OKusky.csv
"""
import csv
import json
import os
import re

import numpy as np
import pandas as pd


def process_raw_data():
    """
    Load raw data from the files in raw_data/ and store it in
    visuomotor_raw_data.json.
    """

    """
    Helper variables and functions
    """
    # Extracted visuomotor areas
    area_list = ["V1", "V2", "V3", "V4", "V3A", "DP", "V6", "MT", "V4t",
                 "PIP", "TEO", "V6A", "TEOm", "8m", "8l", "MIP", "F1", "45B",
                 "STPc", "LIP", "PBc", "8r", "MST", "F3", "7A", "F5", "F4",
                 "46d", "9-46v", "VIP", "9-46d", "TEpd", "46v", "STPi",
                 "F2", "F7", "PGa", "F6", "TEpv", "STPr", "IPa", "TEad",
                 "TH-TF"]

    # Area names that differ between area_list and the M132 json files
    key_map = {'TH-TF': 'TH_TF', '9-46v': '9_46v', '9-46d': '9_46d'}

    # to skip the explanatory headers in the .csv-files
    def skip_header(reader):
        next(reader)
        next(reader)

    """
    Set input and output paths
    """
    basepath = os.path.abspath(os.path.dirname(__file__))
    datapath = os.path.join(basepath, 'raw_data')
    out_label = ''
    out_path = basepath

    """
    1. Processed realistic neuron number data (M132)
    """
    with open(os.path.join(datapath,
                           'm132_neuron_numbers_visuomotor_data.json'), 'r') as f:
        neuronal_numbers = json.load(f)

    raw_neuronal_numbers = {}
    for area in area_list:
        raw_neuronal_numbers[area] = neuronal_numbers[key_map.get(area, area)]

    """
    2. Architectural types
    """
    with open(os.path.join(datapath,
                           'M132_visual_areas_architectural_type.json'), 'r') as f:
        architecture_completed = json.load(f)

    """
    3. Surface areas (F99)
    """
    with open(os.path.join(datapath,
                           'midcortical_surface_areas_F99.json'), 'r') as f:
        F99_surface_data = json.load(f)

    surface_data = {}
    for area in area_list:
        surface_data[area] = F99_surface_data[key_map.get(area, area) + '_M132']

    """
    4. Total thickness and neuronal density data from Beul et al.
    """
    beul = pd.read_csv(os.path.join(
        datapath, 'macaque_structural_data_Beul_Hilgetag_Barbas_Wagstyl.csv'))

    # Keep only the areas of interest
    filtered = beul[beul['Area'].isin(area_list)]

    total_thickness = dict(zip(filtered['Area'], filtered['Thickness (mm)']))
    Beul_neuronal_densities = dict(
        zip(filtered['Area'], filtered['Neuron density (neuron/mm3)']))

    """
    5. Laminar thickness data from the Visual MAM and F1 laminar fractions
    """
    with open(os.path.join(datapath, 'vmam_laminar_thickness.json'), 'r') as f:
        visual_MAM_laminar_thicknesses = json.load(f)

    with open(os.path.join(datapath, 'F1_thickness_data.json'), 'r') as f:
        F1_laminar_fraction = json.load(f)

    """
    6. Layer-specific and overall neuronal densities from the Visual MAM
    """
    with open(os.path.join(datapath, 'VMAM_neuronal_densities.json'), 'r') as f:
        visual_MAM_neuronal_densities = json.load(f)

    """
    7. Overlap data
    """
    with open(os.path.join(datapath, 'overlap_data.json'), 'r') as f:
        overlap = json.load(f)

    """
    8. FLN data
    a. Markov et al. (2014), taken from the Visual MAM GitHub page
    b. Markov and Kennedy data provided by Aitor
    """
    temp = pd.read_csv(os.path.join(datapath, 'Markov2014_FLN_rawdata.csv'),
                       sep='\t', skiprows=2,
                       usecols=['case', 'monkey', 'source_area', 'target_area',
                                'FLN', 'NLN', 'status'],
                       names=['case', 'monkey', 'source_area', 'target_area',
                              'FLN', 'NLN', 'status'])
    FLN_Data_markov = {}
    for i in range(len(temp)):
        monkey = temp.iloc[i]['monkey']
        target = temp.iloc[i]['target_area']
        source = temp.iloc[i]['source_area']
        FLN = float(temp.iloc[i]['FLN'])
        status = temp.iloc[i]['status']

        if monkey in FLN_Data_markov:
            FLN_Data_markov[monkey]['source_areas'].update(
                {source: {'FLN': FLN, 'status': status}})
        else:
            FLN_Data_markov[monkey] = {'target_area': target, 'source_areas': {
                source: {'FLN': FLN, 'status': status}}}

    temp_mk = pd.read_csv(os.path.join(
        datapath, 'macaque_injection_data_Markov_Kennedy.csv'))

    FLN_Data_aitor = {}
    for i in range(len(temp_mk)):
        monkey = temp_mk.iloc[i]['Monkey']
        target = temp_mk.iloc[i]['Target']
        source = temp_mk.iloc[i]['Source']
        FLN = float(temp_mk.iloc[i]['FLN'])

        if monkey in FLN_Data_aitor:
            FLN_Data_aitor[monkey]['source_areas'].update(
                {source: {'FLN': FLN}})
        else:
            FLN_Data_aitor[monkey] = {'target_area': target, 'source_areas': {
                source: {'FLN': FLN}}}

    """
    9. SLN data
    """
    # SLN data provided by Aitor
    SLN_Data_aitor = {}
    for i in range(len(temp_mk)):
        monkey = temp_mk.iloc[i]['Monkey']
        target = temp_mk.iloc[i]['Target']
        source = temp_mk.iloc[i]['Source']
        FLN = float(temp_mk.iloc[i]['FLN'])
        S = temp_mk.iloc[i]['Supragranular neurons']
        I = temp_mk.iloc[i]['Infragranular neurons']
        TOT = temp_mk.iloc[i]['Cell count']
        SLN = temp_mk.iloc[i]['SLN']

        entry = {'FLN': float(FLN), 'S': float(S), 'I': float(I),
                 'TOT': float(TOT), 'SLN': float(SLN)}
        if monkey in SLN_Data_aitor:
            SLN_Data_aitor[monkey]['source_areas'].update({source: entry})
        else:
            SLN_Data_aitor[monkey] = {'target_area': target,
                                      'source_areas': {source: entry}}

    # SLN data from Markov et al. (2014)
    temp = pd.read_csv(os.path.join(datapath, 'SLN_Data.csv'),
                       skiprows=3,
                       sep=' ',
                       names=['index', 'target_area', 'source_area', 'S', 'I',
                              'TOT', 'DIST', 'DENS', 'monkey', 'lFLN', 'SLN',
                              'INJ', 'FLN', 'cSLN'])

    SLN_Data_markov = {}
    for i in range(len(temp)):
        monkey = temp.iloc[i]['monkey']
        target = temp.iloc[i]['target_area']
        source = temp.iloc[i]['source_area']
        FLN = float(temp.iloc[i]['FLN'])
        S = temp.iloc[i]['S']
        I = temp.iloc[i]['I']
        TOT = temp.iloc[i]['TOT']
        SLN = temp.iloc[i]['SLN']

        entry = {'FLN': float(FLN), 'S': int(S), 'I': int(I),
                 'TOT': int(TOT), 'SLN': float(SLN)}
        if monkey in SLN_Data_markov:
            SLN_Data_markov[monkey]['source_areas'].update({source: entry})
        else:
            SLN_Data_markov[monkey] = {'target_area': target,
                                       'source_areas': {source: entry}}

    """
    10. Distance data
    a. Markov-Kennedy white-matter distances provided by Aitor
    b. Thom distances in the parcellation of Markov et al. (2014)
    c. White-matter distance matrix from Morales-Gregorio
    """
    markov_distance_data = {}
    for i in range(len(temp_mk)):
        monkey = temp_mk.iloc[i]['Monkey']
        target = temp_mk.iloc[i]['Target']
        source = temp_mk.iloc[i]['Source']
        distance = temp_mk.iloc[i]['WM distance (mm)']

        if monkey in markov_distance_data:
            markov_distance_data[monkey]['source_areas'].update(
                {source: {'distance': distance}})
        else:
            markov_distance_data[monkey] = {'target_area': target,
                                            'source_areas': {
                                                source: {'distance': distance}}}

    # Dictionary of visuomotor distance values
    Dist_Data_vm = {}
    for monkey in markov_distance_data:
        vm_area = markov_distance_data[monkey]['target_area']
        if vm_area not in area_list:
            continue
        if vm_area not in Dist_Data_vm:
            Dist_Data_vm[vm_area] = {}
        for source, val in markov_distance_data[monkey]['source_areas'].items():
            if source in Dist_Data_vm[vm_area]:
                Dist_Data_vm[vm_area][source] = np.append(
                    Dist_Data_vm[vm_area][source], val['distance'])
            else:
                Dist_Data_vm[vm_area][source] = np.array([val['distance']])

    # Arithmetic means; arrays are padded with zeros if necessary
    for target in Dist_Data_vm:
        dimension = 0
        for source in Dist_Data_vm[target]:
            if Dist_Data_vm[target][source].size > dimension:
                dimension = Dist_Data_vm[target][source].size
        for source in Dist_Data_vm[target]:
            array = np.append(Dist_Data_vm[target][source], np.zeros(
                dimension - Dist_Data_vm[target][source].size))
            Dist_Data_vm[target][source] = np.mean(array)

    # Remove intrinsic connections
    for target in Dist_Data_vm:
        if target in Dist_Data_vm[target]:
            del Dist_Data_vm[target][target]

    # Add target areas without data
    for a in area_list:
        if a not in Dist_Data_vm:
            Dist_Data_vm.update({a: {}})

    with open(os.path.join(datapath, 'Thom_Distances_MERetal12.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header(myreader)
        temp = next(myreader)
        areas = temp
        thom_distance_data_markov = {}
        for i in range(1, 93, 1):
            temp = next(myreader)
            dict_ = {}
            for j in range(1, 93, 1):
                dict_[areas[j]] = float(temp[j])
            thom_distance_data_markov[areas[i]] = dict_

    dist = pd.read_csv(os.path.join(
        datapath, 'WM_distances_MoralesGregorio_in_array_form.csv'), index_col=0)

    # Keep only areas from area_list that exist in the csv file
    valid_areas = [a for a in area_list if a in dist.index and a in dist.columns]
    df_filtered = dist.loc[valid_areas, valid_areas]
    aitor_distance_data = df_filtered.to_dict()

    """
    11. Intrinsic FLN data
    """
    with open(os.path.join(datapath, 'Intrinsic_FLN_Data.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header(myreader)
        Intrinsic_FLN_Data = {}
        for i in range(4):
            temp = next(myreader)
            dict_ = {'mean': float(temp[1]), 'error': float(temp[2])}
            Intrinsic_FLN_Data[temp[0]] = dict_

    """
    12. Intrinsic connectivity from Potjans & Diesmann (2014)
    """
    with open(os.path.join(datapath, 'Intrinsic_Connectivity.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header(myreader)
        intrinsic_connectivity = {}
        areas = next(myreader)
        for i in range(1, 9, 1):
            temp = next(myreader)
            dict_ = {}
            for j in range(1, 10, 1):
                dict_[areas[j]] = float(temp[j])
            intrinsic_connectivity[areas[i]] = dict_

    """
    13. Numbers of neurons and external inputs in V1
    """
    with open(os.path.join(datapath, 'Numbers_V1.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header(myreader)
        num_V1 = {}
        for i in range(0, 9, 1):
            temp = next(myreader)
            num_V1[temp[0]] = {'neurons': float(temp[1]),
                               'ext_inputs': float(temp[2])}

    """
    14. Binzegger data relating synapse location to cell body location
    """
    with open(os.path.join(datapath, 'BinzeggerData.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header(myreader)

        pre_cells = next(myreader)
        binzegger_data = {}
        for i in range(1, 38, 1):
            temp = next(myreader)
            if temp[0] != '':
                subdict = {}
                if temp[1] == '':
                    syn_layer = re.sub(r"\D", "", re.sub(r".*\(", "", temp[0]))
                else:
                    syn_layer = re.sub(r"\D", "", temp[1])
                for j in range(3, len(temp), 1):
                    try:
                        subdict[pre_cells[j]] = float(temp[j])
                    except ValueError:
                        subdict[pre_cells[j]] = temp[j]
                if temp[0] in binzegger_data:
                    binzegger_data[temp[0]]['syn_dict'].update(
                        {syn_layer: subdict})
                else:
                    binzegger_data[temp[0]] = {
                        'occurrence': float(temp[2]),
                        'syn_dict': {syn_layer: subdict}}

    """
    15. Average in-degree
    """
    temp = np.loadtxt(os.path.join(datapath, 'SynapticDensity_Cragg.csv'),
                      delimiter='\t', skiprows=2)
    av_indegree_Cragg = float(temp)

    temp = np.loadtxt(os.path.join(datapath, 'SynapticDensity_OKusky.csv'),
                      delimiter='\t', skiprows=2)
    av_indegree_OKusky = float(temp)

    """
    Store experimental data in json file
    """
    raw_data = {'area_list': area_list,
                'architecture_completed': architecture_completed,
                'markov_dist_data_aitor': Dist_Data_vm,
                'thom_distance_data_markov': thom_distance_data_markov,
                'aitor_distance_data': aitor_distance_data,
                'surface_data': surface_data,
                'FLN_Data': FLN_Data_markov,
                'Intrinsic_FLN_Data': Intrinsic_FLN_Data,
                'Intrinsic_Connectivity': intrinsic_connectivity,
                'SLN_Data': SLN_Data_aitor,
                'num_V1': num_V1,
                'Binzegger_Data': binzegger_data,
                'av_indegree_Cragg': av_indegree_Cragg,
                'av_indegree_OKusky': av_indegree_OKusky,
                'overlap': overlap,
                'total_thickness_data_beul': total_thickness,
                'neuronal_density_data_beul': Beul_neuronal_densities,
                'visual_MAM_neuronal_densities': visual_MAM_neuronal_densities,
                'raw_neuronal_numbers': raw_neuronal_numbers}

    with open(os.path.join(out_path,
                           ''.join(('visuomotor_raw_data', out_label, '.json'))),
              'w') as f:
        json.dump(raw_data, f)


if __name__ == '__main__':
    process_raw_data()
