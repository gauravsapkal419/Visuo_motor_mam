"""
prepare_data
============

Creates the json files that the mesoscopic model needs, if they are missing.
The files are produced by the three scripts in visuomotor_data/ in this order:

1. Visuomotor_Data.process_raw_data()
       raw_data/*  ->  visuomotor_raw_data.json
2. Visuomotor_Data_Processing.process_data()
       visuomotor_raw_data.json  ->  visuomotor_processed_data.json
3. Visuomotor_Model.compute_Model_params()
       visuomotor_processed_data.json  ->  visuomotor_Data_Model.json
4. Visuomotor_Model.compute_connectivity_files()
       visuomotor_Data_Model.json  ->  visuomotor_connection_prob_data.json
                                       visuomotor_indegree_data.json

Every step is only run if its output file is missing. Existing files are
never overwritten. Delete a file to rebuild it and everything after it.

The raw file is created first in every case, because Visuomotor_Model
imports default_params_VMAM, which reads it at import time.

mu_values.json (mean input from the Siegert model) cannot be created here
and has to exist already.
"""
import importlib
import os
import sys
import json

RAW_FILE = 'visuomotor_raw_data.json'
PROCESSED_FILE = 'visuomotor_processed_data.json'


def _import_from(directory, module_name):
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module(module_name)


def ensure_mean_input(data_dir, data_files, input_params, area_list, pop_list,
                      verbose=True):
    from external_input import MU_PARAMS_FILE, build_mean_input_file

    mu_path = os.path.join(data_dir, data_files['mean_input'])
    par_path = os.path.join(data_dir, MU_PARAMS_FILE)

    rebuild = not os.path.exists(mu_path)
    if not rebuild and os.path.exists(par_path):
        with open(par_path) as f:
            rebuild = json.load(f) != json.loads(json.dumps(input_params))
    if rebuild:
        if verbose:
            print(f"[prepare_data] creating {data_files['mean_input']}")
        build_mean_input_file(data_dir, data_files, input_params,
                              area_list, pop_list)

def ensure_model_data(data_dir, data_files, input_params = None, verbose=True):
    """
    Create the missing json files of the visuomotor data pipeline.

    Parameters
    ----------
    data_dir : str
        Folder with Visuomotor_Data.py, Visuomotor_Data_Processing.py,
        Visuomotor_Model.py and raw_data/ (VM_MAM_models/visuomotor_data).
    data_files : dict
        params['data_files'] of the network parameters.
    """
    data_dir = os.path.abspath(data_dir)

    def exists(name):
        return os.path.exists(os.path.join(data_dir, name))

    def say(msg):
        if verbose:
            print(f"[prepare_data] {msg} \n")

    model_fn = data_files['model_data']
    conn_fn = data_files['connection_probs']
    mean_input_fn = data_files['mean_input']

    need_model = not exists(model_fn)
    need_conn = not exists(conn_fn)

    # 1. raw data
    if not exists(RAW_FILE):
        say("creating " + RAW_FILE)
        _import_from(data_dir, 'Visuomotor_Data').process_raw_data()
        if not exists(RAW_FILE):
            raise RuntimeError(
                f"Visuomotor_Data.process_raw_data() did not create {RAW_FILE} "
                f"in {data_dir}.")

    # 2. processed data, created whenever it is missing
    if not exists(PROCESSED_FILE):
        say("creating " + PROCESSED_FILE)
        _import_from(data_dir, 'Visuomotor_Data_Processing').process_data()
        if not exists(PROCESSED_FILE):
            raise RuntimeError(
                f"Visuomotor_Data_Processing.process_data() did not create "
                f"{PROCESSED_FILE} in {data_dir}.")

    # 3. model parameters
    model_data = None
    if need_model:
        say("creating " + model_fn)
        model_data = _import_from(data_dir, 'Visuomotor_Model').compute_Model_params()

    # 4. connectivity
    if need_conn:
        say("creating " + conn_fn)
        _import_from(data_dir, 'Visuomotor_Model').compute_connectivity_files(model_data)

    for name in (model_fn, conn_fn):
        if not exists(name):
            raise RuntimeError(
                f"{name} was not created in {data_dir}. The scripts in that "
                f"folder write fixed file names; check data_files in "
                f"default_params.py.")
        
    if input_params is not None:
        from default_params import complete_area_list, population_list
        ensure_mean_input(data_dir, data_files, input_params,
                          complete_area_list, population_list, verbose)
    if not exists(mean_input_fn):
        raise FileNotFoundError(f"{mean_input_fn} is missing in {data_dir}.")
