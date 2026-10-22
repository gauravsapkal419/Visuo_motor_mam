"""
Paths of the mesoscopic visuomotor model.

base_path        : folder of this file (Mesoscopic_visuomotor_mam)
data_path        : folder in which every simulation gets its own subfolder
model_data_path  : folder with the json files written by Visuomotor_Model.py
                   (VM_MAM_models/visuomotor_data)
"""
import os

base_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(base_path, 'simulations')
model_data_path = os.path.abspath(os.path.join(base_path, '..', 'visuomotor_data'))
