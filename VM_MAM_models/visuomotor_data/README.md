# visuomotor_data

Anatomical data, data processing and connectivity of the visuo-motor model. This folder turns **raw experimental data** (neuron densities, cortical thickness, surface areas, tract-tracing connectivity, distances) into the **model files** that the models read: population sizes, synapse numbers, indegrees and connection probabilities of 43 macaque cortical areas (visual, parietal, premotor and motor areas, including F1–F7), each with eight populations (23E, 23I, 4E, 4I, 5E, 5I, 6E, 6I).

The processing follows the data pipeline of the multi-area model of macaque visual cortex (Schmidt et al., 2018) and extends it with the motor and premotor areas.

---

## Table of contents

1. [Overview of the pipeline](#overview-of-the-pipeline)
2. [Folder content](#folder-content)
3. [Quick start](#quick-start)
4. [Step 1: raw data import](#step-1-raw-data-import)
5. [Step 2: data processing](#step-2-data-processing)
6. [Step 3: model parameters and connectivity](#step-3-model-parameters-and-connectivity)
7. [Step 4: diagnostics and plots](#step-4-diagnostics-and-plots)
8. [Output files](#output-files)
9. [The `out_label` argument](#the-out_label-argument)
10. [Requirements](#requirements)
11. [Sources of the data](#sources-of-the-data)
12. [Troubleshooting](#troubleshooting)

---

## Overview of the pipeline

```
raw_data/ (csv, json, txt)
      │   Visuomotor_Data.py            process_raw_data()
      ▼
visuomotor_raw_data.json
      │   Visuomotor_Data_Processing.py process_data()
      ▼
visuomotor_processed_data.json
      │   Visuomotor_Model.py           compute_Model_params()
      ▼
visuomotor_Data_Model.json
      │   Visuomotor_Model.py           compute_connectivity_files()
      ▼
visuomotor_connection_prob_data.json
visuomotor_indegree_data.json
      │   Visuomotor_Diagnostics.py     run_diagnostics()
      ▼
visuomotor_area_level_degrees.json  +  figures/
```

Every step reads the output of the step before it. The scripts write their files into this folder.

---

## Folder content

| Path | Content |
|---|---|
| `raw_data/` | Raw input files (csv, json, txt, R script for the SLN fit) |
| `Visuomotor_Data.py` | Imports the raw data and writes `visuomotor_raw_data.json` |
| `Visuomotor_Data_Processing.py` | Fills missing data, fits, computes neuron numbers, FLN and SLN, writes `visuomotor_processed_data.json` |
| `Visuomotor_Model.py` | Computes population sizes, synapse numbers, indegrees and connection probabilities |
| `Visuomotor_Diagnostics.py` | Checks and plots (connectivity, external input, connection probabilities) |
| `figures/` | Figures written by the diagnostics (created when needed) |
| `*.json` | Generated files (see [Output files](#output-files)) |

The data processing was first developed in the notebook `visuomotor_mam_data_processing.ipynb` and the model computation in `visuomotor_model.ipynb`. The scripts contain the same steps.

---

## Quick start

Run the steps in this order from this folder:

```python
import Visuomotor_Data
import Visuomotor_Data_Processing
import Visuomotor_Model
import Visuomotor_Diagnostics

Visuomotor_Data.process_raw_data()               # -> visuomotor_raw_data.json
Visuomotor_Data_Processing.process_data()        # -> visuomotor_processed_data.json
model_data = Visuomotor_Model.compute_Model_params()          # -> visuomotor_Data_Model.json
Visuomotor_Model.compute_connectivity_files(model_data)       # -> connection probabilities, indegrees
Visuomotor_Diagnostics.run_diagnostics()         # checks, degrees, figures
```

You do not have to do this by hand when you use the mesoscopic model. `Mesoscopic_visuomotor_mam/prepare_data.py` runs every step whose output file is missing when the model is created, and never overwrites existing files. To rebuild a file, delete it and all files that come after it in the pipeline.

---

## Step 1: raw data import

**Script:** `Visuomotor_Data.py` · **Function:** `process_raw_data()` · **Output:** `visuomotor_raw_data.json`

Reads the files in `raw_data/` and collects them in one json file:

| Content | Raw files (examples) |
|---|---|
| Neuron numbers of the visuo-motor areas | `m132_neuron_numbers_visuomotor_data.json` |
| Architectural type of the areas | `M132_visual_areas_architectural_type.json` |
| Surface areas (F99) | `midcortical_surface_areas_F99.json` |
| Neuron densities and total cortical thickness (Beul et al.) | `macaque_structural_data_Beul_Hilgetag_Barbas_Wagstyl.csv` |
| Neuron densities of the visual multi-area model | `VMAM_neuronal_densities.json` |
| Laminar thickness | `vmam_laminar_thickness.json`, `F1_thickness_data.json` |
| Overlap between area parcellations | `overlap_data.json` |
| FLN (Markov et al., 2014) | `Markov2014_FLN_rawdata.csv` |
| SLN | `SLN_Data.csv` |
| Intrinsic connectivity and intrinsic FLN | `Intrinsic_Connectivity.csv`, `Intrinsic_FLN_Data.csv` |
| Cell numbers of V1 | `Numbers_V1.csv` |
| Laminar synapse data (Binzegger et al.) | `BinzeggerData.csv` |
| Synaptic density | `SynapticDensity_Cragg.csv`, `SynapticDensity_OKusky.csv` |
| Distances between areas | distance data stored in the raw file |

Area names that contain a hyphen are mapped to the names in the source files (`TH-TF` → `TH_TF`, `9-46v` → `9_46v`, `9-46d` → `9_46d`).

**Main keys of `visuomotor_raw_data.json`:** `area_list`, `architecture_completed`, `thom_distance_data_markov`, `aitor_distance_data`, `surface_data`, `FLN_Data`, `Intrinsic_FLN_Data`, `Intrinsic_Connectivity`, `SLN_Data`, `num_V1`, `Binzegger_Data`, `av_indegree_Cragg`, `av_indegree_OKusky`, `overlap`, `total_thickness_data_beul`, `neuronal_density_data_beul`, `visual_MAM_neuronal_densities`, `raw_neuronal_numbers`.

---

## Step 2: data processing

**Script:** `Visuomotor_Data_Processing.py` · **Function:** `process_data()` · **Input:** `visuomotor_raw_data.json` · **Output:** `visuomotor_processed_data.json`

Fills the gaps of the experimental data and derives the quantities the model needs.

**Neuronal densities and thickness**

1. Linear fit between the neuron densities of the visual multi-area model and of Beul et al. to put both on one scale.
2. Densities for areas without data: average density of the cortical category (architectural type) for each layer and overall.
3. Linear fit of density against total cortical thickness for missing thickness values.
4. Laminar thicknesses: relative thicknesses from the available areas, regression for layer 4 from the overall density, then conversion to absolute values.
5. For the motor areas F1–F7, the neuron densities are recomputed from the available neuron numbers, the laminar thickness and the surface area.

**Neuron numbers**

Neuron numbers of each population follow from the density, thickness and surface area of each layer. The ratio of excitatory to inhibitory neurons is taken from V1 for each layer and assumed constant across areas.

**Cortico-cortical connectivity**

1. **FLN**: raw FLN values are collected per target and source area. Missing values are filled with an **exponential distance rule** fitted to the existing data.
2. **SLN**: raw SLN values are collected. Missing values are filled with a **sigmoidal fit** of SLN against the logarithmic ratio of neuron densities (the fit is run in R; see [Requirements](#requirements)).
3. **Binary symmetric connectivity**: two areas are connected if an FLN or SLN value exists in either direction.

**Main keys of `visuomotor_processed_data.json`:** `bin_sym_conn_data`, `FLN_completed`, `SLN_completed`, `architecture_completed`, `total_neuronal_densities`, `realistic_neuronal_numbers`, `total_thicknesses`, `laminar_thicknesses`, `category_density`.

---

## Step 3: model parameters and connectivity

**Script:** `Visuomotor_Model.py`

### `compute_Model_params(out_label='')`

Computes the structure of the network and writes `visuomotor_Data_Model.json`:

- **Population sizes** for each area (in a 1 mm² patch for the model, full-scale numbers for the cortico-cortical synapses).
- **Intra-area synapses** from the intrinsic connectivity of the microcircuit (Potjans & Diesmann, 2014), scaled to each area with the conversion factors between the microcircuit and the area.
- **Cortico-cortical synapses**: FLN values are normalized to `1 - FLN_intrinsic - 0.013` (1.3 % subcortical input, Markov et al., 2011) and distributed over the source populations.
- **Feedback specificity**: the synapses of feedback connections (SLN < 0.35) are redistributed so that the excitatory specificity reaches 0.93, if this option is on.
- **External synapses**: the difference between the total number of synapses and the intrinsic and cortico-cortical synapses, distributed so that every population of an area receives the same external indegree.
- **Distances** between areas.

Among others the file contains `area_list`, `population_list`, `structure`, `synapses`, `neuron_numbers` and `distances`. It returns the same dictionary.

### `compute_connectivity_files(collected_data=None, out_label='')`

Reads the model data (the dictionary given, or the json file if `None`) and writes

- `visuomotor_connection_prob_data.json`: the connection probability of every pair of populations, `[target_area][target_pop][source_area][source_pop]`, from the synapse numbers and the population sizes: `C = 1 - (1 - 1/(N_i·N_j))^N_syn`.
- `visuomotor_indegree_data.json`: the number of synapses per target neuron from each source population, including the external input.

---

## Step 4: diagnostics and plots

**Script:** `Visuomotor_Diagnostics.py` · **Function:** `run_diagnostics(out_label='', show=False, ratio_threshold=0.3)`

**Console diagnostics**

- Connections with `N_syn / (N_i · N_j)` above a threshold (values near 1 mean that more synapses than possible connections were assigned).
- Minimum and maximum connection probability.
- Difference between the approximate (`N_syn / (N_i · N_j)`) and the full expression for the connection probability.

**Files**

- `visuomotor_area_level_degrees.json`: area-level indegrees (per neuron of the target area, with external input) and outdegrees (per neuron of the source area).

**Plots** (saved to `figures/`)

| Plot | File |
|---|---|
| External synapses per area | `external_synapses_per_area.png` |
| External indegree per population | `external_indegrees.png` |
| Fraction of inputs from each source area | `input_fractions.png` |
| SLN matrix | `sln_completed.png` |
| Population-level connection probability | `connection_probability_matrix.png` |

Further connectivity plots (binary symmetric connectivity, FLN and an overview with SLN and connection probability) are available through the functions `plot_bin_sym_connectivity`, `plot_fln_matrix` and `plot_connectivity_overview` if you have added the connectivity plot block to this file. In all area matrices the rows are target areas and the columns source areas.

---

## Output files

| File | Written by | Used by |
|---|---|---|
| `visuomotor_raw_data.json` | `Visuomotor_Data.py` | `Visuomotor_Data_Processing.py`, `Visuomotor_Model.py` |
| `visuomotor_processed_data.json` | `Visuomotor_Data_Processing.py` | `Visuomotor_Model.py`, diagnostics |
| `visuomotor_Data_Model.json` | `Visuomotor_Model.py` | mesoscopic model (population sizes, distances), diagnostics |
| `visuomotor_connection_prob_data.json` | `Visuomotor_Model.py` | mesoscopic model (weights), diagnostics |
| `visuomotor_indegree_data.json` | `Visuomotor_Model.py` | mean external input of the mesoscopic model, diagnostics |
| `visuomotor_area_level_degrees.json` | `Visuomotor_Diagnostics.py` | analysis |
| `mu_values.json` | `Mesoscopic_visuomotor_mam/external_input.py` | mesoscopic model (constant input of every population) |

`mu_values.json` is not created by the scripts in this folder; it is built from `visuomotor_indegree_data.json` by the mesoscopic model.

---

## The `out_label` argument

The functions of `Visuomotor_Model.py`, `Visuomotor_Data_Processing.py` and `Visuomotor_Diagnostics.py` take an `out_label`. It is added to the file names (`visuomotor_Data_Model<out_label>.json`, ...), so that you can keep several versions of the data, for example with different options, side by side. The default is an empty string.

---

## Requirements

- Python 3 with `numpy`, `scipy`, `pandas`, `matplotlib`, `nested_dict`
- `scikit-learn` or `statsmodels` are not needed
- The SLN fit uses an R script (`SLN_logdensities.R`). With R and the library used by the script installed, the fit is recomputed. Without R, the stored result of the fit in `raw_data/` has to be available. <!-- TODO: state exactly which R packages are needed and whether the script falls back to stored values -->
- `Visuomotor_Model.py` imports `network_params` from `default_params_VMAM.py` (in the parent folder `VM_MAM_models/`). That module reads `visuomotor_raw_data.json` when it is imported, so the raw data file has to exist before `Visuomotor_Model` is imported.

---

## Sources of the data

<!-- TODO: check the references and add the ones that are missing -->

- Markov NT et al. (2014). Anatomy of hierarchy: feedforward and feedback pathways in macaque visual cortex. *Journal of Comparative Neurology* 522: 225 (FLN and SLN data).
- Markov NT et al. (2011). Weight consistency specifies regularities of macaque cortical networks. *Cerebral Cortex* 21: 1254 (subcortical input fraction).
- Beul SF, Hilgetag CC and colleagues: cortical structural data used for neuron density and thickness (`macaque_structural_data_Beul_Hilgetag_Barbas_Wagstyl.csv`).
- Binzegger T, Douglas RJ, Martin KAC (2004). A quantitative map of the circuit of cat primary visual cortex. *Journal of Neuroscience* 24: 8441.
- Potjans TC, Diesmann M (2014). The cell-type specific cortical microcircuit: relating structure and activity in a full-scale spiking network model. *Cerebral Cortex* 24: 785.
- Schmidt M, Bakker R, Hilgetag CC, Diesmann M, van Albada SJ (2018). Multi-scale account of the network structure of macaque visual cortex. *Brain Structure and Function* 223: 1409.
- Cragg BG (1967), O'Kusky J and Colonnier M (1982): synaptic density data.

---

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `FileNotFoundError: visuomotor_raw_data.json` when importing `Visuomotor_Model` | `default_params_VMAM` reads the raw data file on import. Run `Visuomotor_Data.process_raw_data()` first. |
| `ModuleNotFoundError: No module named 'Visuo_motor_MAM'` | Harmless if the next import (`Visuo_motor_mam...`) works. `Visuomotor_Model.py` tries both spellings of the package name. |
| A step does not run | Its output file already exists. Delete the file (and the files after it) to rebuild it. |
| Files are missing in `raw_data/` | Copy the raw input files into `raw_data/`. The import step stops with a file error naming the missing file. |
| Changed file names | The scripts write fixed file names. The mesoscopic model expects the names in the table above (`data_files` in `default_params.py`). |
