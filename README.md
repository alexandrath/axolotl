# Axolotl

Axolotl is a spike sorting pipeline for primate retina data that leverages spatial spike propagation (electrical images, or EIs) to separate overlapping units and improve detection accuracy. It includes GPU-accelerated template matching, multi-pass unit refinement, and flexible residual subtraction tools.

---

## Folder Overview

This repo includes:

- `*.py`: Main pipeline components and utilities
- `*.ipynb`: Notebooks for running per-unit loops and diagnostics
- `run_multi_gpu_ei_scan.py`: Fast template scoring on MEA data
- `verify_cluster.py`: EI-based waveform clustering
- `prepare_subtraction_templates.py`: Channel-wise subtraction template generator
- `plot_ei_waveforms.py`: Spatial plotting of waveforms by electrode layout

---

## Requirements

- Python 3.9+
- PyTorch (CUDA 11.4+ recommended)
- NumPy, SciPy, scikit-learn, matplotlib
- `joblib`, `networkx`, and `h5py` for some submodules
- (Optional) MATLAB for legacy `.mat` file compatibility

---

## Running the Pipeline

This repo is designed to be run from Jupyter notebooks or Python scripts. You can:

1. Extract spike snippets
2. Cluster by EI shape
3. Match to template using GPU scan
4. Subtract matched spikes and repeat

Example notebooks:
- `axolotl_main.ipynb`
- `EI_pursuit_single_unit.ipynb`

---

## Data

Raw `.dat` files and spike times are **not included** due to size. Please contact Alexandra Kling for access.

---

## Contact

Alexandra Kling  
Senior Research Scientist, Chichilnisky Lab, Stanford University  
alexth@stanford.edu

