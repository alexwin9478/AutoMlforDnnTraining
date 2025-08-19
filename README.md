# 🤖 AutoML for LSTM-Based DNNs with Optuna on RWTH HPC

[![DOI](https://zenodo.org/badge/1036040896.svg)](https://doi.org/10.5281/zenodo.16903152)

This repository contains Python code and documentation for an **Automated Machine Learning (AutoML)** workflow using the open-source framework [Optuna](https://optuna.org/).  
The AutoML studies focus on:  

- **Hyperparameter Optimization (HPO)**  
- **Neural Architecture Search (NAS)**  

for an **LSTM-based Deep Neural Network (DNN)**.  
The resulting models have been integrated into a **Deep Nonlinear Model Predictive Control (NMPC)** framework, as published in [Paper Link].  

Authors:  
- **Alexander Winkler** (winkler_a@mmp.rwth-aachen.de)  

---
  
## 📂 Repository Structure

	├── python/ # Scripts for execution of AutoML pipelines
	├── data/ # Input datasets for training and evaluation
	├── hpc_exec/ # Scripts and SLURM job files for HPC execution
	├── apptainer/ # Apptainer & Docker container creation files + cheat sheet
	├── results/ # Optuna study results, trained models, and logs
	├── plots/ # MATLAB scripts to generate plots from Optuna CSV outputs
	└── README.md # This file


---

## ⚙️ Workflow

### 1. Container Setup
- Generate an **Apptainer container** from the files in the `apptainer/` folder before running experiments on RWTH HPC.  
- Alternatively, Docker setup files are also provided for local execution.  
- The `apptainer/` folder also contains a **cheat sheet** with useful commands.  

### 2. Create a Study
- Before running, a study must be created:  
  ```bash
  python create_study_GE.py    # Generalization Error study
  python create_study_MO.py    # Multi-objective (GE + FLOPs) study
  ```

### 3. Execute a Study
- Run the created study with:  
  ```bash
  python execute_study_AutoML_GE.py   # Run HPO/NAS for Generalization Error
  python execute_study_AutoML_MO.py   # Run Multi-objective optimization
  ```
- Available study types:  
  - **_GE (Generalization Error)** → minimize test error.  
  - **_MO (Multi-Objective)** → optimize both generalization error and FLOPs (floating-point operations).  

### 4. Local Execution
- Run studies from the `python/` folder with input data from `data/`:  
  ```bash
  cd python
  python execute_study_AutoML_GE.py --config config_local.yaml
  ```

### 5. HPC Execution
- Submit distributed studies on the RWTH HPC cluster using SLURM:  
  ```bash
  cd hpc_exec
  sbatch run_optuna.slurm
  ```
- Monitor jobs:  
  ```bash
  squeue -u <your-username>
  ```

---

## 📊 Results and Plots

- Results are stored in the `results/` folder (Optuna databases, CSV logs, trained models).  
- Plots can be generated with the MATLAB scripts in the `plots/` folder:  
  - Convergence plots  
  - Pareto fronts (MO studies)  
  - Training/validation curves  

---

## 🛠 Dependencies

The code runs on **Python 3.9+** and requires:  
- [Optuna](https://optuna.org/)  
- [scikit-learn](https://scikit-learn.org/stable/)  
- [pandas](https://pandas.pydata.org/)  
- [numpy](https://numpy.org/)  

HPC requirements:  
- RWTH Aachen University HPC cluster access or any other HPC cluster 
- [Apptainer](https://apptainer.org/)  

---

## 📑 Cite us

If you are using this code, please cite the following publications:  
- [Dummy1] Paper 1, tbd   
- [Paper Link] (Deep NMPC with LSTM models)
- Data publication on Zenodo:
[![DOI](https://zenodo.org/badge/1036040896.svg)](https://doi.org/10.5281/zenodo.16903152)

---

## 📜 License

This project is licensed under the  
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).  
