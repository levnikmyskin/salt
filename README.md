# SALτ : Efficiently Stopping TAR by Improving Priors Estimates
We publish in this Github repository the Python code for our paper *SALτ : Efficiently Stopping TAR by Improving Priors Estimates*.
Our method details can be found in `sld/sld_stopping.py` file.

## Datasets
The RCV1-v2 dataset is available via `scikit-learn` and it will automatically be downloaded if you don't already own it. For the CLEF dataset, we make available the tf-idf vectors and their labels at https://zenodo.org/record/7142640 
## Reproducing experiments  
In order to reproduce the experiments in our paper, install the necessary packages listed in `requirements.txt`. You can then execute `run_experiments.py` and `run_experiments_sr.py` to run experiments for the RCV1-v2 and the CLEF EMED 2019 datasets respectively.  
Use the `--help` or `-h` option for more information on the available arguments.
