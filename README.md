# Computationally efficient goodness-of-fit tests through kernelized Stein discrepancy

Reproduction code for the manuscript by Zhihan Huang and Ziang Niu. The `reproduction-code/` folder contains six simulation studies (Gaussianity, two KEF studies, conditional Gaussian, RBM, and Watson sphere) and the TCGA-LUAD RPPA analysis. Run the commands below from the repository root.

**Requirements:** A Unix-like shell, Python 3.10–3.12, and R 4.4.1. Install the pinned Python and R packages with:

```bash
python -m pip install -r reproduction-code/requirements.txt
Rscript -e 'renv::restore(project = "reproduction-code")'
```

Package versions are listed in `reproduction-code/requirements.txt` and `reproduction-code/renv.lock`.

## Real data source

The TCGA-LUAD inputs come from the pinned [rBGR revision](https://github.com/bayesrx/rBGR/tree/95d8a6710950920ca0e2eb5001896b8d0c47d4df): `TCGA_Data/Proteomics_Data_Y/Y_LUAD.rds` (RPPA) and `TCGA_Data/mRNA_Immune_X/X_LUAD.rds` (immune signatures). `./reproduction-code/run_all_real_data.sh full` downloads and verifies both files, then joins them by participant.

## Option 1 (quick): Recreate figures from bundled results

The saved results are included under `reproduction-code/results/`, including one `model_evaluation.csv` table for each TCGA graph. No simulation rerun or TCGA data download is needed.

```bash
./reproduction-code/run_all_simulations.sh plot-only
./reproduction-code/run_all_real_data.sh plot-only
```

Figures are saved alongside the results.

## Option 2 (complete): Rerun all analyses

A small simulation check is available before the publication-scale runs:

```bash
./reproduction-code/run_all_simulations.sh check
```

Run the full analyses with:

```bash
./reproduction-code/run_all_simulations.sh full
./reproduction-code/run_all_real_data.sh full
```

`full` reruns all six simulations and both TCGA graph analyses. Results go to `reproduction-code/results/simulation-analysis-full/` and `reproduction-code/results/real-data-analysis-full/`. These runs can take many hours.
