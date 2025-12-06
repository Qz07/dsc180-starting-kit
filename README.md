<img src="https://github.com/unlearning-challenge/starting-kit/assets/277639/d1fa7889-5d91-4e6d-8082-7d59ef728f9c" style="width: 100px">

# DSC 180A Capstone – Machine Unlearning on CIFAR-10  
## Fork of the NeurIPS 2023 Machine Unlearning Challenge Starting Kit

This repository is a **fork** of the official starting kit for the  
**NeurIPS 2023 Machine Unlearning Challenge**, adapted for my DSC 180A  
Quarter 1 capstone project at UC San Diego.

All experiments are done on the **CIFAR-10** dataset with a **ResNet-18** backbone.  
In addition to the original baselines, this fork includes my method:

- **UNDIAL** – a decision-level unlearning approach with Truth Ratio, KL, and margin metrics.

The primary goal of this repo is **reproducibility**: anyone should be able to clone the repo, set up the environment, and re-run the main notebooks.

---

## 1. Repository Structure

Key files (not exhaustive):

```text
.
├─ README.md
├─ notebooks/
│  ├─ Unlearning-CIFAR10.ipynb          # Main starting-kit notebook (data, model, baselines)
│  ├─ methods_metrics_results.ipynb     # Aggregated baseline methods + metrics
│  └─ undial_experiments_results.ipynb  # UNDIAL method + UNDIAL-specific metrics
└─ data/                                # Created automatically on first run
