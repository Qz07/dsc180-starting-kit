<img src="https://github.com/unlearning-challenge/starting-kit/assets/277639/d1fa7889-5d91-4e6d-8082-7d59ef728f9c" style="width: 100px">

# DSC 180A Quarter 1 Project: Machine Unlearning on CIFAR-10
## Fork of the NeurIPS 2023 Machine Unlearning Challenge Starting Kit

This repository is a **fork** of the official starting kit for the **NeurIPS 2023 Machine Unlearning Challenge**, adapted for my DSC 180A Capstone project at UC San Diego.

The project focuses on removing specific data subsets (forget sets) from a pre-trained **ResNet-18** model trained on **CIFAR-10**, while preserving utility on the remaining data.

### Key Contribution: UNDIAL
In addition to the standard challenge baselines (Fine-tuning, Gradient Ascent), this repository implements **UNDIAL** (Unlearning via Decision-level Importance and Adaptive Loss). This custom method utilizes:
* **Truth Ratio**: To measure decision confidence shifts.
* **KL Divergence**: To constrain the model from deviating too far from the original distribution on retained data.
* **Margin Metrics**: To ensure robust decision boundaries.

---

## 1. Repository Structure

The code is organized to separate the standard challenge kit from my custom experiments.

```text
.
├─ README.md
├─ requirements.txt             # Python dependencies
├─ forget_idx.npy               # Indices of the specific images to "unlearn"
├─ weights_resnet18_cifar10.pth # Pre-trained model weights
│
├─ notebooks/
│   ├─ Unlearning-CIFAR10.ipynb         # Original challenge starter (Data setup & basic visualizations)
│   ├─ methods_metrics_results.ipynb    # Implementation and aggregation of Baseline methods
│   └─ undial_experiments_results.ipynb # Implementation of UNDIAL method + custom metrics
│
└─ data/                        # Directory created automatically to store CIFAR-10
