<img src="https://github.com/unlearning-challenge/starting-kit/assets/277639/d1fa7889-5d91-4e6d-8082-7d59ef728f9c" style="width: 100px">

# DSC 180A Quarter 1 Project: Machine Unlearning on CIFAR-10
## Fork of the NeurIPS 2023 Machine Unlearning Challenge Starting Kit

This repository is a **fork** of the official starting kit for the **NeurIPS 2023 Machine Unlearning Challenge**, adapted for my DSC 180A Capstone project at UC San Diego.

The project focuses on removing specific data subsets (forget sets) from a pre-trained **ResNet-18** model trained on **CIFAR-10**, while preserving utility on the remaining data.

### Methods Implemented
This project implements and compares three distinct machine unlearning approaches:

1.  **Fine-tuning (Baseline)**: Standard approach of fine-tuning the model on the retained data to gradually shift weights away from the forget set.
2.  **Gradient Ascent (Baseline)**: A method that intentionally maximizes the loss on the forget set to degrade the model's performance on those specific samples.
3.  **UNDIAL (Proposed Method)**: Unlearning via Decision-level Importance and Adaptive Loss. This custom approach utilizes:
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
```

---

## 2. Accessing and Storing Data

This project utilizes the **CIFAR-10** dataset. Data access is handled automatically via the provided scripts.

* **Dataset**: The code uses `torchvision.datasets.CIFAR10`. You do **not** need to download the dataset manually. When you run the notebooks for the first time, the script will check for the `data/` directory and download the necessary files if they are missing.
* **Forget Set**: The specific indices targeted for unlearning are provided in `forget_idx.npy` (included in the repo).
* **Model Weights**: The pre-trained ResNet-18 weights (`weights_resnet18_cifar10.pth`) are included in the repository root.

---

## 3. Software Dependencies

The code is implemented in Python. To reproduce the results, you must install the necessary dependencies.

### Installation
It is recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Qz07/dsc180-starting-kit.git](https://github.com/Qz07/dsc180-starting-kit.git)
    cd dsc180-starting-kit
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

**Key libraries used:** `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `jupyter`.

---

## 4. Reproducing Results

The primary way to reproduce the results is through the provided Jupyter Notebooks.

### Launching the Environment
To start the notebook server, run:

```bash
jupyter notebook
```

### Experiment Workflow
Run the notebooks in the following order to replicate the full analysis:

1.  **SimNPO & RMU Experiments (`notebooks/methods_metrics_results.ipynb`)**:
    * **Action**: Run all cells in this notebook.
    * **Output**: This will train the models using the SimNPO and RMU algorithms and generate their respective accuracy and MIA (Membership Inference Attack) scores.

2.  **UNDIAL Method Evaluation (`notebooks/undial_experiments_results.ipynb`)**:
    * **Action**: Run all cells in this notebook.
    * **Output**: This will execute the UNDIAL algorithm, perform decision-level analysis, and calculate metrics such as Truth Ratio and KL Divergence.
