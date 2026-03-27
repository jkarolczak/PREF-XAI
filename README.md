# Personalized Rule-Based Explanations for Black-Box ML Decisions

This repository contains the official codebase for the paper **"Personalized Rule-Based Explanations for Black-Box ML
Decisions"** by Salvatore Greco, Jacek Karolczak, Jerzy Stefanowski, and Roman Słowiński.

The project provides an explainable AI (XAI) framework that generates personalized, local, rule-based explanations for
black-box machine learning models. By combining human-interpretable "if-then" decision rules with formal preference
modeling via Robust Ordinal Regression (ROR) and Stochastic Multicriteria Acceptability Analysis (SMAA), the method
tailors explanations to individual user preferences without sacrificing the predictive accuracy of the underlying model.

## Features

- **Black-Box Model:** Multilayer Perceptron (MLP) classifier with learnable embeddings for categorical features.
- **Rule Induction:** Apriori-based induction of "if-then" decision rules from MLP-labeled data, evaluated using
  Support, Confirmation (e.g., $N$ or $S$ measure), and Complexity criteria.
- **Preference Modeling:** Employs holistic preference elicitation through simulated Decision Makers (DMs) ranking a
  small reference set of rules.
- **Optimization Strategies:**
    - Max $\epsilon$ (Epsilon Maximization): Identifies a single, maximum-margin utility function that discriminates
      user-ranked rules.
    - $H\&R^C$ (Hit-and-Run Centroid): Utilizes SMAA to uniformly sample the compatible weight space via Hit-and-Run
      MCMC sampling to derive highly robust, expected preference parameters.

## Repository Structure

    .
    ├── lib/                             # Core library
    │   ├── data.py                      # Dataset loaders and preprocessors
    │   ├── model.py                     # PyTorch-based MLP classifier and training loops
    │   ├── metrics.py                   # Evaluation metrics (Kendall's tau, Jaccard Index)
    │   ├── config.py                    # Global configuration and fixed seeds
    │   ├── tools.py                     # Helper functions
    │   ├── rules/                       # Rule induction modules
    │   │   ├── apriori.py               
    │   │   ├── frequent_itemsets.py     
    │   │   ├── association_rules.py     
    │   │   └── utils.py                 
    │   ├── optimization/                # Preference learning and optimization routines
    │   │   ├── prus.py                  # Preference-based Rule Utility Score (PRUS)
    │   │   ├── epsilon.py               # Epsilon Maximization linear programming
    │   │   ├── hit_and_run.py           # Hit-and-Run MCMC sampling for SMAA
    │   │   └── extreme_ranking.py       
    │   └── dms/                         # Simulated Decision Makers for evaluation
    │       ├── interface_dm.py          
    │       ├── random_dm.py             
    │       └── cards_dm.py              
    ├── experiment_instance.py           # Generate personalized explanations for a single instance
    ├── experiment_metrics.py            # Compute large-scale metrics across datasets (RQ1 & RQ2)
    ├── experiment_cv.py                 # Cross-validation for the base MLP models
    ├── experiment_reference_ranking_len.py # Evaluate impact of reference ranking length
    ├── experiment_top_k.py              # Evaluate top-K rule overlaps
    ├── bulk.sh                          # Bash script for automating bulk execution
    └── requirements.txt                 # Python dependencies

## Installation

1. Clone the repository:

```shell
   git clone https://github.com/jkarolczak/personalized-rule-explanations.git
   cd personalized-rule-explanations
```

2. Install the required dependencies:

```shell
   pip install -r requirements.txt
```

## Datasets

The framework evaluates real-world tabular datasets. To run the experiments, download the datasets and place them in the
`data/` directory.

Supported datasets included in the original paper are:

1. **Churn (Banking):** https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
2. **Churn in Telecom's Dataset:** https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset
3. **HELOC (Home Equity Line of Credit):**
   Kaggle https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc
4. (Toy Example) **Breast Cancer Wisconsin (Original):
   ** https://www.kaggle.com/datasets/mariolisboa/breast-cancer-wisconsin-original-data-set

## Usage

### 1. Model Cross-Validation

Evaluate the base predictive performance (Accuracy, Balanced Accuracy) of the MLP on a specific dataset:

    python experiment_cv.py churn_banking

### 2. Single Instance Explanation

Generate a personalized rule explanation for a specific instance. Use `--retrain` (or `-r`) to train the MLP from
scratch before rule induction.

    # Example: Dataset 'churn_banking', Target Class 1, Instance Index 0, using S-measure
    python experiment_instance.py churn_banking 1 0 --confirm_measure S --confirm_threshold 0.15 -r

### 3. Large-Scale Mass Evaluation

Run the automated simulated user evaluation across all instances to compute ranking fidelity and top-k Jaccard
similarities:

    # Uses all available CPU cores
    python experiment_metrics.py churn_banking --n_jobs -1

### 4. Evaluate Reference Ranking Length Impact

Test how the length of the user's initial reference ranking (e.g., 5 vs 10 rules) impacts the PRUS parameter recovery:

    python experiment_reference_ranking_len.py churn_banking --n_jobs -1

### Automating Bulk Runs

You can use the provided bash script to iterate over multiple datasets and configurations simultaneously:

    bash bulk.sh
