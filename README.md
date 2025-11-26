# TSAD-MetricX: A Problem-Oriented Evaluation Framework for Time Series Anomaly Detection

This repository provides the official implementation for the experiments in the paper:

**“A Problem-Oriented Taxonomy of Evaluation Metrics for Time Series Anomaly Detection”**  
(available on arXiv)

It extends an existing open-source evaluation library for time series anomaly detection (TSAD) by adding a new metric (PATE) and includes the experimental setup used in the paper for comparative analysis.

---

## 🔧 Extension to TSAD\_eval Repository

Specifically:A new metric(PATE) has been added to expand the existing collection of TSAD evaluation methods.🔗 https://github.com/sondsorb/TSAD_eval

---

## 📎 Included Experimental Code

This repository also contains the **full experimental pipeline** used in the paper, including:

- Synthetic dataset generation (5 anomaly types)
- Controlled anomaly contamination (5%–20%)
- Quality-gradient degradation (from genuine to random predictions)
- Metrics benchmarking:
  - Classical metrics
  - Event-based metrics
  - Tolerance-aware metrics
  - Cost-aware metrics
  - Robustness-oriented metrics

The experiments are designed **not to test detector performance**, but rather to **evaluate the discriminative ability and robustness of evaluation metrics** themselves.

---

## 🚀 How to Run the Experiments

All experiments can be executed using a single script:

```bash

bash run_experiment.sh <mode> [gpu_devices]

bash run_experiment.sh full 0,1,2,3

