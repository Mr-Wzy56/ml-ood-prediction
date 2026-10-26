# OOD Detection and Selective Prediction

This is a small course project on out-of-distribution (OOD) detection and selective prediction. It compares standard training with energy-aware fine-tuning on CIFAR-10 and CIFAR-100, using SVHN and DTD for OOD evaluation.

The code includes data preparation, ResNet-18 training, scoring, and evaluation. Current milestone results are in [results_summary.csv](runs/results_summary.csv); experiment details are in [the Pre2 report](runs/pre2_milestone_report.md).

## Setup

Install dependencies with `pip install -r requirements.txt`. Dataset files and model checkpoints are not included; see the report and source scripts for data preparation and experiment commands.

This repository is for coursework and experimentation, not a production system.
