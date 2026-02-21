#!/bin/bash

# Create base directories
mkdir -p data-science-toolbox/{environment,datasets/{raw,processed,examples},notebooks,src,templates/{notebooks,project_docs},docs/{architecture,methodology,references}}

# EDA
mkdir -p data-science-toolbox/notebooks/eda

# Preprocessing
mkdir -p data-science-toolbox/notebooks/preprocessing

# Modeling structure
mkdir -p data-science-toolbox/notebooks/modeling/{supervised,unsupervised,deep_learning}

# Supervised → Regression
mkdir -p data-science-toolbox/notebooks/modeling/supervised/regression/{linear_models,tree_based,other}

# Supervised → Classification
mkdir -p data-science-toolbox/notebooks/modeling/supervised/classification/{linear_models,tree_based,other}

# Supervised → Forecasting
mkdir -p data-science-toolbox/notebooks/modeling/supervised/forecasting

# Supervised → Deep Learning
mkdir -p data-science-toolbox/notebooks/modeling/supervised/deep_learning

# Unsupervised → Clustering
mkdir -p data-science-toolbox/notebooks/modeling/unsupervised/clustering

# Unsupervised → Association Rules
mkdir -p data-science-toolbox/notebooks/modeling/unsupervised/association_rules

# Unsupervised → Dimensionality Reduction
mkdir -p data-science-toolbox/notebooks/modeling/unsupervised/dimensionality_reduction

# Unsupervised → Anomaly Detection
mkdir -p data-science-toolbox/notebooks/modeling/unsupervised/anomaly_detection

# Deep Learning (general)
mkdir -p data-science-toolbox/notebooks/modeling/deep_learning

# Evaluation
mkdir -p data-science-toolbox/notebooks/evaluation

# Pipelines
mkdir -p data-science-toolbox/notebooks/pipelines

# src modules
mkdir -p data-science-toolbox/src/{preprocessing,modeling,evaluation,utils,visualization}

# Templates
mkdir -p data-science-toolbox/templates/notebooks
mkdir -p data-science-toolbox/templates/project_docs

# Project documentation templates
touch data-science-toolbox/templates/project_docs/{pace_strategy_template.md,executive_summary_template.md,raci_chart_template.md,stakeholder_map_template.md,problem_statement_template.md,ds_project_brief_template.md}

# Root files
touch data-science-toolbox/README.md
touch data-science-toolbox/environment/{base_environment.yml,requirements.txt}

echo "Repository structure created successfully!"
