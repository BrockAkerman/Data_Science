DECISIONING:

                     ┌──────────────────────────────┐
                     │  Do you have labeled data?   │
                     └───────────────┬──────────────┘
                                     │
               Yes (Supervised)      │        No (Unsupervised)
                                     │
                                     ▼
                     ┌──────────────────────────────┐
                     │   What is the task type?     │
                     └───────────────┬──────────────┘
                                     │
          ┌───────────────┬──────────┴───────────┬───────────────┐
          │ Regression    │ Classification       │ Forecasting   │ Deep Learning?
          │               │                      │               │
          ▼               ▼                      ▼               ▼
   notebooks/        notebooks/             notebooks/       notebooks/
   modeling/         modeling/              modeling/        modeling/
   supervised/       supervised/            supervised/      supervised/
   regression/       classification/        forecasting/     deep_learning/





                           ┌──────────────────────────────┐
                           │ What type of unsupervised    │
                           │     learning is this?        │
                           └───────────────┬──────────────┘
                                           │
        ┌───────────────┬──────────────────┴─────────────────────┬──────────────────┐
        │ Clustering    │ Association Rules                      │ Dimensionality   │ Anomaly
        │               │                                        │ Reduction        │ Detection
        ▼               ▼                                        ▼                  ▼
notebooks/        notebooks/                               notebooks/          notebooks/
modeling/         modeling/                                modeling/           modeling/
unsupervised/     unsupervised/                            unsupervised/       unsupervised/
clustering/       association_rules/                       dimensionality_     anomaly_detection/
                                                             reduction/





STRUCTURE:

data-science-toolbox/
│
├── README.md
│
├── environment/
│   ├── base_environment.yml
│   └── requirements.txt
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── examples/
│
├── notebooks/
│   ├── eda/
│   │   ├── univariate_analysis.ipynb
│   │   ├── bivariate_analysis.ipynb
│   │   └── visualization_templates.ipynb
│   │
│   ├── preprocessing/
│   │   ├── missing_values.ipynb
│   │   ├── feature_engineering.ipynb
│   │   ├── scaling_encoding.ipynb
│   │   └── outlier_detection.ipynb
│   │
│   ├── modeling/
│   │   ├── supervised/
│   │   │   ├── regression/
│   │   │   │   ├── linear_models/
│   │   │   │   │   ├── linear_regression.ipynb
│   │   │   │   │   ├── lasso.ipynb
│   │   │   │   │   └── ridge.ipynb
│   │   │   │   ├── tree_based/
│   │   │   │   │   ├── decision_tree_regressor.ipynb
│   │   │   │   │   ├── random_forest_regressor.ipynb
│   │   │   │   │   └── gradient_boosting_regressor.ipynb
│   │   │   │   └── other/
│   │   │   │       ├── svr.ipynb
│   │   │   │       └── knn_regressor.ipynb
│   │   │   │
│   │   │   ├── classification/
│   │   │   │   ├── linear_models/
│   │   │   │   │   ├── logistic_regression.ipynb
│   │   │   │   │   └── linear_discriminant_analysis.ipynb
│   │   │   │   ├── tree_based/
│   │   │   │   │   ├── decision_tree_classifier.ipynb
│   │   │   │   │   ├── random_forest_classifier.ipynb
│   │   │   │   │   └── xgboost_classifier.ipynb
│   │   │   │   └── other/
│   │   │   │       ├── svm_classifier.ipynb
│   │   │   │       └── knn_classifier.ipynb
│   │   │   │
│   │   │   ├── forecasting/
│   │   │   │   ├── arima.ipynb
│   │   │   │   ├── prophet.ipynb
│   │   │   │   └── lstm_forecasting.ipynb
│   │   │   │
│   │   │   └── deep_learning/
│   │   │       ├── feedforward_nn.ipynb
│   │   │       ├── cnn.ipynb
│   │   │       └── rnn.ipynb
│   │   │
│   │   ├── unsupervised/
│   │   │   ├── clustering/
│   │   │   │   ├── kmeans.ipynb
│   │   │   │   ├── dbscan.ipynb
│   │   │   │   └── hierarchical.ipynb
│   │   │   │
│   │   │   ├── association_rules/
│   │   │   │   ├── apriori.ipynb
│   │   │   │   └── fp_growth.ipynb
│   │   │   │
│   │   │   ├── dimensionality_reduction/
│   │   │   │   ├── pca.ipynb
│   │   │   │   ├── tsne.ipynb
│   │   │   │   └── umap.ipynb
│   │   │   │
│   │   │   └── anomaly_detection/
│   │   │       ├── isolation_forest.ipynb
│   │   │       └── one_class_svm.ipynb
│   │   │
│   │   └── deep_learning/
│   │       ├── autoencoder.ipynb
│   │       ├── variational_autoencoder.ipynb
│   │       └── gan.ipynb
│   │
│   ├── evaluation/
│   │   ├── regression_metrics.ipynb
│   │   ├── classification_metrics.ipynb
│   │   └── model_comparison.ipynb
│   │
│   └── pipelines/
│       ├── sklearn_pipeline_template.ipynb
│       ├── end_to_end_regression.ipynb
│       └── end_to_end_classification.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── modeling/
│   ├── evaluation/
│   ├── utils/
│   └── visualization/
│
├── templates/
│   ├── notebooks/
│   │   ├── notebook_template.ipynb
│   │   └── pipeline_template.py
│   │
│   ├── project_docs/
│   │   ├── pace_strategy_template.md
│   │   ├── executive_summary_template.md
│   │   ├── raci_chart_template.md
│   │   ├── stakeholder_map_template.md
│   │   ├── problem_statement_template.md
│   │   └── ds_project_brief_template.md
│   │
│   └── project_structure_template.md
│
└── docs/
    ├── architecture/
    ├── methodology/
    └── references/
