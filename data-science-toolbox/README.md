# DS Toolkit — Personal Cookbook & Project Template

## How This Repo Works

Every new project starts by copying `_project_template.ipynb` into your working directory.
As you develop useful code in a project, harvest it back into the corresponding cookbook folder here.

**The Bijection:**
| Project Notebook Section | Cookbook Folder |
|---|---|
| 00. Problem Definition | `00_problem_framing/` |
| 01. Data Ingestion & First Look | `01_data_ingestion/` |
| 02. Data Cleaning | `02_data_cleaning/` |
| 03. Feature Engineering | `03_feature_engineering/` |
| 04. EDA | `04_eda/` |
| 05. Preprocessing & Splitting | `05_preprocessing_splitting/` |
| 06a. Statistical / AB Testing | `06a_statistical_testing/` |
| 06b. Model Development | `06b_model_development/` |
| 07a. Causal Inference | `07a_causal_inference/` |
| 07b. Model Evaluation | `07b_model_evaluation/` |
| 08. Business Translation | `08_business_translation/` |
| 09. Conclusions & Future Work | `09_conclusions/` |
| — | `utils/` (cross-cutting helpers) |

## Harvest Loop
```
Copy template → Build project → Harvest best code → Deposit into cookbook
                                        ↑________________________________|
```

## Branch Logic
- **Predictive modeling projects** → use sections 06b, 07b
- **AB test / inference projects** → use sections 06a, 07a
- **Some projects use both** (e.g., AB test + uplift model)
