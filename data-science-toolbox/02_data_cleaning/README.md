# 02_data_cleaning — Cookbook Folder

## File Taxonomy

| Prefix | Purpose | When to open |
|---|---|---|
| `__REF_` | Process knowledge: ordered steps, checklists, decision guides | **Before** starting — orient yourself |
| `REF_` | Ad-hoc snippets: for loops, quick utilities, one-liners | **During** cleaning — grab what you need |
| `TMPL_PIPELINE_` | Production pipelines: config-driven, audit-logged functions | **When** you need reproducible, recruiter-ready code |

## Cleaning Order (always follow this sequence)

```
1. Precheck          →  REF_data_precheck.ipynb
2. Duplicates        →  REF_duplicate_handling       |  TMPL_PIPELINE_duplicate_handling
3. Missing Values    →  REF_handle_missing_values    |  TMPL_PIPELINE_missing_values
4. Data Types        →  REF_dtypes_and_formatting    |  (included in TMPL_PIPELINE_data_cleaning)
5. Outliers          →  REF_outlier_detection        |  TMPL_PIPELINE_outlier_handling
6. Value Correction  →  REF_dtypes_and_formatting
7. Rename / Reorder  →  REF_dtypes_and_formatting
8. Validate/Export   →                               |  TMPL_PIPELINE_validation
```

## Sibling File Map

| Concept | REF_ (ad-hoc) | TMPL_PIPELINE_ (production) | QUICKREF |
|---|---|---|---|
| Full process reference | `__REF_DATA_CLEANING_MASTER.ipynb` | `TMPL_PIPELINE_data_cleaning.ipynb` | — |
| Precheck | `REF_data_precheck.ipynb` | — | — |
| Duplicates | `REF_duplicate_handling.ipynb` | `TMPL_PIPELINE_duplicate_handling.ipynb` | `..._QUICKREF.md` |
| Missing Values | `REF_handle_missing_values.ipynb` | `TMPL_PIPELINE_missing_values.ipynb` | `..._QUICKREF.md` |
| Data Types | `REF_dtypes_and_formatting.ipynb` | *(in data_cleaning pipeline)* | — |
| Outliers | `REF_outlier_detection.ipynb` | `TMPL_PIPELINE_outlier_handling.ipynb` | `..._QUICKREF.md` |
| Memory reduction | `REF_downtyping.ipynb` | — | — |
| Validation | — | `TMPL_PIPELINE_validation.ipynb` | — |
