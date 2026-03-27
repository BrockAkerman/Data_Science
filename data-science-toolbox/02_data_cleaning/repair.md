```python
import json
import os

def manual_nb_to_md(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    md_content = []
    # Add a header for the RAG system to find
    md_content.append(f"---")
    md_content.append(f"source_file: {filename}")
    md_content.append(f"---")
    md_content.append(f"# {filename.replace('.ipynb', '')}\n")
    
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'markdown':
            md_content.append("".join(cell['source']))
            md_content.append("\n")
        elif cell['cell_type'] == 'code':
            md_content.append("```python")
            md_content.append("".join(cell['source']))
            md_content.append("```\n")
            
    output_name = filename.replace('.ipynb', '.md')
    with open(output_name, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_content))
    return output_name

# Process all notebooks in current folder
notebooks = [f for f in os.listdir('.') if f.endswith('.ipynb')]
for nb in notebooks:
    try:
        out = manual_nb_to_md(nb)
        print(f"Successfully created: {out}")
    except Exception as e:
        print(f"Failed {nb}: {str(e)}")
```

    Successfully created: REF_data_precheck.md
    Successfully created: REF_downtyping.md
    Successfully created: REF_dtypes_and_formatting.md
    Successfully created: REF_duplicate_handling.md
    Successfully created: REF_handle_missing_values.md
    Successfully created: REF_outlier_detection.md
    Successfully created: repair.md
    Successfully created: TMPL_PIPELINE_data_cleaning.md
    Successfully created: TMPL_PIPELINE_duplicate_handling.md
    Successfully created: TMPL_PIPELINE_missing_values.md
    Successfully created: TMPL_PIPELINE_outlier_handling.md
    Successfully created: TMPL_PIPELINE_validation.md
    Successfully created: __REF_DATA_CLEANING_MASTER.md
    
