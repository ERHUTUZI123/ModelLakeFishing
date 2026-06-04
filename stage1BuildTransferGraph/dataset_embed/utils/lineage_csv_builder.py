from huggingface_hub import HfApi
from pathlib import Path
import pandas as pd

api = HfApi()

model_ids = \
["google/gemma-4-31B-it", 
 "gghfez/gemma-4-31b-it-control-vectors",
 "nvidia/Gemma-4-31B-IT-NVFP4",
 "virtuous7373/Gemma-4-Harmonia-31B",
 "unsloth/gemma-4-31B-it-GGUF"]

df = pd.DataFrame(columns=["model", "relation", "base_model"])

for model_id in model_ids:
    child_model_id = model_id
    info = api.model_info(
        repo_id=model_id,
        expand=["baseModels"]
    )

    print(info.base_models)
    relation = info.base_models.get('relation')
    print(relation)
    models_list = info.base_models.get('models', [])
    if models_list:
        for model in models_list:
            parent_model_id = model.get('id')
            df.loc[len(df)] = [child_model_id, relation, parent_model_id]
    else:
        continue
print(df)

script_dir = Path(__file__).resolve().parent
output_dir = script_dir.parent / "data"
output_file = output_dir / "lineage_records.csv"
output_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False)
print(f"File successfully saved to: {output_file}")