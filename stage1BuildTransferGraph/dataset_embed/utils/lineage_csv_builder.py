from huggingface_hub import HfApi
from pathlib import Path
import pandas as pd

api = HfApi()

model_ids = \
["TransferGraph/marcelcastrobr_sagemaker-distilbert-emotion-finetuned-lora-tweet_eval_offensive",
 "google/gemma-4-31B-it"]

df = pd.DataFrame(columns=["child_model_id", "relation", "parent_model_id"])

for model_id in model_ids:
    info = api.model_info(
        repo_id=model_id,
        expand=["baseModels"]
    )

    print(info.base_models)
    relation = info.base_models.get('relation')
    models_list = info.base_models.get('models', [])
    if models_list:
        child_model_id = models_list[0].get('_id')
        parent_model_id = models_list[0].get('id')
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