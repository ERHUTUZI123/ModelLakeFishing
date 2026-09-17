from pathlib import Path
import json
base = Path(__file__).resolve().parent
manifest = json.loads((base.parent/'A0_6/delivery/metrics/A0_EVALUATION_MANIFEST.json').read_text())
refs = {}
def walk(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            refs[value['path']] = value
        else:
            for child in value.values(): walk(child)
    elif isinstance(value, list):
        for child in value: walk(child)
walk(manifest)
(base/'required_files.json').write_text(json.dumps(refs, indent=2)+'\n')
for path, rec in refs.items(): print(rec.get('bytes',0),path)
print('ROOTED_PATH_RESOLUTION', Path('/u801/x98liu/model_lake').resolve())
