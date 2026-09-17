"""Restore A0.1 prefixes and append reviewed UTF-8 files without a shell pipe."""
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent


def rec(path):
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    records=[]
    for stem in ('A0_COMMANDS','A0_DISCREPANCIES'):
        path=OUT.parent/(stem+'.md')
        old=OUT/'frozen'/(stem+'.encoding_failure.md')
        assert not old.exists(), 'Preserve previous encoding-repair evidence'
        shutil.copy2(path,old)
        prefix=(OUT/'frozen'/(stem+'.md')).read_text(encoding='utf-8')
        appendix=(OUT/(stem+'_APPEND.md')).read_text(encoding='utf-8')
        merged=prefix+'\n'+appendix
        assert '???' not in merged and '\ufffd' not in merged
        path.write_text(merged,encoding='utf-8')
        assert path.read_text(encoding='utf-8').startswith(prefix)
        records.append({'original':rec(OUT/'frozen'/(stem+'.md')),
                        'failed_append':rec(old),'UTF8_append_source':rec(OUT/(stem+'_APPEND.md')),
                        'corrected':rec(path),'original_prefix_unchanged':True})
    checked=[]
    for path in (OUT.parent.parent/'A0.md',OUT.parent.parent/'A0.2.md',
                 OUT.parent/'A0_COMMANDS.md',OUT.parent/'A0_DISCREPANCIES.md'):
        text=path.read_text(encoding='utf-8')
        assert '???' not in text and '\ufffd' not in text
        checked.append(dict(rec(path),CJK_characters=sum(0x4e00 <= ord(c) <= 0x9fff for c in text)))
    report={'stage':'A0.2','status':'PASS','cause':'PowerShell stdin encoding damaged non-ASCII literals in temporary documentation commands; production edits used UTF-8 apply_patch and were unaffected',
            'resolution':'Read/write explicit UTF-8 source files; preserve original A0.1 prefixes and failed append evidence',
            'records':records,'final_text_checks':checked,'production_code_or_graph_changed':False}
    (OUT/'A0_DOC_ENCODING_REPAIR.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','verified_UTF8_documents':len(checked),'frozen_prefixes_preserved':len(records)}))


if __name__=='__main__':
    main()
