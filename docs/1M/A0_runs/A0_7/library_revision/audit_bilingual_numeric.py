"""Compare all pure numeric table cells in corresponding EN and ZH tables."""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
assert len(sys.argv) in (1, 3), "Pass both final EN and ZH paths or neither."
paths = [Path(name).resolve() for name in sys.argv[1:]] if len(sys.argv) == 3 else [HERE / f"EVIDENCE_SOURCE_LIBRARY_{lang}.draft.md" for lang in ("en", "zh")]

def tables(path):
    found, current = [], []
    in_code = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if line.startswith("|") and not in_code:
            current.append([cell.strip() for cell in line.strip("|").split("|")])
        elif current:
            found.append(current)
            current = []
    if current:
        found.append(current)
    return found

en, zh = [tables(path) for path in paths]
assert len(en) == len(zh), ("table count", len(en), len(zh))
checked = []
for ti, (a, b) in enumerate(zip(en, zh)):
    assert len(a) == len(b), ("rows", ti, len(a), len(b))
    for ri, (ra, rb) in enumerate(zip(a, b)):
        assert len(ra) == len(rb), ("columns", ti, ri, len(ra), len(rb))
        for ci, (ca, cb) in enumerate(zip(ra, rb)):
            clean = ca.replace("**", "")
            if re.fullmatch(r"[\d,.%+ /−-]+|—", clean):
                # Localize list separators only; thousands and decimal formatting stay exact.
                list_separator_only = ", " in ca and "、" in cb and ca.replace(", ", ",") == cb.replace("、", ",")
                assert ca == cb or list_separator_only, ("numeric table mismatch", ti, ri, ci, ca, cb)
                if re.search(r"\d", clean):
                    checked.append({"table": ti, "row": ri, "column": ci, "en": ca, "zh": cb})
out = {"status": "PASS", "table_count": len(en), "numeric_cells_identical": len(checked),
       "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
       "source_validation": "EN146 measured table cells separately checked against current A0 report by audit_en_numeric.py; corresponding ZH numeric cells must match exactly, allowing localized Chinese separators in configuration seed lists only.",
       "checks": checked}
(HERE / ("audit_bilingual_numeric_published.json" if len(sys.argv) == 3 else "audit_bilingual_numeric.json")).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k:out[k] for k in ("status", "table_count", "numeric_cells_identical")}))
