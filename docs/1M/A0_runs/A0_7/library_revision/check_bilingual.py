"""Read-only structural and literal parity checks for the A0 evidence libraries.

The input files are never modified.  Exit 0 means automated parity checks found
no mismatches, not that scientific claims or translations have been approved.
Only the two explicitly supplied files are read; snapshot directories are not
searched.  Local links are resolved at their eventual docs/1M destination.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?%?(?!(?:[A-Za-z0-9_]|\.\d))")
DIGEST = re.compile(r"(?<![a-fA-F0-9])[a-fA-F0-9]{64}(?![a-fA-F0-9])")
LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\((<[^>]+>|[^)\n]+)\)")
INLINE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
OLD_RESULT = re.compile(r"(?:[FXYZ]\d+[A-Za-z0-9_-]*_runs[/\\]|(?:exports_x\d*|metrics_[xyz]\d+)[/\\]|X4GD_full_s|hnsw_y2_eval\.bin|(?:[FXYZ]\d+)_REPORT\.json)", re.I)
STALE_CLAIM = re.compile(r"\bBM25\b|(?<![\d.])2\.98(?![\d.])|(?<!\d)3,?004(?!\d)", re.I)
MOJIBAKE = re.compile(r"\ufffd|[\ud800-\udfff]|[\ue000-\uf8ff]|锟斤拷|Ã.|Â[\x80-\xff]|â(?:€|†|‡)|鍏朵|鎵ц|鍐荤|鏁版|绱㈠|鈥|涓夌|鏂囦")


def squash(text: str) -> str:
    return " ".join(text.split())


def no_space(text: str) -> str:
    return re.sub(r"\s+", "", text)


def split_table_row(line: str) -> list[str]:
    """Split unescaped pipes outside inline code; preserve empty cells."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    cells, current, code, i = [], [], "", 0
    while i < len(body):
        char = body[i]
        if char == "\\" and i + 1 < len(body):
            current.extend(body[i:i + 2]); i += 2; continue
        if char == "`":
            end = i + 1
            while end < len(body) and body[end] == "`":
                end += 1
            marker = body[i:end]
            code = "" if code == marker else (marker if not code else code)
            current.append(marker); i = end; continue
        if char == "|" and not code:
            cells.append("".join(current).strip()); current = []
        else:
            current.append(char)
        i += 1
    cells.append("".join(current).strip())
    return cells


def parse(path: Path, destination: Path) -> dict:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="strict")
    lines = text.splitlines()
    outside, sections, tables, fences, fence = [], [], [], [], None
    table = None
    for number, line in enumerate(lines, 1):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if marker and marker[1][0] == fence["marker"][0] and len(marker[1]) >= len(fence["marker"]) and not marker[2].strip():
                fences.append(fence); fence = None
            else:
                fence["lines"].append(line)
            continue
        if marker:
            fence = {"line": number, "marker": marker[1], "language": marker[2].strip(), "lines": []}
            table = None
            continue
        outside.append((number, line))
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            prefix = re.match(r"(\d+(?:\.\d+)*)(?:[.)\s]|$)", heading[2])
            sections.append({"line": number, "level": len(heading[1]), "number": prefix[1] if prefix else None})
        if line.strip().startswith("|"):
            if table is None:
                table = {"line": number, "rows": []}; tables.append(table)
            cells = split_table_row(line)
            if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                continue
            table["rows"].append({"line": number, "cells": len(cells), "numbers": [NUMBER.findall(cell) for cell in cells]})
        else:
            table = None
    outside_text = "\n".join(line for _, line in outside)
    math = []
    display_pattern = re.compile(r"\$\$(.*?)\$\$|\\\[(.*?)\\\]|\\begin\{((?:equation|align|gather)\*?)\}(.*?)\\end\{\3\}", re.S)
    for match in display_pattern.finditer(outside_text):
        math.append(no_space(match[0]))
    inline = [squash(match[2]) for match in INLINE.finditer(outside_text) if match[2].isascii()]
    commands = []
    for item in fences:
        # Human comments can be translated. Preserve all executable/text lines.
        body = [squash(line) for line in item["lines"] if line.strip() and not re.match(r"\s*(?:#|//)", line)]
        commands.append({"language": item["language"].lower(), "body": body})
    links, broken = [], []
    for match in LINK.finditer(text):
        target = match[1].strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        else:
            target = re.sub(r"\s+[\"'].*[\"']$", "", target)
        links.append(target)
        # Windows drive paths are local despite their parsed single-letter scheme.
        scheme = urlsplit(target).scheme
        if (scheme and not re.match(r"^[A-Za-z]:[/\\]", target)) or target.startswith("#"):
            continue
        decoded = unquote(target.split("#", 1)[0].split("?", 1)[0])
        resolved = (destination / decoded).resolve()
        if not resolved.exists():
            broken.append({"target": target, "resolved": str(resolved)})
    flags = re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*(?:=[^\s`]+)?", text)
    warnings = []
    for number, line in enumerate(lines, 1):
        for kind, pattern in [("stale_other_series_result", OLD_RESULT), ("legacy_claim_requires_manual_review", STALE_CLAIM), ("suspected_unicode_corruption", MOJIBAKE)]:
            found = list(pattern.finditer(line))
            if found:
                warnings.append({"kind": kind, "line": number, "matches": [m[0] for m in found], "context": line[:400]})
        controls = [f"U+{ord(c):04X}" for c in line if ord(c) < 32 and c != "\t"]
        if controls:
            warnings.append({"kind": "control_characters", "line": number, "matches": controls})
    # Compare Arabic-number facts in prose separately from commands/formulas/URLs.
    prose = display_pattern.sub("", outside_text)
    prose = LINK.sub(lambda m: m[0].split("](", 1)[0] + "]", prose)
    prose = INLINE.sub("", prose)
    prose = "\n".join(line for line in prose.splitlines() if not line.strip().startswith("|"))
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "sections": sections, "tables": tables, "display_math": math,
            "inline_ascii_literals": inline, "fenced_commands": commands,
            "flags": flags, "digests": DIGEST.findall(text), "links": links,
            "prose_numbers": NUMBER.findall(prose), "broken_links": broken,
            "warnings": warnings, "unclosed_fence": fence is not None}


def compare_sequence(name: str, left: list, right: list) -> list[dict]:
    if left == right:
        return []
    a = [json.dumps(x, sort_keys=True, ensure_ascii=False) for x in left]
    b = [json.dumps(x, sort_keys=True, ensure_ascii=False) for x in right]
    changes = []
    for kind, i, j, x, y in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if kind != "equal":
            changes.append({"change": kind, "en_range": [i, j], "zh_range": [x, y], "en": left[i:j], "zh": right[x:y]})
    return [{"check": name, "en_count": len(left), "zh_count": len(right), "differences": changes}]


def review(en: Path, zh: Path, destination: Path) -> dict:
    docs = {"en": parse(en, destination), "zh": parse(zh, destination)}
    left, right = docs["en"], docs["zh"]
    mismatches = []
    sections = lambda d: [{"level": x["level"], "number": x["number"]} for x in d["sections"]]
    table_signature = lambda d: [[{"cells": r["cells"], "numbers": r["numbers"]} for r in t["rows"]] for t in d["tables"]]
    mismatches += compare_sequence("section_structure", sections(left), sections(right))
    mismatches += compare_sequence("ordered_table_numeric_cells", table_signature(left), table_signature(right))
    for name in ["display_math", "inline_ascii_literals", "fenced_commands", "flags", "digests", "links", "prose_numbers"]:
        mismatches += compare_sequence(name, left[name], right[name])
    warnings = [{"language": lang, **item} for lang, doc in docs.items() for item in doc["warnings"]]
    broken = [{"language": lang, **item} for lang, doc in docs.items() for item in doc["broken_links"]]
    unclosed = [lang for lang, doc in docs.items() if doc["unclosed_fence"]]
    return {"schema_version": "a0.bilingual_literal_review.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Automated literal/structural checks only. Scientific correctness, contextual exceptions and translation accuracy require human/agent review.",
            "read_only_inputs": True, "link_destination": str(destination.resolve()),
            "status": "REVIEW_REQUIRED" if mismatches or warnings or broken or unclosed else "AUTOMATED_CHECKS_CLEAR",
            "inputs": {lang: {key: doc[key] for key in ["path", "sha256", "bytes"]} for lang, doc in docs.items()},
            "counts": {lang: {key: len(doc[key]) for key in ["sections", "tables", "display_math", "inline_ascii_literals", "fenced_commands", "flags", "digests", "links", "prose_numbers"]} for lang, doc in docs.items()},
            "mismatches": mismatches, "review_flags": warnings, "broken_links": broken, "unclosed_fences": unclosed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en", type=Path, required=True)
    parser.add_argument("--zh", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[3], help="Final directory used for resolving links; defaults to docs/1M")
    parser.add_argument("--out", type=Path, help="Optional new JSON review artifact; stdout if omitted")
    args = parser.parse_args()
    inputs = {args.en.resolve(), args.zh.resolve()}
    if args.out and args.out.resolve() in inputs:
        parser.error("--out must not overwrite either input document")
    try:
        result = review(args.en, args.zh, args.destination)
    except (OSError, UnicodeError) as error:
        result = {"schema_version": "a0.bilingual_literal_review.v1", "status": "INPUT_ERROR", "error": str(error)}
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(json.dumps({"status": result["status"], "review": str(args.out.resolve()), "mismatch_categories": len(result.get("mismatches", [])), "review_flags": len(result.get("review_flags", [])), "broken_links": len(result.get("broken_links", []))}))
    else:
        sys.stdout.write(rendered)
    return 0 if result["status"] == "AUTOMATED_CHECKS_CLEAR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
