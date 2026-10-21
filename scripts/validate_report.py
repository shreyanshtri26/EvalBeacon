"""Validate EvalBeacon JSON files against schemas/report.schema.json (needs the `dev` extra: jsonschema).

  python scripts/validate_report.py results/report.json            # a full report
  python scripts/validate_report.py --kind run results/baseline.json
Exit 0 when valid, 1 with the first errors listed, 64 on usage errors.
"""
import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "report.schema.json").read_text(encoding="utf-8"))


def validate(obj: object, kind: str = "report") -> list[str]:
    v = Draft202012Validator({"$schema": SCHEMA["$schema"], "$defs": SCHEMA["$defs"], "$ref": f"#/$defs/{kind}"})
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:120]}" for e in v.iter_errors(obj)]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["report", "run", "gate", "drift"], default="report")
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args(argv)
    bad = 0
    for p in a.paths:
        try:
            errs = validate(json.loads(Path(p).read_text(encoding="utf-8")), a.kind)
        except (OSError, ValueError) as e:
            errs = [f"cannot read: {e}"]
        print(f"{'OK  ' if not errs else 'FAIL'} {p}")
        for e in errs[:10]:
            print(f"     {e}")
        bad += bool(errs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
