"""Assemble the slim Vercel bundle (no secrets, no tests) in OUT_DIR.

  uv run python scripts/stage_vercel.py <out_dir> && cd <out_dir> && vercel deploy --prod
"""
import shutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1])
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)

ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
for src in ("src/evalbeacon", "src/exporters", "src/collectors"):
    shutil.copytree(root / src, out / src, ignore=ignore)
(out / "src" / "evalbeacon" / "ui").mkdir(exist_ok=True)
shutil.copytree(root / "datasets", out / "datasets")
shutil.copytree(root / "assets", out / "assets", ignore=shutil.ignore_patterns("source"))
(out / "configs").mkdir()
shutil.copy(root / "configs/gate.json", out / "configs/gate.json")
shutil.copy(root / "deploy/app.py", out / "app.py")
shutil.copy(root / "deploy/requirements.txt", out / "requirements.txt")
shutil.copy(root / "deploy/vercel.json", out / "vercel.json")
(out / ".vercelignore").write_text(".env\n.env.*\nresults/\n")
print(f"staged {sum(1 for _ in out.rglob('*') if _.is_file())} files in {out}")
