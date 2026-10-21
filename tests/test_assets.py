"""Asset integrity: a file's extension must match its real format (a PNG named .svg breaks browsers and the README)."""
import re
import xml.dom.minidom
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.evalbeacon.api import router

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assets():
    return [p for p in ASSETS.rglob("*") if p.is_file()]


@pytest.mark.parametrize("path", _assets(), ids=lambda p: str(p.relative_to(ROOT)))
def test_extension_matches_content(path):
    head = path.read_bytes()[:16]
    if path.suffix == ".svg":
        assert not head.startswith(PNG_MAGIC) and not head.startswith(b"\xff\xd8"), f"{path.name} is a raster image, not an SVG"
        xml.dom.minidom.parse(str(path))  # must be well-formed XML
    elif path.suffix == ".png":
        assert head.startswith(PNG_MAGIC), f"{path.name} is not a PNG"
    elif path.suffix in (".jpg", ".jpeg"):
        assert head.startswith(b"\xff\xd8"), f"{path.name} is not a JPEG"


def test_shipped_logos_are_small_enough_for_web_use():
    assert (ASSETS / "logo.png").stat().st_size < 400_000  # README and social previews
    assert (ASSETS / "logo-64.png").stat().st_size < 30_000  # favicon and header


def test_every_asset_referenced_by_readme_and_ui_exists_and_is_served():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # every image the README shows, as HTML <img src> or Markdown ![alt](path); local ones must exist and be real PNGs
    refs = re.findall(r'<img[^>]+src="([^"]+)"', readme) + re.findall(r"!\[[^\]]*\]\(([^)\s]+)", readme)
    local = [r for r in refs if not r.startswith(("http://", "https://", "data:"))]
    assert any("logo" in r for r in local), "the README should show the logo"
    for ref in local:
        path = ROOT / ref.removeprefix("./")
        assert path.is_file(), f"README references a missing file: {ref}"
        assert path.read_bytes().startswith(PNG_MAGIC), f"README image is not a real PNG: {ref}"
    ui = (ROOT / "src/evalbeacon/ui/index.html").read_text(encoding="utf-8")
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    for url in sorted(set(re.findall(r'(?:href|src)="(/logo[^"]*)"', ui))):
        r = client.get(url)
        assert r.status_code == 200, url
        assert r.headers["content-type"] == "image/png" and r.content.startswith(PNG_MAGIC), url


def test_no_dangling_references_to_the_removed_svg():
    for p in (ROOT / "README.md", ROOT / "src/evalbeacon/ui/index.html", ROOT / "src/evalbeacon/api.py"):
        assert "logo.svg" not in p.read_text(encoding="utf-8"), p.name
