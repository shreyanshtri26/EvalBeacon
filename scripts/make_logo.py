"""Build the web-ready logo files from the source artwork.

  uv run --with pillow python scripts/make_logo.py

Reads assets/source/logo-original.png (a large RGB image whose rounded square sits on a black background) and writes
  assets/logo.png     512x512, transparent outside the rounded square (README, social previews)
  assets/logo-64.png   64x64, same artwork (favicon and the dashboard header)
Pillow is only needed to run this script, not the project.
"""
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SRC = ASSETS / "source" / "logo-original.png"
SUPERSAMPLE = 4  # draw the mask large, then shrink: smooth anti-aliased corners


def rounded_mask(size: int, inset_frac: float = 0.021, radius_frac: float = 0.185) -> Image.Image:
    big = size * SUPERSAMPLE
    inset = round(big * inset_frac)
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle([inset, inset, big - inset - 1, big - inset - 1], radius=round(big * radius_frac), fill=255)
    return mask.resize((size, size), Image.LANCZOS)


def build(src: Image.Image, size: int) -> Image.Image:
    img = src.convert("RGB").resize((size, size), Image.LANCZOS).convert("RGBA")
    img.putalpha(ImageChops.multiply(img.getchannel("A"), rounded_mask(size)))
    return img


def main() -> None:
    src = Image.open(SRC)
    build(src, 512).save(ASSETS / "logo.png", optimize=True)
    build(src, 64).save(ASSETS / "logo-64.png", optimize=True)
    for name in ("logo.png", "logo-64.png"):
        print(f"{name}: {(ASSETS / name).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
