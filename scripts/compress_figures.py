"""Convert the photo-heavy qualitative PNGs to JPEG (quality 88) so the PDF stays small; main.tex prefers .jpg if present."""
from pathlib import Path

from PIL import Image

for f in sorted(Path("figures").glob("fig_*.png")):
    Image.open(f).convert("RGB").save(f.with_suffix(".jpg"), quality=88, optimize=True)
    print("compressed", f.name)
