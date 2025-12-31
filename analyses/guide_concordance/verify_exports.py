"""Check export structure and prepare full-resolution inspection crops."""

from pathlib import Path
import json
import xml.etree.ElementTree as ET
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "figures/supplement/S25"
WORK = ROOT / "work/s25_qc"


def main():
    svg = ET.parse(FIGURE / "S25.svg")
    assert not svg.findall(".//{http://www.w3.org/2000/svg}image")
    assert b"/Subtype /Image" not in (FIGURE / "S25.pdf").read_bytes()
    for label, path in [("png", FIGURE / "S25.png"), ("pdf", WORK / "pdf_render.png"), ("svg", WORK / "svg_render.png")]:
        image = Image.open(path)
        w, h = image.size
        image.crop((int(w * .45), int(h * .79), int(w * .97), int(h * .97))).save(WORK / f"{label}_footer_crop.png")
        image.crop((0, int(h * .3), int(w * .47), int(h * .57))).save(WORK / f"{label}_cis_crop.png")
        image.crop((0, int(h * .605), int(w * .49), int(h * .87))).save(WORK / f"{label}_molecular_crop.png")
        image.resize((780, 960)).save(WORK / f"{label}_final_size.png")
    print(json.dumps({"svg_raster_layers": 0, "pdf_image_objects": 0, "crop_scale": "native pixels; no resampling", "final_size_pixels": [780, 960]}))


if __name__ == "__main__":
    main()
