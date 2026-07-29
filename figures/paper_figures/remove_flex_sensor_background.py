from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageFilter


SRC = Path(
    r"C:\Users\HAO\AppData\Local\Temp"
    r"\codex-clipboard-9f514ab7-184c-4c25-8ffb-a9af1c685e70.png"
)
OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "flex_sensor_cutout.png"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img = Image.open(SRC).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = arr.mean(axis=2)
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    sat = np.divide(maxc - minc, np.maximum(maxc, 1))

    # Sensor seeds: orange substrate plus inner dark traces. Border-connected
    # black pixels are the original background and are excluded.
    orange = (r > 110) & (g > 45) & (g < 210) & (b < 155) & (r > b + 25) & (sat > 0.18)
    dark = gray < 92
    dark_labels, _ = ndi.label(dark)
    border_ids = np.unique(
        np.concatenate(
            [
                dark_labels[0, :],
                dark_labels[-1, :],
                dark_labels[:, 0],
                dark_labels[:, -1],
            ]
        )
    )
    dark_inner = dark & ~np.isin(dark_labels, border_ids)
    midtone = (gray < 190) & (sat > 0.10) & ~((gray < 25) & np.isin(dark_labels, border_ids))

    seed = orange | dark_inner | midtone
    yy, xx = np.where(seed)
    x0, x1 = max(xx.min() - 8, 0), min(xx.max() + 9, arr.shape[1])
    y0, y1 = max(yy.min() - 8, 0), min(yy.max() + 9, arr.shape[0])

    local = seed[y0:y1, x0:x1]
    local = ndi.binary_closing(local, structure=np.ones((5, 5)), iterations=2)
    local = ndi.binary_dilation(local, structure=np.ones((3, 3)), iterations=1)

    labels, count = ndi.label(local)
    keep = np.zeros_like(local, dtype=bool)
    if count:
        sizes = np.bincount(labels.ravel())
        keep_label = 1 + np.argmax(sizes[1:])
        keep = labels == keep_label

    keep = ndi.binary_fill_holes(keep)
    pin_keep = (dark_inner[y0:y1, x0:x1] | orange[y0:y1, x0:x1]) & ndi.binary_dilation(
        keep, iterations=5
    )
    keep = keep | pin_keep
    keep = ndi.binary_closing(keep, structure=np.ones((3, 3)), iterations=1)

    mask = np.zeros(arr.shape[:2], dtype=bool)
    mask[y0:y1, x0:x1] = keep

    yy, xx = np.where(mask)
    pad = 10
    cx0, cx1 = max(xx.min() - pad, 0), min(xx.max() + pad + 1, arr.shape[1])
    cy0, cy1 = max(yy.min() - pad, 0), min(yy.max() + pad + 1, arr.shape[0])

    rgba = img.convert("RGBA")
    alpha = Image.fromarray((mask * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(radius=0.6))
    rgba.putalpha(alpha)
    rgba.crop((cx0, cy0, cx1, cy1)).save(OUT_FILE)
    print(OUT_FILE)


if __name__ == "__main__":
    main()
