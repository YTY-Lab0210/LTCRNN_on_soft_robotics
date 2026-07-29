from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageFilter


SRC = Path(
    r"C:\Users\HAO\AppData\Local\Temp"
    r"\codex-clipboard-03bf0b12-910c-4c24-a0a2-9130019bec75.png"
)
OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "soft_finger_cutout.png"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img = Image.open(SRC).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    gray = arr.mean(axis=2)

    # Bright soft-finger body is used as the seed. The dark inner slots are
    # restored by filling the silhouette, so they stay visible in the cutout.
    seed = gray > 150
    seed = ndi.binary_closing(seed, structure=np.ones((5, 5)), iterations=2)
    seed = ndi.binary_opening(seed, structure=np.ones((2, 2)), iterations=1)

    labels, count = ndi.label(seed)
    if count == 0:
        raise RuntimeError("No foreground detected")

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    main_component = labels == sizes.argmax()

    mask = ndi.binary_dilation(main_component, structure=np.ones((5, 5)), iterations=2)
    mask = ndi.binary_closing(mask, structure=np.ones((7, 7)), iterations=2)
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_erosion(mask, structure=np.ones((3, 3)), iterations=1)
    mask = ndi.binary_dilation(mask, structure=np.ones((3, 3)), iterations=1)

    labels, count = ndi.label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    mask = labels == sizes.argmax()

    alpha = Image.fromarray((mask * 255).astype("uint8")).filter(
        ImageFilter.GaussianBlur(radius=0.45)
    )
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)

    ys, xs = np.where(mask)
    pad = 4
    x0 = max(xs.min() - pad, 0)
    x1 = min(xs.max() + pad + 1, arr.shape[1])
    y0 = max(ys.min() - pad, 0)
    y1 = min(ys.max() + pad + 1, arr.shape[0])

    rgba.crop((x0, y0, x1, y1)).save(OUT_FILE)
    print(OUT_FILE)


if __name__ == "__main__":
    main()
