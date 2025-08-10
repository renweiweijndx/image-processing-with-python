import cv2 as cv
import numpy as np
from pathlib import Path
import tifffile as tiff

# ===== User parameters =====
input_folder  = "/home/ren/Downloads/test_img/1-555"          # folder containing .lsm files
output_folder = "/home/ren/Downloads/test_img/denoised_tiffs"       # output folder for .tif
h = 8                                  # NLM strength (increase for stronger denoise)
templateWindowSize = 7                  # odd, e.g., 7
searchWindowSize   = 21                 # odd, e.g., 21
percentile_clip = (0.5, 99.5)           # robust scaling for 16-bit; set to None for min-max

# ===== Helpers =====
def to_uint8_robust(img16, prc=(0.5, 99.5)):
    """Scale uint16 image to uint8 using robust percentiles."""
    if prc is None:
        lo, hi = int(img16.min()), int(img16.max())
    else:
        lo, hi = np.percentile(img16, prc).astype(np.float32)
        if hi <= lo:  # degenerate
            lo, hi = float(img16.min()), float(img16.max())
    if hi == lo:
        return np.zeros_like(img16, dtype=np.uint8), (0, 1)
    scaled = np.clip((img16.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
    return scaled.astype(np.uint8), (lo, hi)

def from_uint8_rescale(img8, lo_hi):
    lo, hi = lo_hi
    if hi == lo:
        return np.zeros_like(img8, dtype=np.uint16)
    back = img8.astype(np.float32) * (hi - lo) / 255.0 + lo
    return np.clip(back, 0, 65535).astype(np.uint16)

def nlm_gray(img8):
    return cv.fastNlMeansDenoising(img8, None, h=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)

def nlm_color(img8c3):
    return cv.fastNlMeansDenoisingColored(img8c3, None, h=h, hColor=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)

def ensure_2d_or_3c(arr):
    """Return list of 2D planes (or 3-channel images) from common LSM/TIFF shapes."""
    # Accept shapes like (Z,Y,X), (T,Z,Y,X), (C,Y,X), (Y,X,C), (T,C,Z,Y,X) etc. We’ll iterate planes as 2D or 3-channel.
    a = arr
    # Move channels to last if a 'C' axis exists inferred by small size 2–4 and last-not-already
    # Best effort heuristic; for complex layouts prefer selecting first series in Z/T.
    if a.ndim >= 3:
        # Try to find channel axis by common dims 2..4
        ch_axis = None
        for ax in range(a.ndim):
            if a.shape[ax] in (2,3,4) and ax != a.ndim-1 and a.shape[-1] not in (3,4):
                ch_axis = ax
                break
        if ch_axis is not None:
            a = np.moveaxis(a, ch_axis, -1)

    planes = []
    if a.ndim == 2:               # (Y,X)
        planes = [a]
    elif a.ndim == 3:
        if a.shape[-1] in (3,4):  # (Y,X,C)
            planes = [a[..., :3]]
        else:                     # (N,Y,X)
            planes = [a[i, ...] for i in range(a.shape[0])]
    elif a.ndim >= 4:
        # Flatten leading dims except last 2 or last 3 if color
        if a.shape[-1] in (3,4):
            lead = int(np.prod(a.shape[:-3]))
            a = a.reshape((lead, a.shape[-3], a.shape[-2], a.shape[-1]))
            planes = [a[i, ...][..., :3] for i in range(a.shape[0])]
        else:
            lead = int(np.prod(a.shape[:-2]))
            a = a.reshape((lead, a.shape[-2], a.shape[-1]))
            planes = [a[i, ...] for i in range(a.shape[0])]
    return planes

# ===== Main batch =====
in_dir = Path(input_folder)
out_dir = Path(output_folder)
out_dir.mkdir(parents=True, exist_ok=True)

lsm_files = sorted([p for p in in_dir.glob("*.lsm")])
if not lsm_files:
    print("No .lsm files found in", in_dir.resolve())

for fp in lsm_files:
    print(f"Processing {fp.name} ...")
    with tiff.TiffFile(fp) as tf:
        # Use first series (typical for LSM)
        arr = tf.series[0].asarray()  # numpy array; dtype likely uint16
        # Denoise plane-by-plane
        denoised_planes = []
        planes = ensure_2d_or_3c(arr)

        for plane in planes:
            if plane.dtype == np.uint16:
                if plane.ndim == 2:
                    p8, lohi = to_uint8_robust(plane, percentile_clip)
                    d8 = nlm_gray(p8)
                    dout = from_uint8_rescale(d8, lohi)
                else:  # color 16-bit
                    # Split → scale each channel → merge → NLM → rescale per channel
                    chs = cv.split(plane)
                    p8s, lohis = zip(*(to_uint8_robust(c, percentile_clip) for c in chs))
                    merged8 = cv.merge(p8s)
                    d8c = nlm_color(merged8)
                    dchs = cv.split(d8c)
                    dout = cv.merge([from_uint8_rescale(dc, lh) for dc, lh in zip(dchs, lohis)])
                denoised_planes.append(dout)
            else:
                if plane.ndim == 2:
                    denoised_planes.append(nlm_gray(plane.astype(np.uint8)))
                else:
                    denoised_planes.append(nlm_color(plane[..., :3].astype(np.uint8)))

        # Stack back to a multipage TIFF (uint16 if original was 16-bit; else uint8)
        out_stack = np.stack(denoised_planes, axis=0)
        out_path = out_dir / (fp.stem + "_denoised.tif")
        # Save as BigTIFF if large
        tiff.imwrite(
            out_path,
            out_stack,
            bigtiff=True,
            compression=None,
        )
    print(f"Saved -> {out_path.name}")

print("All done.")
