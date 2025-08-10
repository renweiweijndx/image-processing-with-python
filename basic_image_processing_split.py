#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 20:00:39 2025

@author: ren
"""
# %% Split images into mono-channel TIFFs; group same channel in one folder

from pathlib import Path
import numpy as np
import tifffile
import cv2

# ---------- User settings ----------
in_folder    = r"/home/ren/Downloads/test_img/1-555"                             # folder with your ~10 images
out_root     = r"/home/ren/Downloads/test_img/1-555/split"                       # None -> save next to input folder
take_first_z = True                       # set False to save ALL Z-slices
take_first_t = True                       # set False to save ALL time points
valid_exts   = {".lsm", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
# -----------------------------------

in_dir  = Path(in_folder)
out_dir = Path(out_root) 

def load_with_axes(p: Path):
    """Return (array, axes_string). For non-LSM, synthesize axes."""
    if p.suffix.lower() == ".lsm":
        with tifffile.TiffFile(p) as tf:
            s = tf.series[0]
            arr = s.asarray()
            axes = s.axes  # e.g., 'TZCYX', 'ZCYX', 'CYX', 'YXS'
        return arr, axes
    else:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Unreadable image")
        if img.ndim == 2:
            # grayscale -> 'YX'
            return img, "YX"
        elif img.ndim == 3:
            # OpenCV gives HxWxC (BGR) -> 'YXC'
            return img, "YXC"
        else:
            raise ValueError(f"Unexpected ndim={img.ndim} for non-LSM")

def iter_channel_planes(arr, axes, take_first_z=True, take_first_t=True):
    """
    Yields tuples (ch_index0, z_index_or_None, t_index_or_None, plane2d).
    Reorders according to axes metadata.
    """
    axes = axes.upper()
    # Ensure Y and X exist
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"Axes missing Y/X: {axes}")

    # Bring to canonical order: (T?, Z?, Y, X, C?)
    order = []
    for ax in "TZYXC":
        if ax in axes:
            order.append(axes.index(ax))
    arr = np.moveaxis(arr, order, range(len(order)))
    # Now arr has axes in the subset order found within "TZYXC".
    # Build indices
    hasT = "T" in axes
    hasZ = "Z" in axes
    hasC = "C" in axes

    # Determine positions after reordering
    pos = {ax: i for i, ax in enumerate([a for a in "TZYXC" if a in axes])}
    # Build slicer
    Tn = arr.shape[pos["T"]] if hasT else 1
    Zn = arr.shape[pos["Z"]] if hasZ else 1
    Cn = arr.shape[pos["C"]] if hasC else 1

    t_range = [0] if (hasT and take_first_t) else range(Tn)
    z_range = [0] if (hasZ and take_first_z) else range(Zn)

    for t in (t_range if hasT else [None]):
        for z in (z_range if hasZ else [None]):
            for c in (range(Cn) if hasC else [0]):
                sl = [slice(None)] * arr.ndim
                if hasT: sl[pos["T"]] = t
                if hasZ: sl[pos["Z"]] = z
                if hasC: sl[pos["C"]] = c
                plane = arr[tuple(sl)]
                # After slicing, remaining axes must be Y,X (2D plane)
                # If extra singleton dims exist, squeeze:
                plane = np.squeeze(plane)
                # Sanity: enforce 2D YX
                if plane.ndim != 2:
                    raise ValueError(f"Got non-2D plane (ndim={plane.ndim}) for axes {axes}")
                yield (c, z if hasZ else None, t if hasT else None, plane)

# ---- Process all images ----
for p in sorted(in_dir.iterdir()):
    if p.suffix.lower() not in valid_exts: 
        continue
    try:
        arr, axes = load_with_axes(p)
    except Exception as e:
        print(f"Skipping {p.name}: {e}")
        continue

    try:
        planes = list(iter_channel_planes(arr, axes, take_first_z, take_first_t))
    except Exception as e:
        print(f"Skipping {p.name}: {e}")
        continue

    if not planes:
        print(f"Skipping {p.name}: no channels found.")
        continue

    stem = p.stem
    # Save each plane into its channel folder Ch1, Ch2, ...
    for ch_idx, z_idx, t_idx, plane in planes:
        ch_folder = out_dir / f"Ch{ch_idx+1}"
        ch_folder.mkdir(parents=True, exist_ok=True)
        fname = stem
        if t_idx is not None: fname += f"_t{t_idx:03d}"
        if z_idx is not None: fname += f"_z{z_idx:03d}"
        fname += f"_ch{ch_idx+1}.tif"
        tifffile.imwrite(str(ch_folder / fname), plane)
    # Summary
    used_ch = sorted({c for c,_,_,_ in planes})
    print(f"{p.name}: saved {len(planes)} plane(s) into " +
          ", ".join([f'Ch{c+1}' for c in used_ch]))

print("Done.")