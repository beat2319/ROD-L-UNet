"""
Quick chip/mask visualizer. Usage:
  python validate_chips.py path/to/chip.npy
  python validate_chips.py path/to/mask.npy
  python validate_chips.py diagnose  -- print raw values from first mosaic+timestep
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = np.load(sys.argv[1])
print(f"Shape: {data.shape}  Dtype: {data.dtype}")

out_path = Path("./data/validation_visualizations")
out_path.mkdir(parents=True, exist_ok=True)
stem = Path(sys.argv[1]).stem

# --- Mask mode: 2D uint8 with values {0, 1, 255} ---
if data.ndim == 2 and data.dtype == np.uint8:
    vals, counts = np.unique(data, return_counts=True)
    class_names = {0: "canopy", 1: "mortality", 255: "other"}
    print("\nClass distribution:")
    for v, c in zip(vals, counts):
        label = class_names.get(v, f"class_{v}")
        print(f"  {v:>3d} ({label:>10s}): {c:>6d} pixels ({100*c/data.size:.1f}%)")

    cmap = ListedColormap(["#40B0A6", "#DEC79F", "#7f7f7f"])  # green, red, gray
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 255.5], cmap.N)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(f"Mask: {stem}", fontsize=10)
    ax.axis("off")
    # Legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color="#40B0A6", label="0 canopy"),
               mpatches.Patch(color="#DEC79F", label="1 mortality"),
               mpatches.Patch(color="#7f7f7f", label="255 other")]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path / f"{stem}.png")
    print(f"\nSaved to {out_path / f'{stem}.png'}")
    sys.exit(0)

# --- Chip mode: (T, C, H, W) SAR stack ---
chip = data
names = ["VV_dB", "VH_dB", "RVI"]
for c in range(chip.shape[1]):
    vals = chip[:, c][~np.isnan(chip[:, c])]
    nan_pct = np.isnan(chip[:, c]).sum() / chip[:, c].size * 100
    if len(vals) > 0:
        print(f"  {names[c]:5s}  min={vals.min():.2f}  mean={vals.mean():.2f}  max={vals.max():.2f}  "
              f"nan={nan_pct:.1f}%")
    else:
        print(f"  {names[c]:5s}  ALL NaN")

# Quick per-timestep summary
print("\nPer-timestep valid pixel count:")
for t in range(chip.shape[0]):
    valid = (~np.isnan(chip[t, 0])).sum()
    print(f"  T{t}: {valid}/{chip.shape[2]*chip.shape[3]} valid VV pixels")

fig, axes = plt.subplots(chip.shape[0], chip.shape[1], figsize=(12, 20))
cmaps = ["gray", "gray", "RdYlGn"]
vlims = [(-30, 10), (-30, 10), (0, 4)]

for t in range(chip.shape[0]):
    for c in range(chip.shape[1]):
        ax = axes[t, c]
        im = ax.imshow(chip[t, c], cmap=cmaps[c], vmin=vlims[c][0], vmax=vlims[c][1])
        ax.set_title(f"T{t} {names[c]}", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
fig.savefig(out_path / f"{stem}.png")
print(f"\nSaved to {out_path / f'{stem}.png'}")
