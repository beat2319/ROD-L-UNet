"""
Quick chip visualizer. Usage:
  python validate_chips.py path/to/chip.npy
  python validate_chips.py diagnose  -- print raw values from first mosaic+timestep
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

chip = np.load(sys.argv[1])
print(f"Shape: {chip.shape}  Dtype: {chip.dtype}")

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

plt.suptitle(sys.argv[1], fontsize=10)
plt.tight_layout()
plt.savefig("./data/validation_visualizations/fig.png")
