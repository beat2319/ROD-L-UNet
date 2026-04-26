"""
Evaluate trained checkpoints and visualize predicted segmentation maps.

By default this script:
1. Uses the manifest at ROD-TRAINING/manifest.csv
2. Evaluates the validation split (the current evaluation split in the manifest)
3. Auto-selects the two newest runs that contain checkpoints
4. Saves summary metrics plus a limited number of side-by-side PNG panels

Examples:
    python3 ROD-TRAINING/evaluate_visualize.py
    python3 ROD-TRAINING/evaluate_visualize.py --split eval
    python3 ROD-TRAINING/evaluate_visualize.py \
        --models ROD-TRAINING/runs/20260424_062623/best_model.pt \
                 ROD-TRAINING/runs/20260424_065735/best_model.pt
    python3 ROD-TRAINING/evaluate_visualize.py --max-visualizations -1
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from dataset import S1ChangeDataset
from metrics import ChangeDetectionMetrics
from models import S1ChangeDetector


IGNORE_INDEX = 255
DEFAULT_GRID_COLUMNS = 3


class _ILocAccessor:
    """Small pandas-like adapter so S1ChangeDataset can use manifest rows."""

    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.rows[idx]


class ManifestRecords:
    """Minimal DataFrame-like wrapper required by S1ChangeDataset."""

    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows
        self.iloc = _ILocAccessor(self.rows)

    def reset_index(self, drop: bool = True) -> "ManifestRecords":
        return self

    def __len__(self) -> int:
        return len(self.rows)


@dataclass
class ModelBundle:
    name: str
    checkpoint_path: Path
    model: torch.nn.Module
    metrics: ChangeDetectionMetrics
    input_channels: int


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints and save segmentation-map visualizations.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(script_dir / "manifest.csv"),
        help="Path to manifest CSV.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Manifest split to evaluate. Use --split eval if your manifest has an eval split.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=str(script_dir / "runs"),
        help="Directory containing timestamped training runs.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Checkpoint files or run directories. If omitted, auto-select newest runs.",
    )
    parser.add_argument(
        "--num-newest-models",
        type=int,
        default=2,
        help="How many newest checkpoints to auto-select when --models is omitted.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for evaluation: auto, cuda, cpu, or a specific torch device string.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size used for forward passes. Lower this if CUDA runs out of memory.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision during inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=100,
        help="Number of sample PNGs to save. Use -1 to save every sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(script_dir / "eval_outputs" / timestamp),
        help="Directory for metrics and visualization PNGs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N batches.",
    )
    return parser.parse_args()


def resolve_existing_path(path_str: str, fallback_base: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    fallback_candidate = (fallback_base / path).resolve()
    return fallback_candidate


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return path.resolve()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device_arg)


def read_manifest_rows(manifest_path: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split != "all" and row.get("split") != split:
                continue

            normalized = dict(row)
            for key in ("chip_path", "mask_path"):
                resolved = (manifest_path.parent / normalized[key]).resolve()
                normalized[key] = str(resolved)
            rows.append(normalized)

    return rows


def extract_epoch_number(path: Path) -> int:
    match = re.search(r"model_epoch(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def choose_checkpoint_for_run(run_dir: Path) -> Path | None:
    best_model = run_dir / "best_model.pt"
    if best_model.exists():
        return best_model

    epoch_models = sorted(
        run_dir.glob("model_epoch*.pt"),
        key=extract_epoch_number,
        reverse=True,
    )
    if epoch_models:
        return epoch_models[0]

    return None


def discover_newest_checkpoints(runs_dir: Path, count: int) -> list[Path]:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    selected: list[Path] = []
    run_dirs = sorted(
        [path for path in runs_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )

    for run_dir in run_dirs:
        checkpoint = choose_checkpoint_for_run(run_dir)
        if checkpoint is not None:
            selected.append(checkpoint.resolve())
        if len(selected) == count:
            break

    return selected


def resolve_model_paths(
    model_args: list[str] | None,
    runs_dir: Path,
    fallback_base: Path,
    count: int,
) -> list[Path]:
    if model_args:
        resolved: list[Path] = []
        for value in model_args:
            candidate = resolve_existing_path(value, fallback_base)
            if candidate.is_dir():
                checkpoint = choose_checkpoint_for_run(candidate)
                if checkpoint is None:
                    raise FileNotFoundError(
                        f"No checkpoint found in run directory: {candidate}",
                    )
                resolved.append(checkpoint.resolve())
            elif candidate.is_file():
                resolved.append(candidate.resolve())
            else:
                raise FileNotFoundError(f"Model path does not exist: {candidate}")
        return resolved

    discovered = discover_newest_checkpoints(runs_dir, count)
    if len(discovered) < count:
        raise FileNotFoundError(
            f"Requested {count} newest checkpoints, but only found {len(discovered)} "
            f"under {runs_dir}",
        )
    return discovered


def clean_state_dict(raw_state: dict) -> dict[str, torch.Tensor]:
    if "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
        raw_state = raw_state["state_dict"]

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def infer_model_kwargs(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    head_weight = state_dict.get("decoder.head.weight")
    if head_weight is None:
        raise KeyError("Checkpoint missing decoder.head.weight; cannot infer class count.")

    stem_weight = state_dict.get("encoder.spatial.stem.0.weight")
    if stem_weight is None:
        raise KeyError(
            "Checkpoint missing encoder.spatial.stem.0.weight; "
            "cannot infer SAR input channel count.",
        )

    lstm_gate_weight = state_dict.get("encoder.lstms.0.cell.gates.weight")
    if lstm_gate_weight is None:
        raise KeyError(
            "Checkpoint missing encoder.lstms.0.cell.gates.weight; "
            "cannot infer temporal encoding size.",
        )

    num_classes = int(head_weight.shape[0])
    input_channels = int(stem_weight.shape[1])
    temporal_encoding_dim = int(lstm_gate_weight.shape[1] - (256 + 256))
    if temporal_encoding_dim < 0:
        raise ValueError(
            "Inferred a negative temporal encoding dimension from checkpoint shapes.",
        )
    if input_channels not in (2, 3):
        raise ValueError(
            f"Expected checkpoint input channels to be 2 or 3, got {input_channels}.",
        )

    return {
        "num_classes": num_classes,
        "input_channels": input_channels,
        "temporal_encoding_dim": temporal_encoding_dim,
    }


def checkpoint_display_name(checkpoint_path: Path) -> str:
    if checkpoint_path.name == "best_model.pt":
        return checkpoint_path.parent.name

    if checkpoint_path.parent.name == "runs":
        return checkpoint_path.stem

    return f"{checkpoint_path.parent.name}_{checkpoint_path.stem}"


def load_model_bundle(checkpoint_path: Path, device: torch.device) -> ModelBundle:
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    raw_state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = clean_state_dict(raw_state)
    model_kwargs = infer_model_kwargs(state_dict)

    model = S1ChangeDetector(
        checkpoint_path=None,
        num_classes=model_kwargs["num_classes"],
        input_channels=model_kwargs["input_channels"],
        temporal_encoding_dim=model_kwargs["temporal_encoding_dim"],
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return ModelBundle(
        name=checkpoint_display_name(checkpoint_path),
        checkpoint_path=checkpoint_path,
        model=model,
        metrics=ChangeDetectionMetrics(
            num_classes=model_kwargs["num_classes"],
            ignore_index=IGNORE_INDEX,
        ),
        input_channels=model_kwargs["input_channels"],
    )


def robust_scale(channel: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    values = channel[valid_mask]
    if values.size == 0:
        values = channel.reshape(-1)

    low, high = np.percentile(values, [2.0, 98.0])
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return np.zeros(channel.shape, dtype=np.uint8)

    scaled = np.clip((channel - low) / (high - low), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def make_context_rgb(chip: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    latest = chip[-1]
    earliest = chip[0]

    vv_latest = latest[0]
    vh_latest = latest[1] if latest.shape[0] > 1 else latest[0]
    delta = np.sqrt(((latest - earliest) ** 2).sum(axis=0))

    rgb = np.stack(
        [
            robust_scale(vv_latest, valid_mask),
            robust_scale(vh_latest, valid_mask),
            robust_scale(delta, valid_mask),
        ],
        axis=-1,
    )
    rgb[~valid_mask] = np.array([96, 96, 96], dtype=np.uint8)
    return rgb


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    rgb[mask == 0] = np.array([12, 12, 12], dtype=np.uint8)
    rgb[mask == 1] = np.array([220, 42, 42], dtype=np.uint8)
    rgb[mask == IGNORE_INDEX] = np.array([140, 140, 140], dtype=np.uint8)
    return rgb


def error_map_to_rgb(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    rgb = np.zeros(target.shape + (3,), dtype=np.uint8)
    ignore = target == IGNORE_INDEX
    valid = ~ignore

    true_positive = valid & (pred == 1) & (target == 1)
    false_positive = valid & (pred == 1) & (target == 0)
    false_negative = valid & (pred == 0) & (target == 1)
    true_negative = valid & (pred == 0) & (target == 0)

    rgb[true_negative] = np.array([12, 12, 12], dtype=np.uint8)
    rgb[true_positive] = np.array([44, 186, 88], dtype=np.uint8)
    rgb[false_positive] = np.array([244, 159, 54], dtype=np.uint8)
    rgb[false_negative] = np.array([196, 76, 201], dtype=np.uint8)
    rgb[ignore] = np.array([140, 140, 140], dtype=np.uint8)
    return rgb


def binary_segmentation_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    valid = target != IGNORE_INDEX
    pred_valid = pred[valid]
    target_valid = target[valid]

    if target_valid.size == 0:
        return {
            "valid_pixels": 0,
            "pred_positive_pixels": 0,
            "target_positive_pixels": 0,
            "iou_class_1": 0.0,
            "f1_class_1": 0.0,
            "precision_class_1": 0.0,
            "recall_class_1": 0.0,
        }

    tp = int(np.logical_and(pred_valid == 1, target_valid == 1).sum())
    fp = int(np.logical_and(pred_valid == 1, target_valid == 0).sum())
    fn = int(np.logical_and(pred_valid == 0, target_valid == 1).sum())

    union = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn

    precision = tp / precision_den if precision_den > 0 else 0.0
    recall = tp / recall_den if recall_den > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "valid_pixels": int(valid.sum()),
        "pred_positive_pixels": int((pred_valid == 1).sum()),
        "target_positive_pixels": int((target_valid == 1).sum()),
        "iou_class_1": tp / union if union > 0 else 0.0,
        "f1_class_1": f1,
        "precision_class_1": precision,
        "recall_class_1": recall,
    }


def disagreement_pixels(predictions: list[np.ndarray], target: np.ndarray) -> int:
    valid = target != IGNORE_INDEX
    if len(predictions) < 2 or valid.sum() == 0:
        return 0

    stack = np.stack([pred[valid] for pred in predictions], axis=0)
    return int(np.any(stack != stack[0], axis=0).sum())


def safe_sample_stem(index: int, row: dict[str, str]) -> str:
    pieces = [
        f"{index:05d}",
        row.get("label", "sample"),
        row.get("island", "unknown"),
        row.get("date", "unknown"),
        row.get("patch_id", "0"),
    ]
    text = "_".join(str(piece) for piece in pieces)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def truncate_text(text: str, limit: int = 42) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3]}..."


def save_panel_grid(
    output_path: Path,
    sample_title: str,
    panels: list[np.ndarray],
    titles: list[str],
    ncols: int = DEFAULT_GRID_COLUMNS,
) -> None:
    cell_h, cell_w = panels[0].shape[:2]
    padding = 12
    header_h = 34
    title_h = 20
    rows = math.ceil(len(panels) / ncols)
    canvas_w = padding + ncols * (cell_w + padding)
    canvas_h = header_h + padding + rows * (title_h + cell_h + padding)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 8), truncate_text(sample_title, limit=96), fill=(240, 240, 240))

    for idx, (panel, title) in enumerate(zip(panels, titles)):
        row = idx // ncols
        col = idx % ncols
        left = padding + col * (cell_w + padding)
        top = header_h + padding + row * (title_h + cell_h + padding)

        draw.text((left, top - title_h + 2), truncate_text(title), fill=(220, 220, 220))
        canvas.paste(Image.fromarray(panel), (left, top))

    canvas.save(output_path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    manifest_path = resolve_existing_path(args.manifest, script_dir)
    runs_dir = resolve_existing_path(args.runs_dir, script_dir)
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)
    use_amp = device.type == "cuda" and not args.no_amp
    if use_amp:
        print("Using CUDA automatic mixed precision for inference.", flush=True)

    model_paths = resolve_model_paths(
        model_args=args.models,
        runs_dir=runs_dir,
        fallback_base=script_dir,
        count=args.num_newest_models,
    )
    print("Selected checkpoints:", flush=True)
    for checkpoint in model_paths:
        print(f"  - {checkpoint}", flush=True)

    rows = read_manifest_rows(manifest_path, args.split)
    if not rows:
        raise ValueError(
            f"No rows found for split '{args.split}' in manifest {manifest_path}",
        )
    print(f"Loaded {len(rows)} samples from split '{args.split}'.", flush=True)

    bundles = []
    for path in model_paths:
        try:
            bundles.append(load_model_bundle(path, device))
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint {path}") from exc

    print("Loaded models:", flush=True)
    for bundle in bundles:
        channel_label = "VV/VH/RVI" if bundle.input_channels == 3 else "VV/VH"
        print(
            f"  - {bundle.name} ({bundle.input_channels} channels: {channel_label})",
            flush=True,
        )

    input_channel_counts = {bundle.input_channels for bundle in bundles}
    if len(input_channel_counts) != 1:
        raise ValueError(
            "Cannot evaluate checkpoints with different SAR input channel counts "
            "in one run. Run 2-channel and 3-channel checkpoints separately.",
        )
    input_channels = input_channel_counts.pop()

    dataset = S1ChangeDataset(
        ManifestRecords(rows),
        augment=False,
        input_channels=input_channels,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    per_sample_rows: list[dict[str, object]] = []
    visualized = 0
    seen = 0

    with torch.inference_mode():
        for batch_idx, (chip, mask, doy, valid) in enumerate(loader, start=1):
            chip_device = chip.to(device, non_blocking=device.type == "cuda")
            mask_device = mask.to(device, non_blocking=device.type == "cuda")
            doy_device = doy.to(device, non_blocking=device.type == "cuda")

            preds_per_model: list[np.ndarray] = []

            for bundle in bundles:
                try:
                    amp_context = (
                        torch.amp.autocast("cuda") if use_amp else nullcontext()
                    )
                    with amp_context:
                        logits = bundle.model(chip_device, doy_device)
                    bundle.metrics.update(logits, mask_device)
                    preds_per_model.append(
                        logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8),
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    raise RuntimeError(
                        "CUDA ran out of memory during evaluation. "
                        "Try rerunning with --batch-size 1; if it still fails, "
                        "add --device cpu for a slow but memory-safer check.",
                    ) from exc
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed while evaluating model '{bundle.name}' "
                        f"on batch {batch_idx} after {seen} samples.",
                    ) from exc

            chip_np = chip.numpy()
            mask_np = mask.numpy().astype(np.uint8)
            valid_np = valid.numpy().astype(bool)
            batch_size = chip_np.shape[0]

            for batch_offset in range(batch_size):
                row = rows[seen + batch_offset]
                target = mask_np[batch_offset]
                predictions = [pred_batch[batch_offset] for pred_batch in preds_per_model]

                sample_row: dict[str, object] = {
                    "sample_index": seen + batch_offset,
                    "chip_path": row["chip_path"],
                    "mask_path": row["mask_path"],
                    "label": row.get("label", ""),
                    "split": row.get("split", ""),
                    "island": row.get("island", ""),
                    "year": row.get("year", ""),
                    "date": row.get("date", ""),
                    "patch_id": row.get("patch_id", ""),
                    "disagreement_pixels": disagreement_pixels(predictions, target),
                }

                for model_idx, (bundle, pred) in enumerate(zip(bundles, predictions), start=1):
                    metrics = binary_segmentation_metrics(pred, target)
                    prefix = f"model_{model_idx}"
                    sample_row[f"{prefix}_name"] = bundle.name
                    sample_row[f"{prefix}_iou_class_1"] = metrics["iou_class_1"]
                    sample_row[f"{prefix}_f1_class_1"] = metrics["f1_class_1"]
                    sample_row[f"{prefix}_precision_class_1"] = metrics["precision_class_1"]
                    sample_row[f"{prefix}_recall_class_1"] = metrics["recall_class_1"]
                    sample_row[f"{prefix}_pred_positive_pixels"] = metrics["pred_positive_pixels"]

                sample_row["valid_pixels"] = int((target != IGNORE_INDEX).sum())
                sample_row["target_positive_pixels"] = int((target == 1).sum())
                per_sample_rows.append(sample_row)

                should_visualize = (
                    args.max_visualizations < 0
                    or visualized < args.max_visualizations
                )
                if should_visualize:
                    context_panel = make_context_rgb(
                        chip_np[batch_offset],
                        valid_np[batch_offset],
                    )
                    panels = [context_panel, mask_to_rgb(target)]
                    titles = ["Context (latest VV, latest VH, temporal delta)", "Ground Truth"]

                    for bundle, pred in zip(bundles, predictions):
                        metrics = binary_segmentation_metrics(pred, target)
                        panels.append(mask_to_rgb(pred))
                        titles.append(
                            f"{bundle.name} Pred | IoU1={metrics['iou_class_1']:.3f} "
                            f"F1_1={metrics['f1_class_1']:.3f}",
                        )

                    for bundle, pred in zip(bundles, predictions):
                        panels.append(error_map_to_rgb(pred, target))
                        titles.append(f"{bundle.name} Error | TP green, FP orange, FN purple")

                    sample_title = (
                        f"{row.get('label', 'sample')} | {row.get('island', 'unknown')} | "
                        f"{row.get('date', 'unknown')} | patch {row.get('patch_id', '0')}"
                    )
                    output_path = visualization_dir / f"{safe_sample_stem(seen + batch_offset, row)}.png"
                    save_panel_grid(output_path, sample_title, panels, titles)
                    visualized += 1

            seen += batch_size

            if args.progress_every > 0 and batch_idx % args.progress_every == 0:
                print(f"Processed {seen}/{len(dataset)} samples...", flush=True)

    model_metric_rows: list[dict[str, object]] = []
    for bundle in bundles:
        metric_row: dict[str, object] = {
            "model_name": bundle.name,
            "checkpoint_path": str(bundle.checkpoint_path),
            "split": args.split,
            "num_samples": len(rows),
        }
        metric_row.update(bundle.metrics.compute())
        model_metric_rows.append(metric_row)

    write_csv(output_dir / "model_metrics.csv", model_metric_rows)
    write_csv(output_dir / "per_sample_metrics.csv", per_sample_rows)

    summary_lines = [
        f"manifest={manifest_path}",
        f"split={args.split}",
        f"device={device}",
        f"num_samples={len(rows)}",
        f"num_models={len(bundles)}",
        f"visualizations_saved={visualized}",
        "",
        "models:",
    ]
    for bundle in bundles:
        metrics = bundle.metrics.compute()
        summary_lines.append(
            f"- {bundle.name} | checkpoint={bundle.checkpoint_path} | "
            f"mean_iou={metrics['mean_iou']:.4f} | iou_class_1={metrics['iou_class_1']:.4f} | "
            f"f1_class_1={metrics['f1_class_1']:.4f}",
        )

    summary_lines.extend(
        [
            "",
            "legend:",
            "- Ground Truth / Pred: red=change, dark=background, gray=ignore",
            "- Error: green=true positive, orange=false positive, purple=false negative, gray=ignore",
        ],
    )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    print(f"Saved outputs to: {output_dir}")
    print(f"  - {output_dir / 'model_metrics.csv'}")
    print(f"  - {output_dir / 'per_sample_metrics.csv'}")
    print(f"  - {output_dir / 'summary.txt'}")
    print(f"  - {visualization_dir}")


if __name__ == "__main__":
    main()
