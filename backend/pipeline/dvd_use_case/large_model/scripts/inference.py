"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
inference_pose_rc.py

Tool: YOLO-Pose inference — 1×3 comparison plots (RC only)
    Panel 1: RC patch + Ground Truth keypoints (from labels)
    Panel 2: RC patch + Predicted keypoints
    Panel 3: Per-image inference stats

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-04-19

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""
# type: ignore
import os
import re
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from ultralytics import YOLO

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utilities.read_yaml import read_yaml
from scripts.profile_model import profile_model

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
config           = read_yaml(Path(__file__).resolve().parent.parent / 'config.yaml')
inference_config = config.get('inference', {})

MODEL_PATH      = inference_config.get('model_path', '')
TEST_IMAGES_DIR = inference_config.get('test_images_dir', '')
TEST_LABELS_DIR = inference_config.get('test_labels_dir', '')
OUTPUT_DIR      = inference_config.get('output_dir', '')

IOU_THRESHOLD  = inference_config.get('iou_threshold', '')
CONF_THRESHOLD = inference_config.get('conf_threshold', '')

# ─── Visual style ────────────────────────────────────────────
GT_COLOR   = '#FF3333'   # red
PRED_COLOR = '#33FF33'   # lime
KP_SIZE    = 60          # marker size for keypoints
BOX_LW     = 1           # bbox linewidth

# ─── Stats panel style ───────────────────────────────────────
PANEL_BG      = '#1e1e1e'
PANEL_FG      = '#e0e0e0'
SECTION_COLOR = '#4a90d9'
GOOD_COLOR    = '#33FF33'
WARN_COLOR    = '#FFB347'


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def extract_patch_number(filename):
    match = re.search(r'VD_(\d+)', filename)
    return match.group(1) if match else None


def extract_polarization(filename):
    if '_VH_' in filename:
        return 'VH'
    elif '_VV_' in filename:
        return 'VV'
    return None


def load_rc_image(img_path: str) -> tuple:
    ext = Path(img_path).suffix.lower()
    if ext in ('.tif', '.tiff'):
        if not HAS_TIFFFILE:
            raise ImportError("tifffile not installed")
        arr = tifffile.imread(img_path)
        display = arr[3] if arr.ndim == 3 and arr.shape[0] == 4 else arr[0]
        rgb = np.stack([display, display, display], axis=-1)
    else:
        rgb = np.array(Image.open(img_path).convert('RGB'))
    rc_h, rc_w = rgb.shape[:2]
    return rgb, rc_h, rc_w


def load_pose_labels(label_path, img_w, img_h):
    """
    Load YOLO-pose labels.
    Format: class xc yc w h kp_x kp_y kp_vis
    Returns list of dicts with 'bbox' [x1,y1,x2,y2] and 'kp' (x,y) in pixels.
    """
    entries = []
    if not os.path.exists(label_path):
        return entries
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            x_c, y_c, w, h = map(float, parts[1:5])
            x1 = (x_c - w / 2) * img_w
            y1 = (y_c - h / 2) * img_h
            x2 = (x_c + w / 2) * img_w
            y2 = (y_c + h / 2) * img_h

            kp = None
            if len(parts) >= 8:
                kp_x = float(parts[5]) * img_w
                kp_y = float(parts[6]) * img_h
                kp = (kp_x, kp_y)
            else:
                kp = (x_c * img_w, y_c * img_h)

            entries.append({'bbox': [x1, y1, x2, y2], 'kp': kp})
    return entries


def draw_boxes_and_kps(ax, entries, box_color, kp_color, label, show_bbox=True, show_conf=False):
    """Draw bounding boxes and keypoint markers."""
    for i, entry in enumerate(entries):
        if show_bbox:
            x1, y1, x2, y2 = entry['bbox']
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=BOX_LW, edgecolor=box_color,
                facecolor='none', linestyle='--',
                label=f'{label} box' if i == 0 else None
            )
            ax.add_patch(rect)

            if show_conf and 'conf' in entry:
                ax.text(x1, y1 - 2, f'{entry["conf"]:.2f}',
                        color=box_color, fontsize=7, fontweight='bold',
                        verticalalignment='bottom',
                        bbox=dict(facecolor='black', alpha=0.6, pad=1, edgecolor='none'))

        if entry['kp'] is not None:
            kx, ky = entry['kp']
            ax.scatter(kx, ky, c=kp_color, s=KP_SIZE, marker='x',
                       linewidths=2, zorder=5,
                       label=f'{label} kp' if i == 0 else None)


def extract_pose_predictions(results, conf_threshold=0.0):
    """
    Extract bbox + keypoint from YOLO-pose results.
    Returns list of dicts with 'bbox', 'kp', 'conf'.
    """
    entries = []
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return entries

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()

    kps = None
    if r.keypoints is not None and r.keypoints.data is not None:
        kps = r.keypoints.data.cpu().numpy()

    for i in range(len(boxes)):
        if confs[i] < conf_threshold:
            continue
        x1, y1, x2, y2 = boxes[i]
        kp = None
        if kps is not None and i < len(kps):
            kp = (float(kps[i][0][0]), float(kps[i][0][1]))

        entries.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'kp':   kp,
            'conf': float(confs[i]),
        })
    return entries


# ─────────────────────────────────────────────────────────────
# Profile summary figure
# ─────────────────────────────────────────────────────────────
def save_profile_figure(profile_stats: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(PANEL_BG)
    ax.axis('off')
    fig.suptitle('Model Profile Summary', fontsize=14,
                 fontweight='bold', color=PANEL_FG)

    sections = {
        'Complexity':  ['profile/GMACs',          'profile/GFLOPs',
                        'profile/params_M'],
        'Latency':     ['profile/latency_ms',      'profile/fps'],
        'GPU Memory':  ['profile/gpu_mem_alloc_GB','profile/gpu_mem_peak_GB'],
        'GPU Compute': ['profile/gpu_util_avg_%',  'profile/gpu_util_peak_%'],
        'GPU Power':   ['profile/gpu_power_avg_W', 'profile/gpu_power_peak_W'],
        'CPU':         ['profile/cpu_util_avg_%',  'profile/cpu_util_peak_%',
                        'profile/ram_GB',
                        'profile/cpu_power_avg_W', 'profile/cpu_power_peak_W'],
    }

    rows = []
    for section, keys in sections.items():
        rows.append([f'  {section}', ''])
        for k in keys:
            if k in profile_stats:
                rows.append([f'    {k.replace("profile/", "")}',
                             f'{profile_stats[k]:.3f}'])

    table = ax.table(
        cellText=rows,
        colLabels=['Metric', 'Value'],
        cellLoc='left',
        loc='center',
        colWidths=[0.65, 0.35]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(PANEL_BG)
        cell.get_text().set_color(PANEL_FG)
        cell.set_edgecolor('#444')
        text = cell.get_text().get_text()
        if row == 0:                          # header
            cell.set_facecolor(SECTION_COLOR)
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        elif not text.startswith('    '):     # section title row
            cell.set_facecolor('#2c2c2c')
            cell.get_text().set_color(SECTION_COLOR)
            cell.get_text().set_fontweight('bold')

    plt.tight_layout()
    out = os.path.join(output_dir, 'model_profile_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Profile summary saved → {out}")


# ─────────────────────────────────────────────────────────────
# Per-image stats panel
# ─────────────────────────────────────────────────────────────
def draw_stats_panel(ax, pred_entries, infer_ms, profile_stats):
    """Render the inference stats text panel onto ax."""
    ax.set_facecolor(PANEL_BG)
    ax.axis('off')
    ax.set_title('Inference Stats', fontsize=11, color=PANEL_FG, pad=8)

    model_avg_ms = profile_stats.get('profile/latency_ms', 0.0)
    model_fps    = profile_stats.get('profile/fps',        0.0)

    # Confidence scores per detection
    if pred_entries:
        conf_lines = '\n'.join(
            f'    Det {i+1:>2d}:  {e["conf"]:.4f}'
            for i, e in enumerate(pred_entries)
        )
    else:
        conf_lines = '    —'

    stats_text = (
        f'{"─" * 28}\n'
        f' LATENCY\n'
        f'{"─" * 28}\n'
        f'  This image   : {infer_ms:>8.1f} ms\n'
        f'  Model avg    : {model_avg_ms:>8.1f} ms\n'
        f'  FPS (avg)    : {model_fps:>8.1f}\n'
        f'\n{"─" * 28}\n'
        f' MODEL  (constant)\n'
        f'{"─" * 28}\n'
        f'  GFLOPs       : {profile_stats.get("profile/GFLOPs",         0):.2f}\n'
        f'  Params       : {profile_stats.get("profile/params_M",       0):.2f} M\n'
        f'  GPU mem peak : {profile_stats.get("profile/gpu_mem_peak_GB",0):.3f} GB\n'
        f'  GPU util avg : {profile_stats.get("profile/gpu_util_avg_%", 0):.1f} %\n'
        f'  GPU pwr avg  : {profile_stats.get("profile/gpu_power_avg_W",0):.1f} W\n'
        f'  CPU util avg : {profile_stats.get("profile/cpu_util_avg_%", 0):.1f} %\n'
        f'  RAM          : {profile_stats.get("profile/ram_GB",         0):.2f} GB\n'
    )

    ax.text(
        0.05, 0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment='top',
        fontfamily='monospace',
        color=PANEL_FG,
        bbox=dict(
            boxstyle='round,pad=0.6',
            facecolor=PANEL_BG,
            edgecolor='#555',
            alpha=0.95
        )
    )


# ─────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────
def run_inference(args):
    model_path      = args.model  or MODEL_PATH
    test_images_dir = args.images or TEST_IMAGES_DIR
    test_labels_dir = args.labels or TEST_LABELS_DIR
    output_dir      = args.output or OUTPUT_DIR
    iou_thresh      = args.iou
    conf_thresh     = args.conf

    os.makedirs(output_dir, exist_ok=True)
    pred_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(pred_labels_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "inference_pose_log.txt")
    open(log_path, "w").close()

    print(f"Model : {model_path}")
    print(f"Images: {test_images_dir}")
    print(f"Labels: {test_labels_dir}")
    print(f"Conf  : {conf_thresh}  IoU: {iou_thresh}")
    print("=" * 60)


    model = YOLO(model_path)

    # Move model to CUDA if available for profiling
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)

    # ── Profile model (once) ─────────────────────────────────
    print("\nProfiling model...")
    profile_stats = profile_model(model)
    print()

    # Write profile to log
    with open(log_path, "a") as lf:
        lf.write("=" * 60 + "\n")
        lf.write("Model Profile\n")
        lf.write("=" * 60 + "\n")
        for k, v in profile_stats.items():
            lf.write(f"{k:<32}: {v:.3f}\n")
        lf.write("=" * 60 + "\n\n")

    # Save standalone profile figure
    save_profile_figure(profile_stats, output_dir)

    # ── Gather test images ───────────────────────────────────
    test_images = sorted([
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
    ])
    print(f"\nFound {len(test_images)} test images\n")

    # ── Per-image loop ───────────────────────────────────────
    for img_name in test_images:
        patch_num = extract_patch_number(img_name)
        if patch_num is None:
            continue

        polarization = extract_polarization(img_name)
        pol_label    = polarization if polarization else '4ch'
        img_path     = os.path.join(test_images_dir, img_name)
        label_path   = os.path.join(test_labels_dir, Path(img_name).stem + '.txt')

        # Load image
        try:
            img_array, rc_h, rc_w = load_rc_image(img_path)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            continue

        # Ground truth
        gt_entries = load_pose_labels(label_path, rc_w, rc_h)

        # ── Timed inference ──────────────────────────────────
        t0 = time.perf_counter()
        results = model.predict(
            img_path, imgsz=rc_w,
            iou=iou_thresh, conf=conf_thresh,
            verbose=False
        )
        infer_ms = (time.perf_counter() - t0) * 1000

        pred_entries = extract_pose_predictions(results, conf_thresh)

        # ── Save predictions as YOLO-pose label txt ──────────
        pred_txt_path = os.path.join(pred_labels_dir, Path(img_name).stem + ".txt")
        with open(pred_txt_path, "w") as pf:
            for entry in pred_entries:
                x1, y1, x2, y2 = entry['bbox']
                xc = ((x1 + x2) / 2) / rc_w
                yc = ((y1 + y2) / 2) / rc_h
                w = (x2 - x1) / rc_w
                h = (y2 - y1) / rc_h
                if entry['kp'] is not None:
                    kp_x = entry['kp'][0] / rc_w
                    kp_y = entry['kp'][1] / rc_h
                    pf.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {kp_x:.6f} {kp_y:.6f} 2\n")
                else:
                    pf.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # ── 1×3 figure ───────────────────────────────────────
        fig, axes = plt.subplots(
            1, 3, figsize=(21, 7),
            gridspec_kw={'width_ratios': [2, 2, 1]}
        )
        fig.patch.set_facecolor('#111111')
        fig.suptitle(
            f'VD_{patch_num} ({pol_label})  |  '
            f'GT: {len(gt_entries)}   Pred: {len(pred_entries)}  '
            f'(conf ≥ {conf_thresh}, iou = {iou_thresh})  |  '
            f'Latency: {infer_ms:.1f} ms',
            fontsize=13, fontweight='bold', color=PANEL_FG
        )

        # Panel 1 — RC + Ground Truth
        axes[0].set_facecolor('#111111')
        axes[0].imshow(img_array, cmap='gray' if img_array.ndim == 2 else None)
        axes[0].set_title('RC Patch + Ground Truth', fontsize=11, color=PANEL_FG)
        draw_boxes_and_kps(axes[0], gt_entries, GT_COLOR, GT_COLOR, 'GT')
        if gt_entries:
            legend = axes[0].legend(loc='upper right', fontsize=8)
            legend.get_frame().set_facecolor(PANEL_BG)
            for text in legend.get_texts():
                text.set_color(PANEL_FG)
        axes[0].axis('off')

        # Panel 2 — RC + Predictions
        axes[1].set_facecolor('#111111')
        axes[1].imshow(img_array, cmap='gray' if img_array.ndim == 2 else None)
        axes[1].set_title('RC Patch + Predictions', fontsize=11, color=PANEL_FG)
        draw_boxes_and_kps(axes[1], pred_entries, PRED_COLOR, PRED_COLOR, 'Pred', show_conf=True)
        if pred_entries:
            legend = axes[1].legend(loc='upper right', fontsize=8)
            legend.get_frame().set_facecolor(PANEL_BG)
            for text in legend.get_texts():
                text.set_color(PANEL_FG)
        axes[1].axis('off')

        # Panel 3 — Inference stats
        draw_stats_panel(axes[2], pred_entries, infer_ms, profile_stats)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"VD_{patch_num}_{pol_label}_pose.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()

        # Log per-image entry
        with open(log_path, "a") as lf:
            lf.write(
                f"VD_{patch_num} ({pol_label:<4}) | "
                f"GT: {len(gt_entries):>3} | "
                f"Pred: {len(pred_entries):>3} | "
                f"Latency: {infer_ms:>7.2f} ms\n"
            )

        print(
            f"Saved: VD_{patch_num} ({pol_label:<4}) | "
            f"GT: {len(gt_entries):>3} | "
            f"Pred: {len(pred_entries):>3} | "
            f"{infer_ms:.1f} ms"
        )

    print(f"\nDone! Results saved to: {output_dir}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="YOLO-Pose inference with 1×3 RC comparison plots."
    )
    parser.add_argument('--model',  type=str,   default=None, help="Path to pose model weights")
    parser.add_argument('--images', type=str,   default=None, help="Test images directory")
    parser.add_argument('--labels', type=str,   default=None, help="Test labels directory")
    parser.add_argument('--output', type=str,   default=None, help="Output directory")
    parser.add_argument('--conf',   type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument('--iou',    type=float, default=IOU_THRESHOLD,  help="IoU threshold for NMS")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()