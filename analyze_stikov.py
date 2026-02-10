"""Analyze angular g-ratio for each axon in a real histology image."""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from compute_angular_gratio import find_boundary_radius


def load_stikov_data(data_dir):
    """Load all images and morphometrics from a stikov_image directory.

    Returns:
        dict with keys: histology, axon_mask, myelin_mask, instance_map, morphometrics
    """
    histology = np.array(Image.open(os.path.join(data_dir, 'ADS.png')))
    axon_mask = np.array(Image.open(os.path.join(data_dir, 'ADS_seg-axon.png')))
    myelin_mask = np.array(Image.open(os.path.join(data_dir, 'ADS_seg-myelin.png')))
    instance_map = np.array(Image.open(os.path.join(data_dir, 'ADS_instance-map.png')))
    morpho = pd.read_csv(os.path.join(data_dir, 'Morphometrics.csv'), index_col=0)

    return {
        'histology': histology,
        'axon_mask': axon_mask,
        'myelin_mask': myelin_mask,
        'instance_map': instance_map,
        'morphometrics': morpho,
    }


def isolate_axon(data, axon_idx, padding=20):
    """Isolate a single axon and its myelin, cropped from the full image.

    Args:
        data: dict from load_stikov_data.
        axon_idx: Row index in the morphometrics DataFrame.
        padding: Pixels of padding around the bounding box.

    Returns:
        dict with keys: seg_crop, histology_crop, center, bbox, axon_id
        seg_crop has values 255=axon, 127=myelin, 0=background.
    """
    morpho = data['morphometrics']
    row = morpho.iloc[axon_idx]
    instance_map = data['instance_map']

    # Instance label is 1-indexed (row 0 -> label 1)
    label = axon_idx + 1

    # Bounding box from CSV
    bbox_min_y = int(row['bbox_min_y'])
    bbox_min_x = int(row['bbox_min_x'])
    bbox_max_y = int(row['bbox_max_y'])
    bbox_max_x = int(row['bbox_max_x'])

    # Add padding, clamped to image bounds
    h, w = instance_map.shape
    y0 = max(0, bbox_min_y - padding)
    y1 = min(h, bbox_max_y + padding)
    x0 = max(0, bbox_min_x - padding)
    x1 = min(w, bbox_max_x + padding)

    # Crop
    inst_crop = instance_map[y0:y1, x0:x1]
    axon_crop = data['axon_mask'][y0:y1, x0:x1]
    myelin_crop = data['myelin_mask'][y0:y1, x0:x1]

    # Histology crop (may be RGB/RGBA)
    hist_crop = data['histology'][y0:y1, x0:x1]

    # Build isolated seg: only this axon's fiber
    fiber_mask = (inst_crop == label)
    seg_crop = np.zeros(inst_crop.shape, dtype=np.uint8)
    seg_crop[fiber_mask & (axon_crop == 255)] = 255
    seg_crop[fiber_mask & (myelin_crop == 255)] = 127

    # Center relative to crop
    center_x = row['x0 (px)'] - x0
    center_y = row['y0 (px)'] - y0

    return {
        'seg_crop': seg_crop,
        'histology_crop': hist_crop,
        'center': (center_x, center_y),
        'bbox': (y0, y1, x0, x1),
        'axon_id': axon_idx,
    }


def compute_angular_gratio_from_seg(seg, center, n_angles=360):
    """Compute angular g-ratio from an isolated seg image (255/127/0).

    Args:
        seg: 2D uint8 array (255=axon, 127=myelin, 0=background).
        center: Tuple (x, y) in pixel coordinates.
        n_angles: Number of angles.

    Returns:
        dict with keys: angles, r_axon, r_fiber, gratio, center
    """
    max_radius = max(seg.shape)

    angles = np.linspace(0, 360, n_angles, endpoint=False)
    r_axon = np.zeros(n_angles)
    r_fiber = np.zeros(n_angles)

    for i, angle in enumerate(angles):
        r_axon[i] = find_boundary_radius(seg, center, angle, threshold=200, max_radius=max_radius)
        r_fiber[i] = find_boundary_radius(seg, center, angle, threshold=60, max_radius=max_radius)

    gratio = np.divide(r_axon, r_fiber, out=np.zeros_like(r_axon), where=r_fiber != 0)

    return {
        'angles': angles,
        'r_axon': r_axon,
        'r_fiber': r_fiber,
        'gratio': gratio,
        'center': center,
    }


def plot_axon_analysis(results, histology_crop, seg_crop, axon_id, save_path=None):
    """Create 4-panel figure for a single axon.

    Panels: histology crop, seg overlay, polar radii, g-ratio vs angle.
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    theta = np.deg2rad(results['angles'])

    # Panel 1: histology crop
    ax1 = axes[0]
    ax1.imshow(histology_crop, cmap='gray' if histology_crop.ndim == 2 else None)
    ax1.plot(results['center'][0], results['center'][1], 'r+', markersize=12, markeredgewidth=2)
    ax1.set_title(f'Axon {axon_id} - Histology')
    ax1.axis('off')

    # Panel 2: seg overlay on histology
    ax2 = axes[1]
    if histology_crop.ndim == 3:
        gray = np.mean(histology_crop[..., :3], axis=2).astype(np.uint8)
    else:
        gray = histology_crop
    overlay = np.stack([gray, gray, gray], axis=2)
    overlay[seg_crop == 255] = [0, 150, 255]   # axon = blue
    overlay[seg_crop == 127] = [255, 150, 0]    # myelin = orange
    ax2.imshow(overlay)
    ax2.plot(results['center'][0], results['center'][1], 'r+', markersize=12, markeredgewidth=2)
    ax2.set_title('Segmentation overlay')
    ax2.axis('off')

    # Panel 3: polar radii
    ax3 = fig.add_subplot(1, 4, 3, projection='polar')
    axes[2].set_visible(False)  # hide the non-polar axes placeholder
    ax3.plot(theta, results['r_axon'], label='Axon', linewidth=1.5)
    ax3.plot(theta, results['r_fiber'], label='Fiber', linewidth=1.5)
    ax3.set_theta_zero_location('E')
    ax3.set_theta_direction(1)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_title('Radii vs Angle', pad=15, fontsize=10)

    # Panel 4: g-ratio vs angle
    ax4 = axes[3]
    ax4.plot(results['angles'], results['gratio'], linewidth=1.5)
    mean_g = np.mean(results['gratio'][results['gratio'] > 0])
    ax4.axhline(mean_g, color='r', linestyle='--', label=f'Mean: {mean_g:.3f}')
    median_g = np.median(results['gratio'][results['gratio'] > 0])
    ax4.axhline(median_g, color='g', linestyle='--', label=f'Median: {median_g:.3f}')
    ax4.set_xlabel('Angle (degrees)')
    ax4.set_ylabel('G-ratio')
    ax4.set_title('G-ratio vs Angle')
    ax4.set_xlim(0, 360)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.suptitle(f'Axon {axon_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute angular g-ratio for axons in a real histology image.'
    )
    parser.add_argument('--data-dir', type=str, default='stikov_image',
                        help='Directory with ADS files (default: stikov_image)')
    parser.add_argument('--axons', type=int, nargs='*', default=None,
                        help='Axon indices to process (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: <data-dir>/angular_gratio)')
    parser.add_argument('--n-angles', type=int, default=360,
                        help='Number of angles (default: 360)')
    parser.add_argument('--padding', type=int, default=20,
                        help='Padding around bounding box in pixels (default: 20)')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.data_dir, 'angular_gratio')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading data from {args.data_dir}...')
    data = load_stikov_data(args.data_dir)
    n_axons = len(data['morphometrics'])
    print(f'Found {n_axons} axons')

    axon_indices = args.axons if args.axons is not None else range(n_axons)

    all_results = []
    for idx in axon_indices:
        if idx >= n_axons:
            print(f'  Skipping axon {idx} (out of range)')
            continue

        axon_data = isolate_axon(data, idx, padding=args.padding)

        # Skip if axon has no pixels in the instance map
        if np.sum(axon_data['seg_crop'] == 255) == 0:
            print(f'  Axon {idx}: no axon pixels found, skipping')
            continue

        results = compute_angular_gratio_from_seg(
            axon_data['seg_crop'], axon_data['center'], n_angles=args.n_angles
        )

        mean_g = np.mean(results['gratio'][results['gratio'] > 0])
        median_g = np.median(results['gratio'][results['gratio'] > 0])
        ads_g = data['morphometrics'].iloc[idx]['gratio']
        print(f'  Axon {idx}: angular mean g-ratio={mean_g:.3f}, angular median g-ratio={median_g:.3f}, ADS g-ratio={ads_g:.3f}')

        # Save plot
        plot_path = os.path.join(output_dir, f'axon_{idx:04d}.png')
        plot_axon_analysis(results, axon_data['histology_crop'], axon_data['seg_crop'], idx, save_path=plot_path)

        all_results.append({
            'axon_id': idx,
            'angular_mean_gratio': mean_g,
            'angular_std_gratio': np.std(results['gratio'][results['gratio'] > 0]),
            'ads_gratio': ads_g,
        })

    # Save summary
    if all_results:
        summary = pd.DataFrame(all_results)
        summary_path = os.path.join(output_dir, 'summary.csv')
        summary.to_csv(summary_path, index=False)
        print(f'\nSummary saved to {summary_path}')
        print(f'Plots saved to {output_dir}/')
