"""Compute g-ratio as a function of angle for a single axon image."""

import argparse
import os
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, map_coordinates


def get_center_from_csv(csv_path):
    """Read center coordinates from a morphometrics CSV.

    Args:
        csv_path: Path to the axon_morphometrics.csv file.

    Returns:
        Tuple (x, y) in pixel coordinates.
    """
    df = pd.read_csv(csv_path, index_col=0)
    x0 = df.iloc[0]['x0 (px)']
    y0 = df.iloc[0]['y0 (px)']
    return (x0, y0)


def compute_center(image):
    """Compute center of mass of the axon mask.

    Args:
        image: 2D numpy array where axon pixels = 255.

    Returns:
        Tuple (x, y) in pixel coordinates.
    """
    axon_mask = (image == 255)
    y_center, x_center = center_of_mass(axon_mask)
    return (x_center, y_center)


def find_boundary_radius(image, center, angle_deg, threshold, max_radius=None):
    """Find the radius to a mask boundary along a ray from center.

    Casts a ray from the center outward at the given angle and finds
    the last pixel along the ray whose interpolated value meets the threshold.

    Args:
        image: 2D numpy array (height x width), uint8.
        center: Tuple (x, y) in pixel coordinates.
        angle_deg: Angle in degrees (0 = right, counterclockwise).
        threshold: Minimum interpolated pixel value to consider as inside the mask.
        max_radius: Maximum search distance in pixels. Defaults to image diagonal.

    Returns:
        Radius in pixels from center to boundary.
    """
    if max_radius is None:
        max_radius = np.sqrt(image.shape[0]**2 + image.shape[1]**2) / 2

    angle_rad = np.deg2rad(angle_deg)
    dx = np.cos(angle_rad)
    dy = -np.sin(angle_rad)  # negate: image y increases downward

    # Sample at sub-pixel resolution along the ray
    n_samples = int(max_radius * 2)
    distances = np.linspace(0, max_radius, n_samples)

    x_samples = center[0] + distances * dx
    y_samples = center[1] + distances * dy

    # map_coordinates expects (row, col) = (y, x)
    coords = np.vstack([y_samples, x_samples])
    sampled = map_coordinates(image.astype(float), coords, order=1, mode='constant', cval=0)

    mask_points = sampled >= threshold
    if not np.any(mask_points):
        return 0.0

    last_index = np.where(mask_points)[0][-1]
    return distances[last_index]


def compute_angular_gratio(image_path, center=None, n_angles=360, csv_path=None):
    """Compute g-ratio as a function of angle for a single axon image.

    Args:
        image_path: Path to the axon image PNG.
        center: Tuple (x, y) or None to auto-detect.
        n_angles: Number of angles to sample over [0, 360).
        csv_path: Path to morphometrics CSV for center detection.

    Returns:
        Dictionary with keys: angles, r_axon, r_fiber, gratio, center.
    """
    image = imageio.imread(image_path)

    if center is None:
        if csv_path is not None and os.path.exists(csv_path):
            center = get_center_from_csv(csv_path)
        else:
            center = compute_center(image)

    angles = np.linspace(0, 360, n_angles, endpoint=False)
    r_axon = np.zeros(n_angles)
    r_fiber = np.zeros(n_angles)

    for i, angle in enumerate(angles):
        r_axon[i] = find_boundary_radius(image, center, angle, threshold=200)
        r_fiber[i] = find_boundary_radius(image, center, angle, threshold=60)

    gratio = np.divide(r_axon, r_fiber, out=np.zeros_like(r_axon), where=r_fiber != 0)

    return {
        'angles': angles,
        'r_axon': r_axon,
        'r_fiber': r_fiber,
        'gratio': gratio,
        'center': center,
    }


def plot_angular_analysis(results, image_path=None, save_path=None):
    """Create a 3-panel figure: image, polar radii, and g-ratio vs angle."""
    fig = plt.figure(figsize=(18, 6))
    theta = np.deg2rad(results['angles'])

    # Panel 1: original image with center
    if image_path:
        ax1 = fig.add_subplot(131)
        image = imageio.imread(image_path)
        ax1.imshow(image, cmap='gray')
        ax1.plot(results['center'][0], results['center'][1],
                 'r+', markersize=15, markeredgewidth=2)
        ax1.set_title('Image with center')
        ax1.axis('off')

    # Panel 2: polar plot of radii
    ax2 = fig.add_subplot(132, projection='polar')
    ax2.plot(theta, results['r_axon'], label='Axon radius', linewidth=2)
    ax2.plot(theta, results['r_fiber'], label='Fiber radius', linewidth=2)
    ax2.set_theta_zero_location('E')
    ax2.set_theta_direction(1)
    ax2.legend(loc='upper right')
    ax2.set_title('Radii vs Angle', pad=20)

    # Panel 3: g-ratio vs angle
    ax3 = fig.add_subplot(133)
    ax3.plot(results['angles'], results['gratio'], linewidth=2)
    mean_g = np.mean(results['gratio'])
    ax3.axhline(mean_g, color='r', linestyle='--', label=f'Mean: {mean_g:.4f}')
    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('G-ratio')
    ax3.set_title('G-ratio vs Angle')
    ax3.set_xlim(0, 360)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute g-ratio as a function of angle for a single axon image."
    )
    parser.add_argument("image_path", type=str, help="Path to axon image PNG")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to morphometrics CSV for center coordinates")
    parser.add_argument("--center", type=float, nargs=2, default=None,
                        metavar=('X', 'Y'), help="Manual center (x, y) in pixels")
    parser.add_argument("--n-angles", type=int, default=360,
                        help="Number of angles to sample (default: 360)")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="Prefix for output files (plot PNG and CSV)")
    args = parser.parse_args()

    center = tuple(args.center) if args.center else None

    # Auto-detect CSV if not provided
    csv_path = args.csv_path
    if csv_path is None:
        candidate = args.image_path.replace('.png', '_axon_morphometrics.csv')
        if os.path.exists(candidate):
            csv_path = candidate

    results = compute_angular_gratio(
        args.image_path, center=center, n_angles=args.n_angles, csv_path=csv_path
    )

    print(f"Center: ({results['center'][0]:.2f}, {results['center'][1]:.2f})")
    print(f"Mean g-ratio: {np.mean(results['gratio']):.4f}")
    print(f"Std g-ratio:  {np.std(results['gratio']):.4f}")
    print(f"Min g-ratio:  {np.min(results['gratio']):.4f}")
    print(f"Max g-ratio:  {np.max(results['gratio']):.4f}")

    # Determine output prefix
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = args.image_path.replace('.png', '')

    # Save CSV
    csv_output = f"{output_prefix}_angular_gratio.csv"
    df = pd.DataFrame({
        'angle_deg': results['angles'],
        'r_axon_px': results['r_axon'],
        'r_fiber_px': results['r_fiber'],
        'gratio': results['gratio'],
    })
    df.to_csv(csv_output, index=False)
    print(f"Results saved to {csv_output}")

    # Save plot
    plot_path = f"{output_prefix}_angular_analysis.png"
    plot_angular_analysis(results, args.image_path, save_path=plot_path)
