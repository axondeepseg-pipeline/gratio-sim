"""Generate a dictionary of axon images with varying parameters."""

import numpy as np
import os
from pathlib import Path
from simulate_axons import SimulateAxons, calc_myelin_thickness


def generate_axon_dataset(
    axon_diameters=None,
    gratios=None,
    angles=None,
    output_dir="axon_dataset",
):
    """Generate a dataset of axon images with varying parameters.
    
    Args:
        axon_diameters: List of axon radii (in pixels). Defaults to 20 evenly spaced values.
        gratios: List of g-ratios. Defaults to [0.7] (constant).
        angles: List of plane angles in degrees. Defaults to [0] (constant).
        output_dir: Directory to save images.
    
    Returns:
        Dictionary with structure: {(diameter, gratio, angle): image_array}
    """
    
    # Set defaults
    if axon_diameters is None:
        axon_diameters = np.sort(np.append(np.linspace(10, 200, 20), [5, 7, 9, 11, 13, 15 ,17, 19]))
    if gratios is None:
        gratios = [0.7]
    if angles is None:
        angles = [0]
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Calculate image dimensions once, based on largest axon + worst-case conditions
    # Find the worst case: largest diameter + smallest g-ratio (thickest myelin) + largest angle (max stretch)
    max_diameter = max(axon_diameters)
    min_gratio = min(gratios)  # Smallest g-ratio = thickest myelin
    max_angle = max(angles)
    
    # Calculate maximum total radius needed (axon + myelin)
    max_radius = max_diameter / 2
    max_myelin_thickness = calc_myelin_thickness(max_radius, min_gratio)
    max_total_radius = max_radius + max_myelin_thickness
    
    # At the largest angle, the axon stretches along the major axis
    major_axis_stretch = max_total_radius / np.cos(np.deg2rad(max_angle))
    
    # Set fixed dimensions for all images
    width = 1000
    height = 750
    
    print(f"Image dimensions: {width} x {height}")
    print(f"Axon diameters: {len(axon_diameters)} values from {axon_diameters[0]:.1f} to {axon_diameters[-1]:.1f}")
    print(f"G-ratios: {len(gratios)} values from {gratios[0]:.2f} to {gratios[-1]:.2f}")
    print(f"Angles: {len(angles)} values from {angles[0]:.1f}° to {angles[-1]:.1f}°")
    
    dataset = {}
    total_images = len(axon_diameters) * len(gratios) * len(angles)
    count = 0
    
    for diameter in axon_diameters:
        for gratio in gratios:
            for angle in angles:
                count += 1
                axon_radius = diameter / 2
                
                # Create simulator and generate axon
                sim = SimulateAxons(image_dims=[width, height])
                sim.generate_axon(
                    axon_radius=axon_radius,
                    center=[width / 2, height / 2],
                    gratio=gratio,
                    plane_angle=angle,
                )
                
                # Store in dictionary
                key = (float(diameter), float(gratio), float(angle))
                dataset[key] = sim.image.copy()
                
                # Save image
                # Format diameter as 3 digits (zero-padded integer part)
                diam_int = int(diameter)
                filename = f"axon_d{diam_int:03d}_g{gratio:.2f}_a{angle:.1f}.png"
                filepath = os.path.join(output_dir, filename)
                sim.save(filepath)
                
                if count % 50 == 0:
                    print(f"Generated {count}/{total_images} images...")
    
    print(f"Generated {total_images} images in '{output_dir}/'")
    return dataset


if __name__ == "__main__":
    dataset = generate_axon_dataset()
    print(f"\nDataset contains {len(dataset)} axon images")
    print(f"Sample keys: {list(dataset.keys())[:3]}")
