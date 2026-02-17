"""Validate axon simulations against AxonDeepSeg morphometrics."""

import os
import glob
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from generate_axon_dataset import calc_myelin_thickness


def create_segmentation_masks(dataset_dir="axon_dataset"):
    """Create segmentation masks from the simulated images.
    
    The simulated images are already masks:
    - 255 = axon
    - 127 = myelin
    - 0 = background
    """
    print(f"Creating segmentation masks from simulated images in {dataset_dir}...")
    
    # Get all PNG files in the dataset directory
    image_paths = glob.glob(os.path.join(dataset_dir, "axon_*.png"))
    
    if not image_paths:
        print(f"No axon_*.png files found in {dataset_dir}")
        return
    
    print(f"Found {len(image_paths)} simulated images")
    
    import imageio
    
    for i, image_path in enumerate(image_paths, 1):
        filename = Path(image_path).stem
        
        # Read the image
        img = imageio.imread(image_path)
        
        # Create axon mask (255 = axon, rest = 0)
        axon_mask = (img == 255).astype(np.uint8) * 255
        
        # Create myelin mask (127 = myelin, rest = 0)
        myelin_mask = (img == 127).astype(np.uint8) * 255
        
        # Create axonmyelin mask (255 or 127, rest = 0)
        axonmyelin_mask = img.copy()
        axonmyelin_mask[axonmyelin_mask == 0] = 0
        
        # Save masks
        axon_path = image_path.replace(".png", "_seg-axon.png")
        myelin_path = image_path.replace(".png", "_seg-myelin.png")
        axonmyelin_path = image_path.replace(".png", "_seg-axonmyelin.png")
        
        imageio.imwrite(axon_path, axon_mask)
        imageio.imwrite(myelin_path, myelin_mask)
        imageio.imwrite(axonmyelin_path, axonmyelin_mask)
        
        if (i - 1) % 5 == 0:
            print(f"  [{i}/{len(image_paths)}] Created masks for {filename}...")
    
    print(f"Created segmentation masks for all {len(image_paths)} images")



def extract_morphometrics(dataset_dir="axon_dataset"):
    """Extract morphometrics from all segmented images."""
    print(f"\nExtracting morphometrics from {dataset_dir}...")
    
    # Create pixel_size_in_micrometer.txt file (1 pixel = 1 micrometer)
    pixel_size_file = os.path.join(dataset_dir, "pixel_size_in_micrometer.txt")
    with open(pixel_size_file, "w") as f:
        f.write("1.0")
    
    # Get all original simulated images (that start with "axon_d")
    image_paths = sorted(glob.glob(os.path.join(dataset_dir, "axon_d*.png")))
    image_paths = [p for p in image_paths if not "_seg" in p]  # Exclude any masks
    
    if not image_paths:
        print(f"No simulated images found in {dataset_dir}")
        return
    
    print(f"Found {len(image_paths)} simulated images")
    
    # Run morphometrics for each image
    for i, image_path in enumerate(image_paths, 1):
        image_name = Path(image_path).stem
        print(f"  [{i}/{len(image_paths)}] Extracting morphometrics for {image_name}...")
        
        # Use just the filename since we'll run from the dataset directory
        rel_image_path = os.path.basename(image_path)
        output_file = "axon_morphometrics.csv"
        
        result = subprocess.run(
            ["axondeepseg_morphometrics", "-i", rel_image_path, "-a", "circle", "-f", output_file],
            capture_output=True,
            text=True,
            cwd=dataset_dir  # Run from dataset directory
        )
        if result.returncode != 0:
            print(f"    Warning: Morphometrics extraction failed for {image_name}")
            print(f"    Error: {result.stderr[:200]}")


def validate_morphometrics(dataset_dir="axon_dataset", pixel_size_um=1.0):
    """Validate morphometrics against expected values.
    
    Args:
        dataset_dir: Directory containing the dataset
        pixel_size_um: Pixel size in micrometers (for unit conversion)
    """
    print(f"\nValidating morphometrics...")
    
    # Get all morphometrics files, but only the main ones (not index or colorized versions)
    import os
    import re
    all_files = [f for f in os.listdir(dataset_dir) if "axon_morphometrics.csv" in f]
    # Filter to only keep files that match "axon_d###_g*.##_a*.#_axon_morphometrics.csv" pattern (exactly)
    pattern = r'^axon_d\d{3}_g[\d.]+_a[\d.]+_axon_morphometrics\.csv$'
    morpho_files = [f for f in all_files if re.match(pattern, f)]
    morpho_files = [os.path.join(dataset_dir, f) for f in morpho_files]
    
    if not morpho_files:
        print(f"No morphometrics files found in {dataset_dir}")
        return
    
    print(f"Found {len(morpho_files)} main morphometrics files\n")
    
    results = []
    
    for morpho_file in sorted(morpho_files):
        print(f"Processing {os.path.basename(morpho_file)}...")
        # Extract expected values from filename
        filename = Path(morpho_file).stem
        image_name = filename.replace("_axon_morphometrics", "")
        
        # Parse filename: axon_d10.0_g0.70_a0.0
        parts = image_name.split("_")
        expected_diameter_px = float(parts[1].replace("d", ""))
        expected_gratio = float(parts[2].replace("g", ""))
        expected_angle = float(parts[3].replace("a", ""))
        
        # Read morphometrics CSV
        try:
            df = pd.read_csv(morpho_file, index_col=0)
        except Exception as e:
            print(f"  Error reading {morpho_file}: {e}")
            continue
        
        if len(df) == 0:
            print(f"  Warning: No axons found in {image_name}")
            continue
        
        print(f"  Found {len(df)} axon(s)")
        
        # For each axon, compare expected vs measured
        for idx, row in df.iterrows():
            measured_diameter_um = row['axon_diam (um)']
            measured_diameter_px = measured_diameter_um / pixel_size_um  # convert to pixels
            measured_gratio = row['gratio']
            
            # Calculate expected myelin thickness
            expected_radius_px = expected_diameter_px / 2
            expected_myelin_thickness_px = calc_myelin_thickness(expected_radius_px, expected_gratio)
            expected_fiber_diameter_px = expected_diameter_px + 2 * expected_myelin_thickness_px
            
            # Error metrics
            diameter_error_px = measured_diameter_px - expected_diameter_px
            diameter_error_pct = (diameter_error_px / expected_diameter_px) * 100 if expected_diameter_px != 0 else 0
            gratio_error = measured_gratio - expected_gratio
            gratio_error_pct = (gratio_error / expected_gratio) * 100 if expected_gratio != 0 else 0
            
            results.append({
                'image_name': image_name,
                'axon_id': idx,
                'expected_diameter_px': expected_diameter_px,
                'measured_diameter_px': measured_diameter_px,
                'diameter_error_px': diameter_error_px,
                'diameter_error_pct': diameter_error_pct,
                'expected_gratio': expected_gratio,
                'measured_gratio': measured_gratio,
                'gratio_error': gratio_error,
                'gratio_error_pct': gratio_error_pct,
                'expected_angle': expected_angle,
                'axon_area_um2': row['axon_area (um^2)'],
                'myelin_area_um2': row['myelin_area (um^2)'],
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No results found!")
        return
    
    # Print summary statistics
    print("=" * 100)
    print("VALIDATION SUMMARY")
    print("=" * 100)
    print(f"\nTotal axons measured: {len(results_df)}")
    print(f"\nAxon Diameter Statistics (pixels):")
    print(f"  Mean error: {results_df['diameter_error_px'].mean():.2f} px ({results_df['diameter_error_pct'].mean():.2f}%)")
    print(f"  Std dev: {results_df['diameter_error_px'].std():.2f} px")
    print(f"  Min error: {results_df['diameter_error_px'].min():.2f} px ({results_df['diameter_error_pct'].min():.2f}%)")
    print(f"  Max error: {results_df['diameter_error_px'].max():.2f} px ({results_df['diameter_error_pct'].max():.2f}%)")
    
    print(f"\nG-Ratio Statistics:")
    print(f"  Mean error: {results_df['gratio_error'].mean():.4f} ({results_df['gratio_error_pct'].mean():.2f}%)")
    print(f"  Std dev: {results_df['gratio_error'].std():.4f}")
    print(f"  Min error: {results_df['gratio_error'].min():.4f} ({results_df['gratio_error_pct'].min():.2f}%)")
    print(f"  Max error: {results_df['gratio_error'].max():.4f} ({results_df['gratio_error_pct'].max():.2f}%)")
    
    # Save full results
    output_file = os.path.join(dataset_dir, "validation_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\nFull results saved to {output_file}")
    
    # Print first few rows
    print(f"\nFirst 5 measurements:")
    print(results_df[['image_name', 'expected_diameter_px', 'measured_diameter_px', 'diameter_error_pct', 
                       'expected_gratio', 'measured_gratio', 'gratio_error_pct']].head())
    
    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate axon simulations against AxonDeepSeg morphometrics.")
    parser.add_argument("--dataset-dir", type=str, default="axon_dataset",
                        help="Dataset root directory (default: axon_dataset)")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    # Discover all g*/a*/ subdirectories
    subdirs = sorted(glob.glob(os.path.join(dataset_dir, "g*", "a*")))
    if not subdirs:
        print(f"No g*/a*/ subdirectories found in {dataset_dir}")
        exit(1)

    print(f"Found {len(subdirs)} subdirectories in {dataset_dir}\n")

    import re

    all_results = []
    for subdir in subdirs:
        print(f"\n{'='*60}")
        print(f"Processing {subdir}")
        print(f"{'='*60}")

        # Step 1: Create segmentation masks
        create_segmentation_masks(subdir)

        # Step 2: Extract morphometrics
        extract_morphometrics(subdir)

        # Step 3: Clean up extra morphometrics files
        pattern = r'^axon_d\d{3}_g[\d.]+_a[\d.]+_axon_morphometrics\.csv$'
        all_csv = [f for f in os.listdir(subdir) if f.endswith("_axon_morphometrics.csv")]
        extra_csv = [f for f in all_csv if not re.match(pattern, f)]
        for f in extra_csv:
            os.remove(os.path.join(subdir, f))
        if extra_csv:
            print(f"Removed {len(extra_csv)} extra morphometrics files")

        # Step 4: Validate
        results_df = validate_morphometrics(subdir)
        if results_df is not None:
            all_results.append(results_df)

    # Save aggregated results at the top level
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_file = os.path.join(dataset_dir, "validation_results.csv")
        combined.to_csv(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"Aggregated results ({len(combined)} measurements) saved to {output_file}")
