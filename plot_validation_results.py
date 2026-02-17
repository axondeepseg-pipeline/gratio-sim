"""Plot validation results."""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Plot validation results.")
parser.add_argument("--dataset-dir", type=str, default="axon_dataset",
                    help="Dataset root directory (default: axon_dataset)")
args = parser.parse_args()

dataset_dir = args.dataset_dir

# Load validation results
results = pd.read_csv(os.path.join(dataset_dir, "validation_results.csv"))

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot gratio error vs axon diameter, grouped by g-ratio and angle
groups = results.groupby(['expected_gratio', 'expected_angle'])
for (gratio, angle), group in groups:
    label = f"g={gratio:.2f}, a={angle:.0f}\u00b0"
    ax.scatter(group['expected_diameter_px'], group['gratio_error_pct'],
               s=100, alpha=0.7, label=label)

# Add zero reference line
ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

# Labels and title
ax.set_xlabel('Axon Diameter (pixels)', fontsize=12)
ax.set_ylabel('G-Ratio Error (%)', fontsize=12)
ax.set_title('G-Ratio Error vs Axon Diameter', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()

# Save figure
plt.tight_layout()
plot_path = os.path.join(dataset_dir, 'gratio_error_vs_diameter.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to {plot_path}")

# Print summary statistics
print(f"\nG-Ratio Error Statistics:")
print(f"  Mean: {results['gratio_error_pct'].mean():.2f}%")
print(f"  Std Dev: {results['gratio_error_pct'].std():.2f}%")
print(f"  Min: {results['gratio_error_pct'].min():.2f}%")
print(f"  Max: {results['gratio_error_pct'].max():.2f}%")
