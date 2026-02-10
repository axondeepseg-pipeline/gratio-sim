"""Plot validation results."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load validation results
results = pd.read_csv("validation_results.csv")

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot gratio error vs axon diameter
ax.scatter(results['expected_diameter_px'], results['gratio_error_pct'], s=100, alpha=0.7)

# Add zero reference line
ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero error')

# Labels and title
ax.set_xlabel('Axon Diameter (pixels)', fontsize=12)
ax.set_ylabel('G-Ratio Error (%)', fontsize=12)
ax.set_title('G-Ratio Error vs Axon Diameter', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()

# Save figure
plt.tight_layout()
plt.savefig('gratio_error_vs_diameter.png', dpi=150, bbox_inches='tight')
print("Plot saved to gratio_error_vs_diameter.png")

# Print summary statistics
print(f"\nG-Ratio Error Statistics:")
print(f"  Mean: {results['gratio_error_pct'].mean():.2f}%")
print(f"  Std Dev: {results['gratio_error_pct'].std():.2f}%")
print(f"  Min: {results['gratio_error_pct'].min():.2f}%")
print(f"  Max: {results['gratio_error_pct'].max():.2f}%")
