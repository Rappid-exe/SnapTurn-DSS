"""
Create a visual diagram showing the turnaround time calculation process
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Color scheme
colors = {
    'data': '#3498db',
    'process': '#e74c3c',
    'model': '#2ecc71',
    'result': '#f39c12'
}

# 1. Data Flow Diagram
ax1 = axes[0, 0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Data Flow: How Taxi Time is Calculated', fontsize=14, fontweight='bold', pad=20)

# Boxes
boxes = [
    (1, 8, 'ADSB\nTracking', colors['data']),
    (1, 6, 'Gate Time\n23:00:01', colors['data']),
    (1, 4, 'Runway Time\n23:21:48', colors['data']),
    (5, 5, 'Calculate\nDifference', colors['process']),
    (8, 5, 'Taxi Time\n21.78 min', colors['result'])
]

for x, y, text, color in boxes:
    box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=color, linewidth=2)
    ax1.add_patch(box)
    ax1.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows
arrows = [
    ((1.8, 6), (4.2, 5.3)),
    ((1.8, 4), (4.2, 4.7)),
    ((6.6, 5), (7.2, 5))
]

for start, end in arrows:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black')
    ax1.add_patch(arrow)

# Formula
ax1.text(5, 3, 'Formula: runway_time - gate_time', ha='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# 2. Data Cleaning Process
ax2 = axes[0, 1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Data Cleaning Process', fontsize=14, fontweight='bold', pad=20)

# Raw data
ax2.text(5, 9, 'Raw Data: 63,842 operations', ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor=colors['data'], alpha=0.7))

# Cleaning steps
steps = [
    (7.5, 'Remove negative times', '81 removed'),
    (6.0, 'Remove < 2 minutes', '3,449 removed'),
    (4.5, 'Remove > 60 minutes', '1,340 removed')
]

for y, text, removed in steps:
    ax2.text(2, y, text, ha='left', fontsize=10)
    ax2.text(8, y, removed, ha='right', fontsize=9, style='italic', color='red')
    arrow = FancyArrowPatch((5, y+0.5), (5, y-0.3), arrowstyle='->', mutation_scale=15,
                           linewidth=2, color='black')
    ax2.add_patch(arrow)

# Clean data
ax2.text(5, 2.5, 'Clean Data: 58,972 operations', ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor=colors['result'], alpha=0.7))
ax2.text(5, 1.5, '92.4% retention rate', ha='center', fontsize=10, style='italic')

# 3. Feature Engineering
ax3 = axes[1, 0]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Feature Engineering: 18 Input Features', fontsize=14, fontweight='bold', pad=20)

# Feature categories
categories = [
    (1, 8, 'Distance\nFeatures', ['distance', 'shortest path', 'distance_gate', 'distance_long', 'distance_else']),
    (4, 8, 'Traffic\nFeatures', ['other_moving_ac', 'total_queue', 'QDepDep', 'QDepArr', 'QArrDep', 'QArrArr']),
    (7, 8, 'Temporal\nFeatures', ['hour', 'is_peak_hour']),
    (1, 4, 'Operation\nFeatures', ['is_departure']),
    (5, 2, 'Target\nVariable', ['taxi_time'])
]

for x, y, title, features in categories:
    # Title box
    box = FancyBboxPatch((x-0.9, y-0.4), 1.8, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=colors['process'], linewidth=2)
    ax3.add_patch(box)
    ax3.text(x, y, title, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Features
    for i, feat in enumerate(features[:3]):  # Show first 3
        ax3.text(x, y-1-i*0.4, f'• {feat}', ha='center', fontsize=7)
    if len(features) > 3:
        ax3.text(x, y-1-3*0.4, f'+ {len(features)-3} more', ha='center', fontsize=7, style='italic')

# 4. Model Prediction Process
ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Random Forest Prediction Process', fontsize=14, fontweight='bold', pad=20)

# Input
ax4.text(5, 9, 'Input: 18 Features', ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor=colors['data'], alpha=0.7))

# Trees
tree_y = 6.5
for i in range(5):
    x = 1.5 + i * 1.8
    # Tree
    triangle = plt.Polygon([(x, tree_y), (x-0.3, tree_y-0.8), (x+0.3, tree_y-0.8)],
                          facecolor=colors['model'], edgecolor='black', linewidth=1.5)
    ax4.add_patch(triangle)
    ax4.text(x, tree_y-0.3, f'T{i+1}', ha='center', fontsize=8, fontweight='bold')
    
    # Prediction
    ax4.text(x, tree_y-1.2, f'{8+i*2}.{i}m', ha='center', fontsize=7)

ax4.text(5, 7.5, '200 Decision Trees', ha='center', fontsize=9, style='italic')

# Average
arrow = FancyArrowPatch((5, 5.5), (5, 4.5), arrowstyle='->', mutation_scale=20,
                       linewidth=3, color='black')
ax4.add_patch(arrow)
ax4.text(5, 5, 'Average', ha='center', fontsize=10, fontweight='bold')

# Output
ax4.text(5, 3.5, 'Predicted Taxi Time', ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor=colors['result'], alpha=0.7))
ax4.text(5, 2.5, '12.3 minutes', ha='center', fontsize=14, fontweight='bold', color='darkred')

# Accuracy
ax4.text(5, 1.5, 'Accuracy: ±1.83 min (MAE)', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Overall title
fig.suptitle('Taxi Time Calculation & Prediction Process\nManchester Airport ADSB Data', 
             fontsize=16, fontweight='bold', y=0.98)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=colors['data'], label='Data/Input'),
    mpatches.Patch(facecolor=colors['process'], label='Processing'),
    mpatches.Patch(facecolor=colors['model'], label='Model'),
    mpatches.Patch(facecolor=colors['result'], label='Output/Result')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
          bbox_to_anchor=(0.5, 0.02))

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig('turnaround_calculation_process.png', dpi=300, bbox_inches='tight')
print("✅ Process diagram saved to 'turnaround_calculation_process.png'")
