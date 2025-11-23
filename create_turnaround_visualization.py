"""
Create visualization comparing short-haul vs long-haul turnaround times
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from our analysis
arrival_taxi = 6.4  # minutes
departure_taxi = 12.7  # minutes

# Gate operation times (industry standard estimates)
short_haul_gate = 37.5  # 30-45 min average
long_haul_gate = 75.0   # 60-90 min average

# Calculate totals
short_haul_total = arrival_taxi + short_haul_gate + departure_taxi
long_haul_total = arrival_taxi + long_haul_gate + departure_taxi

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))

# Color scheme
colors = {
    'arrival': '#3498db',      # Blue
    'gate': '#e74c3c',         # Red
    'departure': '#2ecc71'     # Green
}

# 1. Stacked Bar Chart
ax1 = plt.subplot(2, 2, 1)
categories = ['Short-Haul\nDomestic', 'Long-Haul\nInternational']
arrival_times = [arrival_taxi, arrival_taxi]
gate_times = [short_haul_gate, long_haul_gate]
departure_times = [departure_taxi, departure_taxi]

x = np.arange(len(categories))
width = 0.5

p1 = ax1.bar(x, arrival_times, width, label='Arrival Taxi', color=colors['arrival'])
p2 = ax1.bar(x, gate_times, width, bottom=arrival_times, label='Gate Operations', color=colors['gate'])
p3 = ax1.bar(x, departure_times, width, bottom=np.array(arrival_times)+np.array(gate_times), 
             label='Departure Taxi', color=colors['departure'])

ax1.set_ylabel('Time (minutes)', fontsize=12, fontweight='bold')
ax1.set_title('Total Turnaround Time Breakdown', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=11)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Add total time labels on top
for i, (cat, total) in enumerate(zip(categories, [short_haul_total, long_haul_total])):
    ax1.text(i, total + 2, f'{total:.1f} min', ha='center', va='bottom', 
             fontsize=11, fontweight='bold')

# 2. Pie Charts
ax2 = plt.subplot(2, 2, 2)
short_haul_data = [arrival_taxi, short_haul_gate, departure_taxi]
labels = ['Arrival Taxi\n6.4 min', 'Gate Operations\n37.5 min', 'Departure Taxi\n12.7 min']
explode = (0.05, 0.1, 0.05)

ax2.pie(short_haul_data, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=[colors['arrival'], colors['gate'], colors['departure']], explode=explode)
ax2.set_title(f'Short-Haul Turnaround\nTotal: {short_haul_total:.1f} minutes', 
              fontsize=13, fontweight='bold')

ax3 = plt.subplot(2, 2, 3)
long_haul_data = [arrival_taxi, long_haul_gate, departure_taxi]
labels = ['Arrival Taxi\n6.4 min', 'Gate Operations\n75.0 min', 'Departure Taxi\n12.7 min']

ax3.pie(long_haul_data, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=[colors['arrival'], colors['gate'], colors['departure']], explode=explode)
ax3.set_title(f'Long-Haul Turnaround\nTotal: {long_haul_total:.1f} minutes', 
              fontsize=13, fontweight='bold')

# 4. Timeline visualization
ax4 = plt.subplot(2, 2, 4)

# Short-haul timeline
y_short = 2
ax4.barh(y_short, arrival_taxi, left=0, height=0.6, color=colors['arrival'], 
         edgecolor='black', linewidth=1.5)
ax4.barh(y_short, short_haul_gate, left=arrival_taxi, height=0.6, color=colors['gate'],
         edgecolor='black', linewidth=1.5)
ax4.barh(y_short, departure_taxi, left=arrival_taxi+short_haul_gate, height=0.6, 
         color=colors['departure'], edgecolor='black', linewidth=1.5)

# Long-haul timeline
y_long = 1
ax4.barh(y_long, arrival_taxi, left=0, height=0.6, color=colors['arrival'],
         edgecolor='black', linewidth=1.5)
ax4.barh(y_long, long_haul_gate, left=arrival_taxi, height=0.6, color=colors['gate'],
         edgecolor='black', linewidth=1.5)
ax4.barh(y_long, departure_taxi, left=arrival_taxi+long_haul_gate, height=0.6,
         color=colors['departure'], edgecolor='black', linewidth=1.5)

# Add labels
ax4.text(arrival_taxi/2, y_short, '6.4', ha='center', va='center', fontsize=9, fontweight='bold')
ax4.text(arrival_taxi + short_haul_gate/2, y_short, '37.5', ha='center', va='center', 
         fontsize=9, fontweight='bold')
ax4.text(arrival_taxi + short_haul_gate + departure_taxi/2, y_short, '12.7', 
         ha='center', va='center', fontsize=9, fontweight='bold')

ax4.text(arrival_taxi/2, y_long, '6.4', ha='center', va='center', fontsize=9, fontweight='bold')
ax4.text(arrival_taxi + long_haul_gate/2, y_long, '75.0', ha='center', va='center',
         fontsize=9, fontweight='bold')
ax4.text(arrival_taxi + long_haul_gate + departure_taxi/2, y_long, '12.7',
         ha='center', va='center', fontsize=9, fontweight='bold')

ax4.set_yticks([y_long, y_short])
ax4.set_yticklabels(['Long-Haul\nInternational', 'Short-Haul\nDomestic'], fontsize=11)
ax4.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
ax4.set_title('Turnaround Timeline Comparison', fontsize=14, fontweight='bold')
ax4.set_xlim(0, max(short_haul_total, long_haul_total) + 5)
ax4.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['arrival'], label='Arrival Taxi (6.4 min)'),
    Patch(facecolor=colors['gate'], label='Gate Operations (30-90 min)'),
    Patch(facecolor=colors['departure'], label='Departure Taxi (12.7 min)')
]
ax4.legend(handles=legend_elements, loc='lower right', fontsize=9)

# Overall title and notes
fig.suptitle('Aircraft Turnaround Time Analysis\nManchester Airport (April-October 2021)', 
             fontsize=16, fontweight='bold', y=0.98)

# Add notes at bottom
notes = """
Notes:
• Taxi times based on actual Manchester Airport data (58,972 operations)
• Gate operation times are industry standard estimates (not in our dataset)
• Short-haul: Domestic/European flights with quick turnarounds
• Long-haul: International flights requiring longer ground operations
• Our model predicts taxi times only (arrival + departure = ~19 min total)
"""
fig.text(0.5, 0.02, notes, ha='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.savefig('turnaround_time_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved to 'turnaround_time_comparison.png'")

# Also create a summary table
print("\n" + "="*70)
print("TURNAROUND TIME SUMMARY")
print("="*70)
print(f"\n{'Component':<25} {'Short-Haul':<15} {'Long-Haul':<15}")
print("-"*70)
print(f"{'Arrival Taxi':<25} {arrival_taxi:>10.1f} min   {arrival_taxi:>10.1f} min")
print(f"{'Gate Operations':<25} {short_haul_gate:>10.1f} min   {long_haul_gate:>10.1f} min")
print(f"{'Departure Taxi':<25} {departure_taxi:>10.1f} min   {departure_taxi:>10.1f} min")
print("-"*70)
print(f"{'TOTAL TURNAROUND':<25} {short_haul_total:>10.1f} min   {long_haul_total:>10.1f} min")
print("="*70)
print(f"\nDifference: {long_haul_total - short_haul_total:.1f} minutes longer for long-haul")
print(f"Gate operations account for {short_haul_gate/short_haul_total*100:.1f}% (short-haul) "
      f"and {long_haul_gate/long_haul_total*100:.1f}% (long-haul) of total time")
print("="*70)
