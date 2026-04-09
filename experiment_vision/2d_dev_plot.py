import matplotlib.pyplot as plt
import numpy as np
from NicoIK.tablet_coords_conversion import sim2tab, pix2cm

targets, hits = [], []

# 1 sec
file_name = f'../experiments/1s/exp_grid/grid_position_target.txt'
with open(file_name, 'r') as f:
    content = f.read().split('\n')[:-1]
    for j in range(len(content)):
        x, y = content[j].split(', ')
        x, y = sim2tab(float(x), float(y))
        targets.append((1920 - x, y))

file_name = f'../experiments/1s/exp_grid/grid_position_result.txt'
with open(file_name, 'r') as f:
    content = f.read().split('\n')[:-1]
    for j in range(len(content)):
        x, y = list(map(float, content[j].split(', ')))
        hits.append((1920 - x, y))

# 2 sec
# file_name = f'../experiments/2s/exp_grid/grid_cartesian_targets.txt'
# with open(file_name, 'r') as f:
#     content = f.read().split('\n')[:-1]
#     for j in range(len(content)):
#         x, y = content[j].split(' ')[:2]
#         x, y = sim2tab(float(x), float(y))
#         targets.append((1920 - x, y))
#
# file_name = f'../experiments/2s/exp_grid/grid_cartesian_results.txt'
# with open(file_name, 'r') as f:
#     content = f.read().split('\n')[:-1]
#     for j in range(len(content)):
#         x, y = list(map(float, content[j].split(' ')))
#         hits.append((1920 - x, y))

# print(targets)
# print(hits)

# Unpack target and attempt positions
target_x, target_y = zip(*targets)
result_x, result_y = zip(*hits)

# Calculate deviations
deviations = [np.sqrt((tx - ax)**2 + (ty - ay)**2) for (tx, ty), (ax, ay) in zip(targets, hits)]
max_deviation = np.max(deviations)
min_deviation = np.min(deviations)
avg_deviation = np.mean(deviations)

# Create plot
fig, ax = plt.subplots(figsize=(19.2, 7.8))

ax.set_axisbelow(True)
ax.set_aspect('equal')

# Plot targets
plt.scatter(target_x, target_y, color='red', label='Targets (cS)', marker='s', zorder=5)

# Plot attempts
plt.scatter(result_x, result_y, color='blue', label='Hits (cR)', zorder=10)

# Draw lines between each target and its corresponding attempt
for (tx, ty), (ax, ay) in zip(targets, hits):
    plt.plot([tx, ax], [ty, ay], color='gray', linestyle='--', zorder=0)

# Set axis limits
plt.xlim(0, 1920)
plt.ylim(300, 1080)

# Add statistics text
stats_text = (f"2D - Max Dev: {round(pix2cm(max_deviation), 2)} cm\n"
              f"2D - Min Dev: {round(pix2cm(min_deviation), 2)} cm\n"
              f"2D - Mean Dev: {round(pix2cm(avg_deviation), 2)} cm")
plt.text(0.01, 1.08, stats_text, transform=plt.gca().transAxes,
         fontsize=15, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

# Add nico hand
plt.text(0.3, 1.08, "NICO\nArm", transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

# Set the font size of ticks
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Add labels and legend
plt.xlabel('X-axis (px)', fontsize=20)
plt.ylabel('Y-axis (px)', fontsize=20)
plt.title('Grid - Without NN correction - Duration: 1 sec', fontsize=25, pad=20, x=0.65)
plt.legend(fontsize=15)
plt.grid(True)

# plt.tight_layout()
fig.set_tight_layout(True)

# Show plot
# plt.show()
# plt.savefig('../plots/2025_2s/exp_grid_2s_2025.png')
plt.savefig('../plots/paper/grid_1s.eps', format='eps')