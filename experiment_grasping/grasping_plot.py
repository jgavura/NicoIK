import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D


def get_color_for_success(successes, total):
    """Returns a color from red to green based on the success rate."""
    if total == 0:
        return 'gray'

    ratio = successes / total
    if ratio == 0.0:
        return '#e63946'  # Red (0/3)
    elif ratio <= 0.34:
        return '#f4a261'  # Orange (1/3)
    elif ratio <= 0.67:
        return '#a6ff4c'  # Yellow-Green (2/3)
    else:
        return '#2a9d8f'  # Dark Green (3/3)


def create_grasping_plot(file_path):
    # Data containers
    plot_data = []  # Stores tuples: (tx, ty, successes, total)
    total_successes_all = 0
    total_attempts_all = 0

    zone_successes = 0
    zone_attempts = 0

    # 1. Load and parse the data
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Ignore empty lines and comments   
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                # Need at least X, Y, Z and some attempts (minimum 4 columns)
                if len(parts) < 4:
                    continue

                # Conversion to CM and flip Y (kept from original code)
                tx = float(parts[0]) * 100
                ty = float(parts[1]) * -100
                # parts[2] is target_z, not needed for the 2D plot right now

                # Extract attempts (everything from the 4th column onwards)
                attempts = [int(p) for p in parts[3:]]
                successes = sum(attempts)
                total = len(attempts)

                plot_data.append((tx, ty, successes, total))

                total_successes_all += successes
                total_attempts_all += total

                # Check if point is inside NN correction zone
                if -22 <= tx <= 22 and -47 <= ty <= -28:
                    zone_successes += successes
                    zone_attempts += total

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_facecolor('white')
    ax.grid(True, linestyle='-', which='major', color='gray', alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # Tick spacing in cm
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_xlim(left=-45, right=45)
    ax.set_ylim(bottom=-55, top=-15)

    # 3. Draw Touchscreen and NN Rectangle
    touchscreen_rect = Rectangle((-24, -53), 48, 27,
                             linewidth=2, edgecolor='cornflowerblue',
                             facecolor='cornflowerblue', alpha=0.1,
                             linestyle='--', zorder=1)
    ax.add_patch(touchscreen_rect)

    nn_zone_rect = Rectangle((-22, -47), 44, 19,
                             linewidth=2, edgecolor='blue',
                             facecolor='blue', alpha=0.1,
                             linestyle='--', zorder=1)
    ax.add_patch(nn_zone_rect)

    arm_reach_circle = Circle((-5, 0), radius=50,
                              linewidth=2, edgecolor='purple',
                              facecolor='none', linestyle='-.',
                              alpha=0.5, zorder=1)
    ax.add_patch(arm_reach_circle)

    # 4. Plotting Data (circles with text)
    circle_size = 800

    for tx, ty, successes, total in plot_data:
        color = get_color_for_success(successes, total)

        # Draw the circle
        ax.scatter(tx, ty, color=color, marker='o', s=circle_size, edgecolor='black', zorder=5)

        # Add text inside (e.g., "1/3")
        text_label = f"{successes}/{total}"
        # Use white text on dark backgrounds, black otherwise
        text_color = 'white' if (successes / total == 0.0 or successes / total > 0.67) else 'black'

        ax.text(tx, ty, text_label, color=text_color, weight='bold',
                fontsize=11, ha='center', va='center', zorder=10)

    # 5. LEGEND
    # Use Line2D to create custom legend handles
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e63946', markersize=14, markeredgecolor='black',
               label='0 Successes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f4a261', markersize=14, markeredgecolor='black',
               label='1 Success'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#a6ff4c', markersize=14, markeredgecolor='black',
               label='2 Successes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2a9d8f', markersize=14, markeredgecolor='black',
               label='3 Successes (100%)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='cornflowerblue', alpha=0.2, markersize=14,
               markeredgecolor='cornflowerblue', label='Touchscreen Area'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', alpha=0.2, markersize=14,
               markeredgecolor='blue', label='NN Correction Area'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='purple', markersize=14, markeredgewidth=2,
               linestyle='None', label='Robot Arm Reach')
    ]

    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.23),
              fontsize=11, frameon=True, edgecolor='black', facecolor='white')

    # 6. Labels and Title
    ax.set_xlabel('X-axis (cm)', fontsize=16)
    ax.set_ylabel('Y-axis (cm)', fontsize=16)
    ax.set_title('Grasping Experiment - Baseline Model - Grid - Z = 5', fontsize=20, pad=90)
    ax.tick_params(axis='both', labelsize=12)

    # 7. Information Boxes
    stats_box = dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='black')
    large_box = dict(boxstyle='square,pad=1.2', facecolor='white', edgecolor='black')

    # Calculate statistics
    if total_attempts_all > 0:
        overall_success_rate = (total_successes_all / total_attempts_all) * 100
    else:
        overall_success_rate = 0.0

    if zone_attempts > 0:
        zone_success_rate = (zone_successes / zone_attempts) * 100
    else:
        zone_success_rate = 0.0

    # Top Left: Overall Statistics
    stats_text = (f"Total Targets: {len(plot_data)}\n"
                  f"Total Attempts: {total_attempts_all}\n"
                  f"Total Successes: {total_successes_all}\n"
                  f"Overall Success Rate: {overall_success_rate:.1f}%\n"
                  f"NN Area Success Rate: {zone_success_rate:.1f}%")

    ax.text(0.006, 1.2, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='left', bbox=stats_box)

    # Top Center: NICO box
    ax.text(0.5, 1.09, "NICO", transform=ax.transAxes, fontsize=16,
            verticalalignment='top', horizontalalignment='center', bbox=large_box)

    plt.subplots_adjust(top=0.75)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    # plt.savefig('./experiment_grasping/plots/z5/baseline_z5_grid.png')


# Run the function
if __name__ == "__main__":
    # Replace with the path to your .txt file
    create_grasping_plot('./experiment_grasping/data/z5/baseline_z5_grid.txt')