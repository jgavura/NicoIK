import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle, Patch, Circle


# Custom handler to draw a square with "Z" inside for the legend
class DataTextHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # SCALE FACTOR: Change 1.5 to make it even bigger or smaller
        # This makes the square 50% larger than the default row height
        scale = 1.5
        side = height * scale

        # Center the square horizontally (x) and vertically (y)
        # y_offset will be negative, making the square overflow into the row padding
        x_offset = (width - side) / 2
        y_offset = (height - side) / 2

        # Create the square patch
        rect = Rectangle([xdescent + x_offset, ydescent + y_offset], side, side,
                         facecolor='gray', edgecolor='black', transform=trans)

        # Create the "Z" text centered over the square
        txt = plt.Text(xdescent + width / 2, ydescent + height / 2, "Z",
                       color='white', weight='bold', fontsize=fontsize * 0.85,
                       ha="center", va="center", transform=trans)
        return [rect, txt]


def create_precision_plot(file_path, show_z_values=False):
    # Data containers
    targets_red_x, targets_red_y = [], []
    targets_green_x, targets_green_y = [], []
    hits_data = []  # Stores (tx, ty, hx, hy, hz) for lines and hits
    deviations_2d = []
    measured_z_values = []

    # 1. Load and parse the data
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()[2:]  # Skip first 2 headers
            for line in lines:
                parts = line.split()
                if len(parts) < 4: continue

                # Conversion to CM and flip Y
                tx = float(parts[0]) * 100
                ty = float(parts[1]) * -100

                hx = float(parts[4]) * 100
                hy = float(parts[5]) * -100
                hz = float(parts[6]) * 100

                # Calculate 2D Euclidean deviation
                deviation_2d = np.sqrt((hx - tx) ** 2 + (hy - ty) ** 2)

                sv_working = parts[8] == 'True'

                if not sv_working:
                    targets_red_x.append(tx)
                    targets_red_y.append(ty)
                else:
                    targets_green_x.append(tx)
                    targets_green_y.append(ty)

                    hits_data.append((tx, ty, hx, hy, hz))
                    deviations_2d.append(deviation_2d)
                    measured_z_values.append(hz)

    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    # 2. Stats calculation
    max_2d, min_2d, mean_2d = np.max(deviations_2d), np.min(deviations_2d), np.mean(deviations_2d)

    if measured_z_values:
        max_z, min_z, mean_z = np.max(measured_z_values), np.min(measured_z_values), np.mean(measured_z_values)
    else:
        max_z = min_z = mean_z = 0.0

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_facecolor('white')
    ax.grid(True, linestyle='-', which='major', color='gray', alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # Tick spacing in cm
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.set_xlim(left=-45, right=45)
    ax.set_ylim(bottom=-55, top=-5)

    # 4. Plotting Data
    # Add Touchscreen Rectangle
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

    # Large size for the plot markers (if Z is shown)
    sq_size_plot = 500 if show_z_values else 80

    # Plot targets using calculated colors
    ax.scatter(targets_red_x, targets_red_y, color='red', marker='s', s=sq_size_plot, zorder=5)
    ax.scatter(targets_green_x, targets_green_y, color='green', marker='s', s=sq_size_plot, zorder=5)

    # Plot Hits, Lines and Z-Text for any target that has hit data
    for tx, ty, hx, hy, hz in hits_data:
        ax.scatter(hx, hy, color='blue', marker='o', s=45, zorder=6)
        ax.plot([tx, hx], [ty, hy], color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=4)

        if show_z_values:
            ax.text(tx, ty, f"{hz:.1f}", color='white', weight='bold',
                    fontsize=11, ha='center', va='center', zorder=10)

    # 5. LEGEND PROXIES
    legend_sq_size = 70
    p1 = ax.scatter([], [], color='red', marker='s', s=legend_sq_size)
    p2 = ax.scatter([], [], color='green', marker='s', s=legend_sq_size)
    p3 = ax.scatter([], [], color='blue', marker='o', s=45)

    # Proxy for the Touchscreen Area
    p4 = ax.scatter([], [], marker='s', s=legend_sq_size, 
                    facecolor='cornflowerblue', edgecolor='cornflowerblue', alpha=0.1,
                    linewidth=1.5)
    p5 = ax.scatter([], [], marker='s', s=legend_sq_size,
                    facecolor='blue', edgecolor='blue', alpha=0.1,
                    linewidth=1.5)
    p6 = ax.scatter([], [], marker='o', s=legend_sq_size,
                    facecolor='none', edgecolor='purple',
                    linewidth=2)

    handles = [p1, p2, p3, p4, p5, p6]
    labels = [
        'Target (2D dev > 10 cm)',
        'Target (2D dev <= 10 cm)',
        'Result',
        'Touchscreen Area',
        'NN Correction Area',
        'Robot Arm Reach'
    ]

    if show_z_values:
        # Add the custom "Z in square" handle
        handles.append(Rectangle((0, 0), 1, 1))
        labels.append('Result Z (cm)')

    # 6. Labels and Title
    ax.set_xlabel('X-axis (cm)', fontsize=16)
    ax.set_ylabel('Y-axis (cm)', fontsize=16)
    ax.set_title('Exp - Vision - Unfocused Stereovision V1 - Grid - Z = 15', fontsize=20, pad=60)
    ax.tick_params(axis='both', labelsize=12)

    # Apply the custom handler to the legend
    if show_z_values:
        ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1.05),
                  handler_map={Rectangle: DataTextHandler()},
                  fontsize=11, frameon=True, edgecolor='black', facecolor='white')
    else:
        ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1.145),
                  handler_map={Rectangle: DataTextHandler()},
                  fontsize=11, frameon=True, edgecolor='black', facecolor='white')

    # 7. Information Boxes
    stats_box = dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black')
    large_box = dict(boxstyle='square,pad=1.2', facecolor='white', edgecolor='black')

    # Top Left: Stats
    stats_text = (f"2D - Max Dev: {max_2d:.2f} cm\n"
                  f"2D - Min Dev: {min_2d:.2f} cm\n"
                  f"2D - Mean Dev: {mean_2d:.2f} cm")

    # Add Z stats if enabled
    if show_z_values:
        stats_text += (f"\nZ - Max: {max_z:.2f} cm\n"
                       f"Z - Min: {min_z:.2f} cm\n"
                       f"Z - Mean: {mean_z:.2f} cm")

        ax.text(0.01, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='left', bbox=stats_box)
    else:
        ax.text(0.006, 1.105, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='left', bbox=stats_box)

    # Pivot point
    # # Convert data coordinates to axes coordinates
    # pivot_axes = ax.transAxes.inverted().transform(ax.transData.transform((-20, -10)))
    #
    # ax.scatter(pivot_axes[0], pivot_axes[1], s=120, color='gray', edgecolor='black', transform=ax.transAxes, zorder=20,
    #            clip_on=False)
    #
    # ax.text(pivot_axes[0], pivot_axes[1] + 0.03, "pivot", transform=ax.transAxes, fontsize=10, ha='center',
    #         va='bottom', clip_on=False)

    ax.scatter(-20, -10, s=120, color='gray', edgecolor='black', zorder=20)

    ax.text(-20, -8, "pivot", fontsize=10, ha='center', va='bottom')

    # Top Center: NICO
    ax.text(0.5, 1.02, "NICO", transform=ax.transAxes, fontsize=16,
            verticalalignment='top', horizontalalignment='center', bbox=large_box)

    plt.subplots_adjust(top=0.75)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()
    # plt.savefig('./experiment_vision/plots/z15/ex_vision_unfocused_stereo_v1_z15_grid.png')
    plt.savefig('./experiment_vision/plots/test2.png')

# Run the function
if __name__ == "__main__":
    create_precision_plot('./experiment_vision/data/z15/unfocused_stereovision_v1_z15_grid_full.txt', show_z_values=True)