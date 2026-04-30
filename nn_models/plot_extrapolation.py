import sys
import os

# add path for tablet_coords_conversion
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from numpy import array
from tensorflow import keras
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D  # <--- Added for custom legend
from tablet_coords_conversion import sim2tab_old

# --- CONFIGURATION ---
X_START, X_END, X_STEP = -0.4, 0.4, 0.1
Y_START, Y_END, Y_STEP = 0.2, 0.5, 0.1
OUTPUT_FILE = "model2_extrapolation_plot.png"


class Nn:
    def __init__(self):
        self.xy2xy_model = keras.models.load_model('xy_to_xy/xy_to_xy_model.keras')
        self.xy2xy_mean_std = {}
        with open("xy_to_xy/xy_to_xy_model_mean_std.txt", "r") as f:
            data = f.read().split('\n')[0].split(' ')
            self.xy2xy_mean_std['x_mean'] = float(data[0])
            self.xy2xy_mean_std['x_std'] = float(data[1])
            self.xy2xy_mean_std['y_mean'] = float(data[2])
            self.xy2xy_mean_std['y_std'] = float(data[3])

        self.xy2xyz_model = keras.models.load_model('xy_to_xyz/xy_to_xyz_model.keras')
        self.xy2xyz_mean_std = {}
        with open("xy_to_xyz/xy_to_xyz_model_mean_std.txt", "r") as f:
            data = f.read().split('\n')[0].split(' ')
            self.xy2xyz_mean_std['x_mean'] = float(data[0])
            self.xy2xyz_mean_std['x_std'] = float(data[1])
            self.xy2xyz_mean_std['y_mean'] = float(data[2])
            self.xy2xyz_mean_std['y_std'] = float(data[3])

    def get_xy2xy_prediction(self, x, y):
        x_mean = self.xy2xy_mean_std['x_mean']
        x_std = self.xy2xy_mean_std['x_std']
        y_mean = self.xy2xy_mean_std['y_mean']
        y_std = self.xy2xy_mean_std['y_std']

        x_tab, y_tab = sim2tab_old(x, y)

        target = array([[x_tab, y_tab]])
        target_norm = (target - x_mean) / x_std

        pred_norm = self.xy2xy_model.predict(target_norm, verbose=0)
        pred = pred_norm * y_std + y_mean

        return pred[0]

    def get_xy2xyz_prediction(self, x, y):
        x_mean = self.xy2xyz_mean_std['x_mean']
        x_std = self.xy2xyz_mean_std['x_std']
        y_mean = self.xy2xyz_mean_std['y_mean']
        y_std = self.xy2xyz_mean_std['y_std']

        x_tab, y_tab = sim2tab_old(x, y)

        target = array([[x_tab, y_tab]])
        target_norm = (target - x_mean) / x_std

        pred_norm = self.xy2xyz_model.predict(target_norm, verbose=0)
        pred = pred_norm * y_std + y_mean

        return pred[0]


def generate_grid_and_plot():
    # Setup Figure and Axes
    fig, ax = plt.subplots(figsize=(10, 6))

    nn = Nn()
    points = []

    # 1. Generate regular grid coordinates
    curr_x = X_START
    while curr_x <= X_END:
        curr_y = Y_START
        while curr_y <= Y_END:
            # Rounding to 3 decimal places for clean output
            points.append((round(curr_x, 3), round(curr_y, 3)))
            curr_y += Y_STEP
        curr_x += X_STEP

    distances_inside = []
    distances_outside = []

    for x, y in points:
        # Get prediction
        pred = nn.get_xy2xy_prediction(y, x)
        pred_y, pred_x = pred[0], pred[1]

        # Convert to cm
        x_plot = x * 100
        y_plot = y * -100

        pred_x_plot = pred_x * 100
        pred_y_plot = pred_y * -100

        # Calculate distance between input point and prediction
        distance = ((pred_x_plot - x_plot) ** 2 + (pred_y_plot - y_plot) ** 2) ** 0.5
        # Check whether the input point is inside the correction area
        is_inside_correction_area = -22 <= x_plot <= 22 and -47 <= y_plot <= -28
        if is_inside_correction_area:
            distances_inside.append(distance)
        else:
            distances_outside.append(distance)

        # Plot original point (red)
        ax.scatter(x_plot, y_plot, color='red', s=40, zorder=5)

        # Plot predicted point (blue)
        ax.scatter(pred_x_plot, pred_y_plot, color='blue', s=40, zorder=5)

        # Plot dashed line between them
        ax.plot([x_plot, pred_x_plot], [y_plot, pred_y_plot], color='gray', linestyle='--', linewidth=1, alpha=0.6,
                zorder=4)

    # Calculate average distances
    avg_inside = sum(distances_inside) / len(distances_inside) if distances_inside else 0
    avg_outside = sum(distances_outside) / len(distances_outside) if distances_outside else 0

    # Statistics textbox
    stats_text = (
        f"Mean prediction deviation\n"
        f"relative to NN correction area:\n"
        f"Inside: {avg_inside:.2f} cm\n"
        f"Outside: {avg_outside:.2f} cm"
    )

    ax.text(
        0.02, 1.18,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            edgecolor='black',
            alpha=0.9
        )
    )

    # --- DRAW AREAS ---
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

    # --- TICK FORMATTING ---
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # --- LEGEND ---
    # Custom handles for a clean and styled legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Input Points'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Predictions'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='cornflowerblue', alpha=0.3, markersize=14,
               markeredgecolor='cornflowerblue', label='Touchscreen Area'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', alpha=0.3, markersize=14,
               markeredgecolor='blue', label='NN Zone Area')
    ]

    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.25),
              fontsize=11, frameon=True, edgecolor='black', facecolor='white')

    # --- PLOT FORMATTING ---
    ax.set_title("NN Model 2 - Extrapolation", fontsize=16, pad=70)
    ax.set_xlabel("X Coordinate (cm)", fontsize=12)
    ax.set_ylabel("Y Coordinate (cm)", fontsize=12)

    # Using the same grid style as the grasping plot
    ax.grid(True, linestyle='-', which='major', color='gray', alpha=0.3)

    # Keep 1:1 ratio to see the square grid
    ax.set_aspect('equal', adjustable='box')

    # Top Center: NICO box
    large_box = dict(boxstyle='square,pad=1.2', facecolor='white', edgecolor='black')
    ax.text(0.52, 1.15, "NICO", transform=ax.transAxes, fontsize=16,
            verticalalignment='top', horizontalalignment='center', bbox=large_box)

    # Adjust layout to make room for the title and legend at the top
    plt.subplots_adjust(top=0.8)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    # plt.show()
    plt.savefig(OUTPUT_FILE)

if __name__ == "__main__":
    generate_grid_and_plot()