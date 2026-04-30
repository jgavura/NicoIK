import sys
import os

# add path for tablet_coords_conversion
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import matplotlib.pyplot as plt
from numpy import array
from tensorflow import keras
from tablet_coords_conversion import sim2tab_old


# --- CONFIGURATION ---
X_START, X_END, X_STEP = -0.4, 0.4, 0.1
Y_START, Y_END, Y_STEP = 0.2, 0.5, 0.1
OUTPUT_FILE = "experiment_grasping/data/z5/baseline_z5_grid.txt"


class Nn:
    def __init__(self):
        self.xy2xy_model = keras.models.load_model('nn_models/xy_to_xy/xy_to_xy_model.keras')
        self.xy2xy_mean_std = {}
        with open("nn_models/xy_to_xy/xy_to_xy_model_mean_std.txt", "r") as f:
            data = f.read().split('\n')[0].split(' ')
            self.xy2xy_mean_std['x_mean'] = float(data[0])
            self.xy2xy_mean_std['x_std'] = float(data[1])
            self.xy2xy_mean_std['y_mean'] = float(data[2])
            self.xy2xy_mean_std['y_std'] = float(data[3])

        self.xy2xyz_model = keras.models.load_model('nn_models/xy_to_xyz/xy_to_xyz_model.keras')
        self.xy2xyz_mean_std = {}
        with open("nn_models/xy_to_xyz/xy_to_xyz_model_mean_std.txt", "r") as f:
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
        # print(f"Input target: {target}")
        target_norm = (target - x_mean) / x_std
        # print(f"Normalized target: {target_norm}")

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
        # print(f"Input target: {target}")
        target_norm = (target - x_mean) / x_std
        # print(f"Normalized target: {target_norm}")

        pred_norm = self.xy2xyz_model.predict(target_norm, verbose=0)
        pred = pred_norm * y_std + y_mean

        return pred[0]

def generate_grid_and_plot():
    plt.figure(figsize=(10, 5))
    
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

    first_label = True

    for x, y in points:
        # Získanie predikcie
        pred = nn.get_xy2xyz_prediction(y, x)
        pred_y, pred_x = pred[0], pred[1]

        x_plot = x * 100
        y_plot = y * -100

        pred_x_plot = pred_x * 100
        pred_y_plot = pred_y * -100

        # Vykreslenie pôvodného bodu (modrý)
        plt.scatter(x_plot, y_plot, color='red', s=30, label='Input points' if first_label else "")
        
        # Vykreslenie predikovaného bodu (červený)
        plt.scatter(pred_x_plot, pred_y_plot, color='blue', s=30, label='Predictions' if first_label else "")
        
        # Vykreslenie čiarkovanej čiary medzi nimi
        plt.plot([x_plot, pred_x_plot], [y_plot, pred_y_plot], color='gray', linestyle='--', linewidth=1, alpha=0.6)

        first_label = False

    # Formatting the plot
    plt.title(f"NN Model 3 - Extrapolation")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    plt.tight_layout(pad=1.0)
    
    # Set axis limits with a small margin
    # plt.xlim(X_START - 0.1, X_END + 0.1)
    # plt.ylim(Y_START - 0.1, Y_END + 0.1)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1.02), borderaxespad=0, fontsize=11)
    plt.gca().set_aspect('equal', adjustable='box') # Keep 1:1 ratio to see the square grid

    # print("Opening grid preview...")
    # plt.show()
    plt.savefig('nn_models/model3_extrapolation_plot.png')

if __name__ == "__main__":
    generate_grid_and_plot()