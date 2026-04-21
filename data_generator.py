import random
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
X_START, X_END, X_STEP = -0.4, 0.4, 0.1
Y_START, Y_END, Y_STEP = 0.2, 0.5, 0.1
OUTPUT_FILE = "experiment_vision/coordinates.txt"

def generate_grid_and_plot():
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

    # 2. Save to text file (space-separated)
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(f"Vision experiment - center and stereovision - grid full\n")
            f.write(f"target_x target_y target_z result_x result_y result_z\n")
            for x, y in points:
                f.write(f"{x} {y} {0.05}\n")
        print(f"Successfully saved {len(points)} grid points to {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error writing to file: {e}")
        return

    # 3. Visualization (2D Map)
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]

    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, color='blue', s=30, label='Grid Points')

    # Formatting the plot
    plt.title(f"2D Regular Grid Map ({len(points)} points)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    # Set axis limits with a small margin
    plt.xlim(X_START - 0.1, X_END + 0.1)
    plt.ylim(Y_START - 0.1, Y_END + 0.1)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box') # Keep 1:1 ratio to see the square grid

    print("Opening grid preview...")
    plt.show()

if __name__ == "__main__":
    generate_grid_and_plot()