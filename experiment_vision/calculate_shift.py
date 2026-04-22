import math
import os

# Define input and output filenames
input_file = "./experiment_vision/center_stereovision_v1_grid_full.txt"
output_file = "./experiment_vision/center_stereovision_v1_shifts.txt"

def generate_shift_map():
    data_points = []

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please create it with your data.")
        return

    with open(input_file, 'r') as f:
        # Skip the header lines
        lines = f.readlines()
        
    for line in lines:
        # Skip empty lines or headers
        if not line.strip() or line.startswith("target") or line.startswith("Vision"):
            continue
            
        parts = line.split()
        try:
            # Parsing the columns based on your provided format:
            # [0]target_x [1]target_y [2]target_z [3]_ [4]result_x [5]result_y [6]result_z [7]distance
            tx = float(parts[0])
            ty = float(parts[1])
            rx = float(parts[4])
            ry = float(parts[5])
            raw_dist = float(parts[7])

            # Calculate the absolute 2D shift (error magnitude) in meters
            # This is the distance the point must travel to get from 'result' to 'target'
            dx = rx - tx
            dy = ry - ty
            shift = math.sqrt(dx**2 + dy**2)

            data_points.append((raw_dist, shift))
        except (ValueError, IndexError):
            # Skip lines that don't match the numerical pattern
            continue

    # Sort all points by raw camera distance (ascending)
    data_points.sort(key=lambda x: x[0])

    # Write the cleaned data to the new file
    with open(output_file, 'w') as f:
        f.write("Vision experiment - focus and stereovision - grid full\n")
        f.write("# Raw_Distance_Cam (m) | Absolute_Shift (m)\n")
        for dist, shift in data_points:
            f.write(f"{dist:.4f} {shift:.4f}\n")

    print(f"Success: Processed {len(data_points)} points.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    generate_shift_map()