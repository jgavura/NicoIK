import numpy as np
import os

# Configuration
input_file = "./experiment_vision/focus_stereovision_v1_shifts.txt"

def calculate_polynomial_fit():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # 1. Load data from the text file
    # It automatically skips the header line starting with '#'
    try:
        data = np.loadtxt(input_file)
        if data.size == 0:
            print("Error: The file is empty.")
            return
            
        distances = data[:, 0] # First column
        shifts = data[:, 1]    # Second column
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 2. Perform Polynomial Fit (Degree 2 = Quadratic)
    # This finds 'a', 'b', and 'c' for: shift = a*dist^2 + b*dist + c
    coeffs = np.polyfit(distances, shifts, 2)
    
    a, b, c = coeffs

    # 3. Output results
    print("-" * 40)
    print("POLYNOMIAL REGRESSION RESULTS")
    print("-" * 40)
    print(f"Formula: shift = ({a:.6f} * dist^2) + ({b:.6f} * dist) + ({c:.6f})")
    print("-" * 40)
    print("Copy these values into your StereoVision class:")
    print(f"self.shift_coeffs = [{a:.6f}, {b:.6f}, {c:.6f}]")
    print("-" * 40)

    # 4. Quick verification
    # Calculate error for the furthest point in your data
    test_dist = distances[-1]
    predicted_shift = (a * test_dist**2) + (b * test_dist) + c
    actual_shift = shifts[-1]
    
    print(f"Verification at {test_dist:.3f}m:")
    print(f"  Actual measured shift:    {actual_shift:.4f}m")
    print(f"  Polynomial predicted shift: {predicted_shift:.4f}m")

if __name__ == "__main__":
    calculate_polynomial_fit()