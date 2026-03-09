import numpy as np

# ====== LOAD DATA ======
data = np.loadtxt("pixdif2head_data.txt", skiprows=3)

x1 = data[:, 0]
x2 = data[:, 1]
y1 = data[:, 2]
y2 = data[:, 3]

# ====== FUNCTION: LINEAR FIT WITHOUT INTERCEPT ======
def linear_fit_no_intercept(x, y):
    k = np.sum(x * y) / np.sum(x * x)
    
    y_pred = k * x
    residuals = y - y_pred
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    
    # Outlier condition
    outlier_indices = np.where(np.abs(residuals) > 2 * std_res)[0]
    
    # Collect detailed outlier info
    outlier_details = []
    for idx in outlier_indices:
        outlier_details.append({
            "row_index_loaded_data": int(idx),
            "x": float(x[idx]),
            "y_actual": float(y[idx]),
            "y_predicted": float(y_pred[idx]),
            "residual": float(residuals[idx]),
            "absolute_error": float(abs(residuals[idx]))
        })
    
    return k, residuals, r2, mean_res, std_res, outlier_details


# ====== CALCULATIONS ======
k1, res1, r2_1, mean_res1, std_res1, out1 = linear_fit_no_intercept(x1, y1)
k2, res2, r2_2, mean_res2, std_res2, out2 = linear_fit_no_intercept(x2, y2)

# ====== OUTPUT ======
print("=== RELATIONSHIP: COLUMN 1 → COLUMN 3 ===")
print("Coefficient (k):", k1)
print("R^2:", r2_1)
print("Mean residual:", mean_res1)
print("Residual standard deviation:", std_res1)

print("\nOutliers:")
for o in out1:
    print(o)

print("\n=== RELATIONSHIP: COLUMN 2 → COLUMN 4 ===")
print("Coefficient (k):", k2)
print("R^2:", r2_2)
print("Mean residual:", mean_res2)
print("Residual standard deviation:", std_res2)

print("\nOutliers:")
for o in out2:
    print(o)


import matplotlib.pyplot as plt

# ====== ERROR vs INPUT PLOT ======

plt.figure(figsize=(12,5))

# --- Column 1 → Column 3 ---
plt.subplot(1,2,1)
plt.scatter(x1, np.abs(res1))
plt.xlabel("Input (Column 1)")
plt.ylabel("Absolute Error")
plt.title("Error magnitude vs Input (1 → 3)")
plt.grid(True)

# --- Column 2 → Column 4 ---
plt.subplot(1,2,2)
plt.scatter(x1, np.abs(res2))
plt.xlabel("Input (Column 2)")
plt.ylabel("Absolute Error")
plt.title("Error magnitude vs Input (2 → 4)")
plt.grid(True)

plt.tight_layout()
plt.show()