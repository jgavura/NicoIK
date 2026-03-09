import numpy as np
import matplotlib.pyplot as plt

# ====== LOAD DATA ======
data = np.loadtxt("pixdif2head_data.txt", skiprows=3)

x1 = data[:, 0]
x2 = data[:, 1]
y1 = data[:, 2]
y2 = data[:, 3]

X = np.column_stack((x1, x2))  # combined input matrix


# ====== FUNCTION: MULTIVARIATE LINEAR FIT (NO INTERCEPT) ======
def multivariate_linear_fit(X, y):
    # Solve least squares: y = X @ beta
    beta, residuals_ls, rank, s = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ beta
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
            "x1": float(X[idx, 0]),
            "x2": float(X[idx, 1]),
            "y_actual": float(y[idx]),
            "y_predicted": float(y_pred[idx]),
            "residual": float(residuals[idx]),
            "absolute_error": float(abs(residuals[idx]))
        })

    return beta, residuals, r2, mean_res, std_res, outlier_details


# ====== CALCULATIONS ======
beta1, res1, r2_1, mean_res1, std_res1, out1 = multivariate_linear_fit(X, y1)
beta2, res2, r2_2, mean_res2, std_res2, out2 = multivariate_linear_fit(X, y2)


# ====== OUTPUT ======
print("=== MODEL: (Column 1, Column 2) → Column 3 ===")
print("Coefficients [a, b]:", beta1)
print("Model: y = a*x1 + b*x2")
print("R^2:", r2_1)
print("Mean residual:", mean_res1)
print("Residual standard deviation:", std_res1)

print("\nOutliers:")
for o in out1:
    print(o)

print("\n=== MODEL: (Column 1, Column 2) → Column 4 ===")
print("Coefficients [a, b]:", beta2)
print("Model: y = a*x1 + b*x2")
print("R^2:", r2_2)
print("Mean residual:", mean_res2)
print("Residual standard deviation:", std_res2)

print("\nOutliers:")
for o in out2:
    print(o)


# ====== ERROR vs INPUT PLOTS ======

plt.figure(figsize=(12,5))

# --- Error vs x1 (for y1 model) ---
plt.subplot(1,2,1)
plt.scatter(x1, np.abs(res1))
plt.xlabel("Input x1")
plt.ylabel("Absolute Error (model → y1)")
plt.title("Error magnitude vs x1")
plt.grid(True)

# --- Error vs x2 (for y1 model) ---
plt.subplot(1,2,2)
plt.scatter(x2, np.abs(res1))
plt.xlabel("Input x2")
plt.ylabel("Absolute Error (model → y1)")
plt.title("Error magnitude vs x2")
plt.grid(True)

plt.tight_layout()
plt.show()