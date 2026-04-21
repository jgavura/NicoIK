import numpy as np


config_path = "stereo_intrinsics/stereo_config.npz"
data = np.load(config_path)

for matrix in ['K_left', 'D_left', 'K_right', 'D_right', 'R', 'T', 'R1', 'R2', 'P1', 'P2', 'Q']:
    print(f"{matrix}:")
    print(data[matrix])
