import numpy as np
from scipy.interpolate import Rbf
import os


class RbfInterpolator:
    def __init__(self,
                 file_paths_yaw=['rbf/rh_yaw_z5.txt', 'rbf/rh_yaw_z10.txt', 'rbf/rh_yaw_z15.txt'],
                 file_paths_z=['rbf/rh_z.txt'],
                 smooth_yaw=0.1,
                 smooth_z=0.01,
        ):
        """
        Initialize the interpolator..
        """
        self.rbf_yaw = None
        self.rbf_z = None
        self.smooth_yaw = smooth_yaw
        self.smooth_z = smooth_z
        
        self.load_yaw_rbf_from_files(file_paths_yaw)
        self.load_z_rbf_from_files(file_paths_z)
        
        
    def load_yaw_rbf_from_files(self, file_paths):
        """
        Reads text files, filters out comments and empty lines, 
        and prepares the data for RBF.
        """
        all_points = []
        
        for path in file_paths:
            if not os.path.exists(path):
                print(f"Warning: File {path} not found.")
                continue
                
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines or comments
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        # Parse x, y, z, yaw
                        parts = list(map(float, line.split()))
                        if len(parts) == 4:
                            all_points.append(parts)
                    except ValueError:
                        continue
        
        data = np.array(all_points)
        if data.size == 0:
            raise ValueError("No valid data points found in the provided files.")
            
        # Extract columns
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        yaw = data[:, 3]
        
        # Build the RBF model (x, y, z -> yaw)
        # Using multiquadric as it is generally robust for 3D spatial data
        self.rbf_yaw = Rbf(x, y, z, yaw, function='multiquadric', smooth=self.smooth_yaw)
        print(f"RBF model built successfully with {len(data)} points.")
    

    def load_z_rbf_from_files(self, file_paths):
        """
        Reads text files, filters out comments and empty lines, 
        and prepares the data for RBF.
        """
        all_points = []
        
        for path in file_paths:
            if not os.path.exists(path):
                print(f"Warning: File {path} not found.")
                continue
                
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines or comments
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        # Parse x, y, z
                        parts = list(map(float, line.split()))
                        if len(parts) == 3:
                            all_points.append(parts)
                    except ValueError:
                        continue
        
        data = np.array(all_points)
        if data.size == 0:
            raise ValueError("No valid data points found in the provided files.")
            
        # Extract columns
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        
        # Build the RBF model (x, y -> z)
        # Using multiquadric as it is generally robust for 3D spatial data
        self.rbf_z = Rbf(x, y, z, function='multiquadric', smooth=self.smooth_z)
        print(f"RBF model built successfully with {len(data)} points.")


    def predict_yaw(self, x, y, z):
        """
        Predicts the yaw for the given X, Y, Z.
        """
        if self.rbf_yaw is None:
            raise RuntimeError("RBF model not built. Call load_data_from_files first.")
            
        return float(self.rbf_yaw(x, y, z))


    def predict_z(self, x, y):
        """
        Predicts the Z for the given X, Y.
        """
        if self.rbf_z is None:
            raise RuntimeError("RBF model not built. Call load_data_from_files first.")
            
        return float(self.rbf_z(x, y))


# --- Example Usage ---
if __name__ == "__main__":
    # Setup the interpolator
    interpolator = RbfInterpolator()
    
    # Test point
    test_x, test_y, test_z = 0.193, -0.089, 0.12
    
    predicted_yaw = interpolator.predict_yaw(test_x, test_y, test_z)
    print(f"Predicted Yaw: {predicted_yaw:.3f}")

    predicted_z = interpolator.predict_z(test_x, test_y)
    print(f"Predicted Z: {predicted_z:.3f}")