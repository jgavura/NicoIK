# juraj

import pybullet as p
from grasper import Grasper
import time


print("Initializing Grasper...")
try:
    grasper = Grasper(
        urdf_path="./urdf/nico_grasper.urdf",
        motor_config="./nico_humanoid_upper_rh7d_ukba.json",
        connect_robot=True,     # Connect to the real robot hardware
        gui=True
    )
    print("Grasper initialized successfully for real robot.")
except Exception as e:
    print(f"Error initializing Grasper for real robot: {e}")

time.sleep(3)

# while True:
grasper.init_position_full()
