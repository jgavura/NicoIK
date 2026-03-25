import cv2
import numpy as np
from stereo_vision import StereoVision
import pybullet as p


SPEED = 0.03
INIT_POS = {  # both hands down
        'head_z': 0.0, 'head_y': 0.0, 'r_shoulder_z': -30, 'r_shoulder_y': 13,
        'r_arm_x': 0, 'r_elbow_y': 104, 'r_wrist_z': -4, 'r_wrist_x': -55,
        'r_thumb_z': -62, 'r_thumb_x': -180, 'r_indexfinger_x': -170, 'r_middlefingers_x': -180,
        'l_shoulder_z': -30.0, 'l_shoulder_y': 13.0, 'l_arm_x': 0.0, 'l_elbow_y': 104.0,
        'l_wrist_z': -4.0, 'l_wrist_x': -55.0, 'l_thumb_z': -62.0, 'l_thumb_x': -180.0,
        'l_indexfinger_x': -170.0, 'l_middlefingers_x': -180.0
}


def init_position_full(robot):
        for joint_name, angle in INIT_POS.items():
            robot.setAngle(joint_name, angle, SPEED)

def disable_torque_arms(robot, joint_names):
    for joint in joint_names:
        if 'head' not in joint:
            robot.disableTorque(joint)

def enable_torque_arms(robot, joint_names):
    for joint in joint_names:
        if 'head' not in joint:
            robot.enableTorque(joint)


from nicomotion.Motion import Motion
motorConfig = './nico_humanoid_upper_rh7d_ukba.json'
try:
    robot = Motion(motorConfig=motorConfig)
    print('Robot initialized')
except Exception as e:
    print('Motors are not operational')
    print(e)
    exit()


init_position_full(robot)


config_file = "stereo_intrinsics/stereo_config.npz"
sv = StereoVision(config_file)

# Camera Initialization
cap_l = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap_r = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# State Management
is_frozen = False
win_live = "LIVE FEED (Raw)"
win_rect = "ANALYSIS - Rectified View (Click to Measure)"
win_depth = "ANALYSIS - Depth Heatmap (Click to Measure)"

print("Controls:")
print("  [SPACE] - Freeze, Analyze and Measure")
print("  [SPACE] - Close analysis and Resume Live Feed")
print("  [Q]     - Quit")

arms_torque_enabled = True

while True:
    if not is_frozen:
        # --- LIVE STREAM ---
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if not ret_l or not ret_r:
            break

        # Display raw camera images side-by-side
        raw_combined = np.hstack((frame_l, frame_r))
        # Simple resize for preview
        h, w = raw_combined.shape[:2]
        raw_res = cv2.resize(raw_combined, (w//2, h//2))
        cv2.imshow(win_live, raw_res)
        
    # Key handling
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    elif key == ord(' '):
        if not is_frozen:
            print("\nFREEZING... Calculating 3D Depth...")
            is_frozen = True
            
            # 1. Process frames
            rect_l, filtered_disp = sv.process_frame(frame_l, frame_r)
            depth_map = sv.get_visual_depth(filtered_disp)

            # 2. Crop 20%
            h, w = rect_l.shape[:2]
            off_x, off_y = int(w * 0.20), int(h * 0.20)
            rect_l_cropped = rect_l[off_y:-off_y, off_x:-off_x]
            depth_map_cropped = depth_map[off_y:-off_y, off_x:-off_x]

            # 3. ZOOM / RESIZE (e.g., 2x bigger)
            scale = 2.0 
            new_w = int(rect_l_cropped.shape[1] * scale)
            new_h = int(rect_l_cropped.shape[0] * scale)
            
            # Use INTER_CUBIC for better quality on the normal image
            rect_l_zoom = cv2.resize(rect_l_cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            # Use INTER_NEAREST for depth map to keep colors sharp
            depth_map_zoom = cv2.resize(depth_map_cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # 4. Robot state
            head_z = robot.getAngle("head_z")
            head_y = robot.getAngle("head_y")
            
            # Pass offsets AND scale to the callback
            mouse_params = {
                "head_z": head_z,
                "head_y": head_y,
                "off_x": off_x, 
                "off_y": off_y,
                "scale": scale
            }
            
            # 5. Show ZOOMED images
            cv2.namedWindow(win_rect)
            cv2.setMouseCallback(win_rect, sv.mouse_callback, mouse_params)
            cv2.imshow(win_rect, rect_l_zoom)
            
            cv2.namedWindow(win_depth)
            cv2.setMouseCallback(win_depth, sv.mouse_callback, mouse_params)
            cv2.imshow(win_depth, depth_map_zoom)
            
            print(f"Analysis Ready. Zoom: {scale}x. Click to measure.")
        else:
            # --- ACTION: RESUME ---
            print("Resuming Live Feed...")
            is_frozen = False
            cv2.destroyWindow(win_rect)
            cv2.destroyWindow(win_depth)
    
    elif key == ord('a'):  # Toggle arms torque
        joint_names = INIT_POS.keys()

        if arms_torque_enabled:
            disable_torque_arms(robot, joint_names)
            arms_torque_enabled = False
        else:
            enable_torque_arms(robot, joint_names)
            arms_torque_enabled = True

# Cleanup
cap_l.release()
cap_r.release()
cv2.destroyAllWindows()