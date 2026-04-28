import cv2
import time
import os
import pybullet as p
from numpy import deg2rad, sin, cos, tan, pi
from ultralytics import YOLO
from camera import Camera
from stereo_vision import StereoVision
from grasper import Grasper


SCENE_PATH = "./urdf/nico_grasper.urdf"
TEXTURE_PATH = "./urdf/textures/table.jpg"

X2Z_COEF, Y2Y_COEF = 0.37582421139136485, 0.3273428034924306
BETA1 = [0.37536988, -0.00526618]
BETA2 = [0.00983655, 0.33090327]

FRAMES_SAVE_DIR = 'custom_dataset_4'
FRAME_STARTING_INDEX = 0

# Function to save frame
def save_frame(frame, side, timestamp):
    file_name = f"{FRAMES_SAVE_DIR}/{side}/frame_{timestamp:02d}.jpg"
    cv2.imwrite(file_name, frame)
    print(f"Saved frame to {file_name}")


def debug_show_detection(frame, u, v, window_name="AI Raw Detection"):
    """
    Draws a crosshair and a circle at the raw coordinates for debugging.
    """
    if frame is None:
        print("Debug Error: Frame is None")
        return

    # Create a copy so we don't modify the source image
    debug_img = frame.copy()
    
    # Coordinates must be integers for drawing
    u, v = int(u), int(v)
    
    # 1. Draw a small solid circle at the center
    cv2.circle(debug_img, (u, v), 5, (0, 0, 255), -1) # Red BGR
    
    # 2. Draw a larger outer circle
    cv2.circle(debug_img, (u, v), 15, (0, 255, 0), 2) # Green BGR
    
    # 3. Draw crosshairs (Vertical and Horizontal lines)
    length = 20
    cv2.line(debug_img, (u - length, v), (u + length, v), (0, 255, 0), 2)
    cv2.line(debug_img, (u, v - length), (u, v + length), (0, 255, 0), 2)

    # 4. Add text with coordinates
    text = f"Raw: {u}, {v}"
    cv2.putText(debug_img, text, (u + 20, v - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 5. Display
    cv2.imshow(window_name, debug_img)

def get_centroid(model, result, target_class_name, lower=False):
    if target_class_name not in model.names.values():
            return None, None

    for box in result.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == target_class_name:
            cx, cy, w, h = box.xywh[0]  # cx, cy = center x, center y
            cx, cy = int(cx), int(cy)

            if lower:
                cy_lower = cy + h // 4
                return cx, cy_lower
            
            return cx, cy
    
    return None, None



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

box_id = p.createMultiBody(                                 # left eye
        baseMass=0, # Set mass to 0 if it's only visual
        baseCollisionShapeIndex=-1, # No collision shape
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0.0, 1, 0.8]), # Visual shape only
        basePosition=[-0.05, 0, 0.0]
    )

box_id2 = p.createMultiBody(                                # target point where nico is looking at on the tablet, where eyesight ends
        baseMass=0, # Set mass to 0 if it's only visual
        baseCollisionShapeIndex=-1, # No collision shape
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0.0, 0.0, 0.8]), # Visual shape only
        basePosition=[0, 0, 0]
    )


# model = YOLO("custom_dataset_models/yolo12n_custom_hands_1+2.pt")
model = YOLO("custom_dataset_models/yolo12n_objects_1+2+3_hands_only_1+2.pt")
model.overrides['verbose'] = False   # True for logging in console
camera_right = Camera("right")
camera_left = Camera("left")

# Prepare directories for saving the frames if they don't exist
if not os.path.exists(f'{FRAMES_SAVE_DIR}/right'):
    os.makedirs(f'{FRAMES_SAVE_DIR}/right')
if not os.path.exists(f'{FRAMES_SAVE_DIR}/left'):
    os.makedirs(f'{FRAMES_SAVE_DIR}/left')

head_torque_enabled = True
arms_torque_enabled = True
annotate = False

grasper.init_position_full()

config_file = "stereo_intrinsics/stereo_config.npz"
sv = StereoVision(config_file)

target_class = 'Tomato'

print("Press 's' to save frames from both cameras.")
print("Press 'a' to toggle arms torque.")
print("Press 'h' to toggle head torque.")
print("Press 'ESC' to exit.")

frame_index = FRAME_STARTING_INDEX
# Main loop to capture and save frames
while True:
    target_coord_diffs_r, frame_r = None, None
    target_coord_diffs_l, cx_l, cy_l, frame_l = None, None, None, None
    head_z, head_y = None, None
    
    if annotate:
        target_coord_diffs_r, result_r, frame_r = camera_right.annotate(model, target_class, only_one_target=True, filter_hands=True)
        target_coord_diffs_l, result_l, frame_l = camera_left.annotate(model, target_class, only_one_target=True, filter_hands=True)
        head_z = grasper.robot.getAngle("head_z")
        head_y = grasper.robot.getAngle("head_y")
    else:
        camera_right.show()
        camera_left.show()

    actual_position = grasper.get_real_joint_angles()
    for i in range(len(grasper.joint_indices)):
        joint_name = grasper.joint_names[i]
        p.resetJointState(grasper.robot_id, grasper.joint_indices[i], grasper.nicodeg2rad(joint_name, actual_position[joint_name]))

    # Wait for user input
    key = cv2.waitKey(1) & 0xFF

    # Check if ESC is pressed to exit
    if key == 27:  # ESC key
        break

    target_pos = grasper.get_target_position(extra_y_tilt=0.0)
    p.resetBasePositionAndOrientation(box_id2, target_pos, [0, 0, 0, 1])

    # Save both frames if 's' is pressed
    if key == ord('s'):  # Save frames from both cameras
        # timestamp = time.time()  # Use timestamp to ensure unique filenames
        save_frame(camera_right.show(), "right", frame_index)
        save_frame(camera_left.show(), "left", frame_index)
        frame_index += 1
    
    if key == ord('a'):  # Toggle arms torque
        if arms_torque_enabled:
            grasper.disable_torque_arms()
            arms_torque_enabled = False
        else:
            grasper.enable_torque_arms()
            arms_torque_enabled = True
    if key == ord('h'):  # Toggle head torque
        if head_torque_enabled:
            grasper.disable_torque_head()
            head_torque_enabled = False
        else:
            grasper.enable_torque_head()
            head_torque_enabled = True
    
    if key == ord('y'):  # Toggle yolo annotate
        annotate = not annotate
    
    if key == ord('i'):  # Initialize position
        grasper.init_position_full()
        grasper.enable_torque_head()
        head_torque_enabled = True
        grasper.enable_torque_arms()
        arms_torque_enabled = True

    # if key == ord('p'):  # Print head position and target diffs
    #     head_z = grasper.robot.getAngle("head_z")
    #     head_y = grasper.robot.getAngle("head_y")
    #     print(f"head_z, head_y = {head_z} {head_y}")
    #     head_z_dif = head_z - INIT_POS['head_z']
    #     head_y_dif = head_y - INIT_POS['head_y']
    #     print(f"head_z_dif, head_y_dif = {head_z_dif} {head_y_dif}")

    #     if target_coord_diffs_r:
    #         x_dif = (target_coord_diffs_r[0] + target_coord_diffs_l[0]) / 2
    #         y_dif = (target_coord_diffs_r[1] + target_coord_diffs_l[1]) / 2
    #         print(f"x_dif, y_dif = {x_dif} {y_dif}")
    #         # print(f"target_coord_diffs_r = {target_coord_diffs_r}, target_coord_diffs_l = {target_coord_diffs_l}")
    
    if key == ord('f'):                         # find using yolo and iteratiions
        start_time = time.time()  # Record the starting time
        timeout_limit = 5.0      # Set timeout in seconds
        timed_out = False
        
        while True:
            if time.time() - start_time > timeout_limit:
                print(f"Timeout reached: Could not center the target within {timeout_limit} seconds.")
                timed_out = True
                head_z = grasper.robot.getAngle("head_z")
                head_y = grasper.robot.getAngle("head_y")
                grasper.move_head(head_z, head_y + 1.0)
                break

            actual_position = grasper.get_real_joint_angles()
            for i in range(len(grasper.joint_indices)):
                joint_name = grasper.joint_names[i]
                p.resetJointState(grasper.robot_id, grasper.joint_indices[i], grasper.nicodeg2rad(joint_name, actual_position[joint_name]))
            p.stepSimulation()
            
            target_coord_diffs_r = camera_right.annotate(model, target_class)[0]
            target_coord_diffs_l = camera_left.annotate(model, target_class)[0]

            if target_coord_diffs_r:
                head_z = grasper.robot.getAngle("head_z")
                head_y = grasper.robot.getAngle("head_y")

                # print(f"head_z = {head_z}, head_y = {head_y}")

                x_dif = (target_coord_diffs_r[0] + target_coord_diffs_l[0]) / 2
                y_dif = (target_coord_diffs_r[1] + target_coord_diffs_l[1]) / 2
                # print(f"x_dif = {x_dif}, y_dif = {y_dif}")

                if abs(x_dif) < 2 and abs(y_dif) < 2:
                    print(f"Target is close to center")
                    grasper.move_head(head_z, head_y)
                    break
                else:
                    grasper.move_head(head_z + x_dif * 0.7, head_y + y_dif * 0.7)

        # grasper.move_head(head_z, head_y + 1.0)
        grasper.enable_torque_head()
        head_torque_enabled = True
    
    if key == ord('d'):             # print coordinates of an object found with yolo
        if not target_coord_diffs_l:
            print(f'Yolo not activated or object not found')
            continue

        cx_l, cy_l = get_centroid(model, result_l, target_class, lower=False)

        print(f'target_coord_diffs_l: {target_coord_diffs_l}')
        print(f'cx_l: {cx_l}')
        print(f'cy_l: {cy_l}')
        print(f'head_z: {head_z}')
        print(f'head_y: {head_y}')
        
        debug_show_detection(frame_l, cx_l, cy_l)

        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "")
    
    if key == ord('m'):             # print modified coordinates of an object found with yolo while focused
        if not target_coord_diffs_l:
            print(f'Yolo not activated or object not found')
            continue

        cx_l, cy_l = get_centroid(model, result_l, target_class, lower=False)

        print(f'target_coord_diffs_l: {target_coord_diffs_l}')
        print(f'cx_l: {cx_l}')
        print(f'cy_l: {cy_l}')
        print(f'head_z: {head_z}')
        print(f'head_y: {head_y}')
        
        debug_show_detection(frame_l, cx_l, cy_l)

        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "")

        # print("Modified torso point:")
        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "focused")
    
    if key == ord('n'):             # print modified coordinates of an object found with yolo while unfocused
        if not target_coord_diffs_l:
            print(f'Yolo not activated or object not found')
            continue

        cx_l, cy_l = get_centroid(model, result_l, target_class, lower=False)

        print(f'target_coord_diffs_l: {target_coord_diffs_l}')
        print(f'cx_l: {cx_l}')
        print(f'cy_l: {cy_l}')
        print(f'head_z: {head_z}')
        print(f'head_y: {head_y}')
        
        debug_show_detection(frame_l, cx_l, cy_l)

        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "")

        # print("Modified torso point:")
        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "unfocused")
    
    if key == ord('k'):             # print coordinates of a place nico is looking at using kinematic chain
        x, y, z = grasper.get_target_position(extra_y_tilt=0.0)
        p.resetBasePositionAndOrientation(box_id, [x, y, z], [0, 0, 0, 1])

        print(f"Target position: X = {x}, Y = {y}, Z = {z}")

        print(f"{y:.3f} {x:.3f} {z:.3f}")

        # modified
        x, y, z = grasper.get_target_position(extra_y_tilt=-3.0)
        p.resetBasePositionAndOrientation(box_id, [x, y, z], [0, 0, 0, 1])

        print(f"Target position modified: X = {x}, Y = {y}, Z = {z}")

        print(f"{y:.3f} {x:.3f} {z:.3f}")
    
    if key == ord('r'):             # print coordinates of right arm
        cx_l, cy_l = get_centroid(model, result_l, 'RightHand', lower=False)

        print(f'target_coord_diffs_l: {target_coord_diffs_l}')
        print(f'cx_l: {cx_l}')
        print(f'cy_l: {cy_l}')
        print(f'head_z: {head_z}')
        print(f'head_y: {head_y}')
        
        debug_show_detection(frame_l, cx_l, cy_l)

        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "")

        # print("Modified torso point:")
        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "unfocused")
    
    if key == ord('g'):             # grasp
        if not target_coord_diffs_l:
            print(f'Yolo not activated or object not found')
            continue

        cx_l, cy_l = get_centroid(model, result_l, target_class, lower=False)

        debug_show_detection(frame_l, cx_l, cy_l)
        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "unfocused")

        # grasper.pick_object(torso_point, [0, 0, 0], 'right', autozpos=True, autoori=True, shift_for_grasping=3.0)
        grasper.pick_object(torso_point, [0, 0, 0], 'right', autozpos=True, autoori=True)
    

    if key == ord('j'):             # grasp with hand correction
        if not target_coord_diffs_l:
            print(f'Yolo not activated or object not found')
            continue

        target_cx_l, target_cy_l = get_centroid(model, result_l, target_class, lower=False)

        # debug_show_detection(frame_l, target_cx_l, target_cy_l)
        target_torso_point = sv.get_object_3d_position(frame_l, frame_r, target_cx_l, target_cy_l, head_z, head_y, "")

        checking_pos = [target_torso_point[0],target_torso_point[1],target_torso_point[2]+0.05]
        grasper.move_arm(checking_pos, [0, 0, 0], 'right', autoori=True)
        # time.sleep(3)

        # arm_cx_l, arm_cy_l = get_centroid(model, result_l, 'RightHand', lower=False)

        # if arm_cx_l:
        #     debug_show_detection(frame_l, arm_cx_l, arm_cy_l)
        #     arm_torso_point = sv.get_object_3d_position(frame_l, frame_r, arm_cx_l, arm_cy_l, head_z, head_y, "")
        # else:
        #     print('Hand not found')
        




# Release cameras and close windows
camera_right.release()
camera_left.release()
cv2.destroyAllWindows()
