import cv2
import time
import os
import pybullet as p
from numpy import deg2rad, sin, cos, tan, pi, linalg, concatenate
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
align_counter = 0
target_pos = None
target_pos_last = None
hidden_target_coord_difs_l, hidden_target_coord_difs_r = None, None
# Main loop to capture and save frames
while True:
    target_coord_difs_r, frame_r = None, None
    target_coord_difs_l, cx_l, cy_l, frame_l = None, None, None, None
    head_z, head_y = None, None
    
    if annotate:
        target_coord_difs_r, result_r, frame_r = camera_right.annotate(model, target_class, only_one_target=True, filter_hands=True)
        target_coord_difs_l, result_l, frame_l = camera_left.annotate(model, target_class, only_one_target=True, filter_hands=True)
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

    looking_at_pos = grasper.get_target_position(extra_y_tilt=0.0)
    p.resetBasePositionAndOrientation(box_id2, looking_at_pos, [0, 0, 0, 1])

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
        grasper.open_gripper('right')
        grasper.init_position_full()
        grasper.enable_torque_head()
        head_torque_enabled = True
        grasper.enable_torque_arms()
        arms_torque_enabled = True

        align_counter = 0
        target_pos = None
        target_pos_last = None
        hidden_target_coord_difs_l, hidden_target_coord_difs_r = None, None

    if key == ord('p'):  # Print some angles
        r_elbow_y = grasper.robot.getAngle("r_elbow_y")
        r_wrist_z = grasper.robot.getAngle("r_wrist_z")
        r_wrist_x = grasper.robot.getAngle("r_wrist_x")
        print(f"r_elbow_y = {r_elbow_y}")
        print(f"r_wrist_z = {r_wrist_z}")
        print(f"r_wrist_x = {r_wrist_x}")
    
    # if key == ord('f'):                         # find using yolo and iteratiions
    #     start_time = time.time()  # Record the starting time
    #     timeout_limit = 5.0      # Set timeout in seconds
    #     timed_out = False
        
    #     while True:
    #         if time.time() - start_time > timeout_limit:
    #             print(f"Timeout reached: Could not center the target within {timeout_limit} seconds.")
    #             timed_out = True
    #             head_z = grasper.robot.getAngle("head_z")
    #             head_y = grasper.robot.getAngle("head_y")
    #             grasper.move_head(head_z, head_y + 1.0)
    #             break

    #         actual_position = grasper.get_real_joint_angles()
    #         for i in range(len(grasper.joint_indices)):
    #             joint_name = grasper.joint_names[i]
    #             p.resetJointState(grasper.robot_id, grasper.joint_indices[i], grasper.nicodeg2rad(joint_name, actual_position[joint_name]))
    #         p.stepSimulation()
            
    #         target_coord_diffs_r = camera_right.annotate(model, target_class)[0]
    #         target_coord_diffs_l = camera_left.annotate(model, target_class)[0]

    #         if target_coord_diffs_r:
    #             head_z = grasper.robot.getAngle("head_z")
    #             head_y = grasper.robot.getAngle("head_y")

    #             # print(f"head_z = {head_z}, head_y = {head_y}")

    #             x_dif = (target_coord_diffs_r[0] + target_coord_diffs_l[0]) / 2
    #             y_dif = (target_coord_diffs_r[1] + target_coord_diffs_l[1]) / 2
    #             # print(f"x_dif = {x_dif}, y_dif = {y_dif}")

    #             if abs(x_dif) < 2 and abs(y_dif) < 2:
    #                 print(f"Target is close to center")
    #                 grasper.move_head(head_z, head_y)
    #                 break
    #             else:
    #                 grasper.move_head(head_z + x_dif * 0.7, head_y + y_dif * 0.7)

    #     # grasper.move_head(head_z, head_y + 1.0)
    #     grasper.enable_torque_head()
    #     head_torque_enabled = True
    
    if key == ord('b'):                         # move head in direction of target
        head_z = grasper.robot.getAngle("head_z")
        head_y = grasper.robot.getAngle("head_y")

        # print(f"head_z = {head_z}, head_y = {head_y}")

        if align_counter > 0:
            x_dif = (hidden_target_coord_difs_r[0] + hidden_target_coord_difs_l[0]) / 2
            y_dif = ((hidden_target_coord_difs_r[1] + hidden_target_coord_difs_l[1]) / 2).item()
        else:
            x_dif = (target_coord_difs_r[0] + target_coord_difs_l[0]) / 2
            y_dif = ((target_coord_difs_r[1] + target_coord_difs_l[1]) / 2).item()
        # print(f"x_dif = {x_dif}, y_dif = {y_dif}")

        if abs(x_dif) < 2 and abs(y_dif) < 2:
            print(f"Target is close to center")
            grasper.move_head(head_z, head_y)
            break
        else:
            grasper.move_head(head_z + x_dif * 0.1, head_y + y_dif * 0.1)
    
    if key == ord('d'):             # print coordinates of an object found with yolo
        if not target_coord_difs_l:
            print(f'Yolo not activated or object not found')
            continue

        cx_l, cy_l = get_centroid(model, result_l, target_class, lower=False)

        print(f'target_coord_diffs_l: {target_coord_difs_l}')
        print(f'cx_l: {cx_l}')
        print(f'cy_l: {cy_l}')
        print(f'head_z: {head_z}')
        print(f'head_y: {head_y}')
        
        debug_show_detection(frame_l, cx_l, cy_l)

        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "")
    
    if key == ord('m'):             # print modified coordinates of an object found with yolo while focused
        if not target_coord_difs_l:
            print(f'Yolo not activated or object not found')
            continue

        cx_l, cy_l = get_centroid(model, result_l, target_class, lower=False)

        print(f'target_coord_diffs_l: {target_coord_difs_l}')
        print(f'cx_l: {cx_l}')
        print(f'cy_l: {cy_l}')
        print(f'head_z: {head_z}')
        print(f'head_y: {head_y}')
        
        debug_show_detection(frame_l, cx_l, cy_l)

        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "")

        # print("Modified torso point:")
        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "focused")
    
    if key == ord('n'):             # print modified coordinates of an object found with yolo while unfocused
        if not target_coord_difs_l:
            print(f'Yolo not activated or object not found')
            continue

        cx_l, cy_l = get_centroid(model, result_l, target_class, lower=False)

        print(f'target_coord_diffs_l: {target_coord_difs_l}')
        print(f'cx_l: {cx_l}')
        print(f'cy_l: {cy_l}')
        print(f'head_z: {head_z}')
        print(f'head_y: {head_y}')
        
        debug_show_detection(frame_l, cx_l, cy_l)

        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "")
        print(f'Z: {torso_point[2]}')

        # print("Modified torso point:")
        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "unfocused", True)
        print(f'Z: {torso_point[2]}')
    
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

        print(f'target_coord_diffs_l: {target_coord_difs_l}')
        print(f'cx_l: {cx_l}')
        print(f'cy_l: {cy_l}')
        print(f'head_z: {head_z}')
        print(f'head_y: {head_y}')
        
        debug_show_detection(frame_l, cx_l, cy_l)

        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "")

        # print("Modified torso point:")
        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "unfocused")
    
    if key == ord('g'):             # grasp
        if not annotate:
            print(f'Yolo not activated or object not found')
            continue

        if align_counter > 0:
            print(f'target_pos_last[:2]: {target_pos_last[:2]}')
            print(f'[target_pos[2]]: {[target_pos[2]]}')

            target_pos = concatenate([target_pos_last[:2], [target_pos[2]]])

            print(f'target_pos: {target_pos}')

            print('Grasping with aligned hand')
            grasper.pick_object(
                target_pos,
                [0, 0, 0],
                'right',
                nn_model=1,
                autozpos=True,
                autoori=True,
                shift_for_grasping=3.0,
                skip_first_step=True
            )

            align_counter = 0
            target_pos = None
            target_pos_last = None
            hidden_target_coord_difs_l, hidden_target_coord_difs_r = None, None
        
        else:
            cx_l, cy_l = get_centroid(model, result_l, target_class, lower=False)

            debug_show_detection(frame_l, cx_l, cy_l)
            torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "unfocused", True)

            x, y, z = torso_point
            print(f'torso point: {x}, {y}, {z}')
            # x_pred, y_pred = grasper.get_xy2xy_prediction(x, y)
            # print(f'predicted point: {x_pred}, {y_pred}')

            z = max(z, 0.03)

            grasper.pick_object([x, y, z], [0, 0, 0], 'right', nn_model=3, autozpos=True, autoori=True)
            # grasper.pick_object([x, y, z], [0, 0, 0], 'right', autozpos=True, autoori=True)
    

    if key == ord('j'):             # align hand with object
        if not annotate:
            print('YOLO not activated')
            continue

        if align_counter == 0:
            hidden_target_coord_difs_l, hidden_target_coord_difs_r = target_coord_difs_l, target_coord_difs_r

            cx_l, cy_l = get_centroid(model, result_l, target_class, lower=False)

            debug_show_detection(frame_l, cx_l, cy_l)
            target_torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "unfocused", True)

            target_torso_point[2] = max(target_torso_point[2], 0.03)

            grasper.move_arm(target_torso_point, [0, 0, 0], 'right', autozpos=True, z_offset=0.05, autoori=True, adjust_palm=True)

            target_pos = target_torso_point
            target_pos_last = target_torso_point
            align_counter += 1
        else:
            cx_l, cy_l = get_centroid(model, result_l, 'RightHand', lower=False)

            if cx_l == None:
                print('RightHand not found, trying left')
                cx_l, cy_l = get_centroid(model, result_l, 'LeftHand', lower=False)
            
            if cx_l == None:
                print('LeftHand not found, continue')
                continue
                
            debug_show_detection(frame_l, cx_l, cy_l)
            target_torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "unfocused", True)

            print('=' * 20)
            print(f'target_pos: {target_pos}')

            print(f'RH coordinates: {target_torso_point}')

            dist = linalg.norm(target_pos[:2] - target_torso_point[:2])
            print(f'dist: {dist}')

            if dist < 0.01:
                print('Hand close to target')
                continue

            error_xy = target_pos[:2] - target_torso_point[:2]
            new_target = target_pos_last.copy()
            new_target[:2] += error_xy

            new_target[2] = max(new_target[2], 0.03)

            print(f'new target: {new_target}')

            grasper.move_arm(new_target, [0, 0, 0], 'right', autozpos=True, z_offset=0.05, autoori=True, adjust_palm=True)

            target_pos_last = new_target
            align_counter += 1

            print('Hand aligned')
    
    if key == ord('1'):             # grasp model 1
        if not annotate:
            print(f'Yolo not activated or object not found')
            continue

        cx_l, cy_l = get_centroid(model, result_l, target_class, lower=False)

        debug_show_detection(frame_l, cx_l, cy_l)
        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "unfocused", True)

        x, y, z = torso_point
        print(f'torso point: {x}, {y}, {z}')
        # x_pred, y_pred = grasper.get_xy2xy_prediction(x, y)
        # print(f'predicted point: {x_pred}, {y_pred}')

        z = max(z, 0.03)

        # grasper.pick_object([x, y, z], [0, 0, 0], 'right', nn_model=3, autozpos=True, autoori=True)
        grasper.pick_object([x, y, z], [0, 0, 0], 'right', autozpos=True, autoori=True)
    
    if key == ord('2'):             # grasp model 2
        if not annotate:
            print(f'Yolo not activated or object not found')
            continue

        cx_l, cy_l = get_centroid(model, result_l, target_class, lower=False)

        debug_show_detection(frame_l, cx_l, cy_l)
        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "unfocused", True)

        x, y, z = torso_point
        print(f'torso point: {x}, {y}, {z}')
        # x_pred, y_pred = grasper.get_xy2xy_prediction(x, y)
        # print(f'predicted point: {x_pred}, {y_pred}')

        z = max(z, 0.03)

        grasper.pick_object([x, y, z], [0, 0, 0], 'right', nn_model=2, autozpos=True, autoori=True)
        # grasper.pick_object([x, y, z], [0, 0, 0], 'right', autozpos=True, autoori=True)
    
    if key == ord('3'):             # grasp model 3
        if not annotate:
            print(f'Yolo not activated or object not found')
            continue

        cx_l, cy_l = get_centroid(model, result_l, target_class, lower=False)

        debug_show_detection(frame_l, cx_l, cy_l)
        torso_point = sv.get_object_3d_position(frame_l, frame_r, cx_l, cy_l, head_z, head_y, "unfocused", True)

        x, y, z = torso_point
        print(f'torso point: {x}, {y}, {z}')
        # x_pred, y_pred = grasper.get_xy2xy_prediction(x, y)
        # print(f'predicted point: {x_pred}, {y_pred}')

        z = max(z, 0.03)

        grasper.pick_object([x, y, z], [0, 0, 0], 'right', nn_model=3, autozpos=True, autoori=True)
        # grasper.pick_object([x, y, z], [0, 0, 0], 'right', autozpos=True, autoori=True)

    
    if key == ord('e'):                     # show end effector position
        grasper.update_loc_of_ee_box()
        grasper.switch_opacitiy_of_ee_box()
        




# Release cameras and close windows
camera_right.release()
camera_left.release()
cv2.destroyAllWindows()
