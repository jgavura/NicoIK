import cv2
import time
import os
import pybullet as p
from numpy import deg2rad
from ultralytics import YOLO
from camera import Camera


SPEED = 0.03
INIT_POS = {  # both hands down
        'head_z': 0.0, 'head_y': -30.0, 'r_shoulder_z': -30, 'r_shoulder_y': 13,
        'r_arm_x': 0, 'r_elbow_y': 104, 'r_wrist_z': -4, 'r_wrist_x': -55,
        'r_thumb_z': -62, 'r_thumb_x': -180, 'r_indexfinger_x': -170, 'r_middlefingers_x': -180,
        'l_shoulder_z': -30.0, 'l_shoulder_y': 13.0, 'l_arm_x': 0.0, 'l_elbow_y': 104.0,
        'l_wrist_z': -4.0, 'l_wrist_x': -55.0, 'l_thumb_z': -62.0, 'l_thumb_x': -180.0,
        'l_indexfinger_x': -170.0, 'l_middlefingers_x': -180.0
    }
X2Z_COEF, Y2Y_COEF = 0.37582421139136485, 0.3273428034924306
BETA1 = [0.37536988, -0.00526618]
BETA2 = [0.00983655, 0.33090327]


def get_joints_limits(robot_id, num_joints):
    """
    Identify limits, ranges and rest poses of individual robot joints. Uses data from robot model.

    Returns:
        :return [joints_limits_l, joints_limits_u]: (list) Lower and upper limits of all joints
        :return joints_ranges: (list) Ranges of movement of all joints
        :return joints_rest_poses: (list) Rest poses of all joints
    """
    joints_limits_l, joints_limits_u, joints_ranges, joints_rest_poses, joint_names, link_names, joint_indices = [], [], [], [], [], [], []
    for jid in range(num_joints):
        joint_info = p.getJointInfo(robot_id, jid)
        q_index = joint_info[3]
        joint_name = joint_info[1]
        link_name = joint_info[12]
        if q_index > -1:  # Fixed joints have q_index -1
            joint_names.append(joint_info[1].decode("utf-8"))
            link_names.append(joint_info[12].decode("utf-8"))
            joint_indices.append(joint_info[0])
            joints_limits_l.append(joint_info[8])
            joints_limits_u.append(joint_info[9])
            joints_ranges.append(joint_info[9] - joint_info[8])
            joints_rest_poses.append((joint_info[9] + joint_info[8]) / 2)


    return [joints_limits_l,
            joints_limits_u], joints_ranges, joints_rest_poses, joint_names, link_names, joint_indices

def get_real_joints(robot, joints):
    last_position = []

    for k in joints:
        actual = robot.getAngle(k)
        # print("{} : {}, ".format(k,actual),end="")
        last_position.append(actual)
    # print("")

    return last_position

def clamp_angle(joint_name, angle):
    if joint_name not in joint_names:
        raise ValueError(f"Joint '{joint_name}' not found.")
    
    lower_limits, upper_limits = joints_limits
    
    idx = joint_names.index(joint_name)

    lower = lower_limits[idx]
    upper = upper_limits[idx]

    print(f'upper: {upper}')
    print(f'lower: {lower}')

    # Clamp with margin
    if angle > upper:
        return upper - 1
    elif angle < lower:
        return lower + 1
    else:
        return angle

def disable_torque_head(robot):
    robot.disableTorque("head_y")
    robot.disableTorque("head_z")

def enable_torque_head(robot):
    robot.enableTorque("head_y")
    robot.enableTorque("head_z")

def disable_torque_arms(robot, joint_names):
    for joint in joint_names:
        if 'head' not in joint:
            robot.disableTorque(joint)

def enable_torque_arms(robot, joint_names):
    for joint in joint_names:
        if 'head' not in joint:
            robot.enableTorque(joint)

def nicodeg2rad(nicojoints, nicodegrees):
    if isinstance(nicojoints, str):
        nicojoints = [nicojoints]
    if isinstance(nicodegrees, (int, float)):
        nicodegrees = [nicodegrees]

    rads = []

    for nicojoint, nicodegree in zip(nicojoints, nicodegrees):
        if nicojoint == 'r_wrist_z' or nicojoint == 'l_wrist_z':
            rad = deg2rad(nicodegree/2)
        elif nicojoint == 'r_wrist_x' or nicojoint == 'l_wrist_x':
            rad = deg2rad(nicodegree/4)
        else:
            rad = deg2rad(nicodegree)
        rads.append(rad)

    if len(rads) == 1:
        return rads[0]
    return rads

def init_position_full(robot):
        for joint_name, angle in INIT_POS.items():
            robot.setAngle(joint_name, angle, SPEED)

def head_init_position(robot):
    robot.setAngle('head_z', INIT_POS['head_z'], SPEED)
    robot.setAngle('head_y', INIT_POS['head_y'], SPEED)

# Function to save frame
def save_frame(frame, side, timestamp):
    file_name = f"custom_dataset_hand_2/{side}/frame_{timestamp}.jpg"
    cv2.imwrite(file_name, frame)
    print(f"Saved frame to {file_name}")


p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=90, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])

robot_id = p.loadURDF("./urdf/nico.urdf", [0, 0, 0])
# Create table mesh
p.createMultiBody(baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[.30, .45, 0.025],
                                                            rgbaColor=[0.6, 0.6, 0.6, 1]),
                    baseCollisionShapeIndex=p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[.30, .45, 0.025]),
                    baseMass=0, basePosition=[0.26, 0, 0.029])
# Create tablet mesh
p.createMultiBody(baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[.165, .267, 0.02],
                                                            rgbaColor=[0, 0, 0.0, 1]),
                    baseCollisionShapeIndex=p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                                    halfExtents=[.165, .267, 0.02]), baseMass=0,
                    basePosition=[0.395, 0, 0.036])

from nicomotion.Motion import Motion
motorConfig = './nico_humanoid_upper_rh7d_ukba.json'
try:
    robot = Motion(motorConfig=motorConfig)
    print('Robot initialized')
except Exception as e:
    print('Motors are not operational')
    print(e)
    exit()

num_joints = p.getNumJoints(robot_id)
joints_limits, joints_ranges, joints_rest_poses, joint_names, link_names, joint_indices = get_joints_limits(robot_id, num_joints)

model = YOLO("custom_dataset_models/yolo12n_custom_dataset_best.pt")
model.overrides['verbose'] = False   # True for logging in console
camera_right = Camera("right")
camera_left = Camera("left")

# Prepare directories for saving the frames if they don't exist
if not os.path.exists('custom_dataset_hand_2/right'):
    os.makedirs('custom_dataset_hand_2/right')
if not os.path.exists('custom_dataset_hand_2/left'):
    os.makedirs('custom_dataset_hand_2/left')

head_torque_enabled = True
arms_torque_enabled = True
annotate = False

init_position_full(robot)

print("Press 's' to save frames from both cameras.")
print("Press 'a' to toggle arms torque.")
print("Press 'h' to toggle head torque.")
print("Press 'ESC' to exit.")

# Main loop to capture and save frames
while True:
    target_coord_diffs_r = None
    target_coord_diffs_l = None
    
    if annotate:
        target_coord_diffs_r = camera_right.annotate(model, "Tomato", filter_hands=False)
        target_coord_diffs_l = camera_left.annotate(model, "Tomato", filter_hands=False )
    else:
        camera_right.show()
        camera_left.show()

    actual_position = get_real_joints(robot, joint_names)
    for i in range(len(joint_indices)):
        p.resetJointState(robot_id, joint_indices[i], nicodeg2rad(joint_names[i],actual_position[i]))

    # Wait for user input
    key = cv2.waitKey(1) & 0xFF

    # Check if ESC is pressed to exit
    if key == 27:  # ESC key
        break

    # Save both frames if 's' is pressed
    if key == ord('s'):  # Save frames from both cameras
        timestamp = time.time()  # Use timestamp to ensure unique filenames
        save_frame(camera_right.show(), "right", timestamp)
        save_frame(camera_left.show(), "left", timestamp)
    
    if key == ord('a'):  # Toggle arms torque
        if arms_torque_enabled:
            disable_torque_arms(robot, joint_names)
            arms_torque_enabled = False
        else:
            enable_torque_arms(robot, joint_names)
            arms_torque_enabled = True
    if key == ord('h'):  # Toggle head torque
        if head_torque_enabled:
            disable_torque_head(robot)
            head_torque_enabled = False
        else:
            enable_torque_head(robot)
            head_torque_enabled = True
    
    if key == ord('y'):  # Toggle yolo annotate
        annotate = not annotate
    
    if key == ord('i'):  # Initialize head position
        head_init_position(robot)
        enable_torque_head(robot)
        head_torque_enabled = True

    if key == ord('p'):  # Print head position and target diffs
        head_z = robot.getAngle("head_z")
        head_y = robot.getAngle("head_y")
        print(f"head_z, head_y = {head_z} {head_y}")
        head_z_dif = head_z - INIT_POS['head_z']
        head_y_dif = head_y - INIT_POS['head_y']
        print(f"head_z_dif, head_y_dif = {head_z_dif} {head_y_dif}")

        if target_coord_diffs_r:
            x_dif = (target_coord_diffs_r[0] + target_coord_diffs_l[0]) / 2
            y_dif = (target_coord_diffs_r[1] + target_coord_diffs_l[1]) / 2
            print(f"x_dif, y_dif = {x_dif} {y_dif}")
            # print(f"target_coord_diffs_r = {target_coord_diffs_r}, target_coord_diffs_l = {target_coord_diffs_l}")
    
    if key == ord('f'):
        if not target_coord_diffs_r:
            print(f'Yolo not activated or object not found')
            continue

        head_z = robot.getAngle("head_z")
        head_y = robot.getAngle("head_y")
        x_dif = (target_coord_diffs_r[0] + target_coord_diffs_l[0]) / 2
        y_dif = (target_coord_diffs_r[1] + target_coord_diffs_l[1]) / 2

        # head_z_res = min(max(INIT_POS['head_z'] + x_dif * X2Z_COEF, -89), 89)
        # head_y_res = min(max(INIT_POS['head_y'] + y_dif * Y2Y_COEF, -49), 24)

        head_z_res = min(max(INIT_POS['head_z'] + BETA1[0]*x_dif + BETA1[1]*y_dif, -89), 89)
        head_y_res = min(max(INIT_POS['head_y'] + BETA2[0]*x_dif + BETA2[1]*y_dif, -49), 24)

        robot.setAngle('head_z', head_z_res, SPEED)
        robot.setAngle('head_y', head_y_res, SPEED)

        print(f'head_z_res: {head_z_res}')
        print(f'head_y_res: {head_y_res}')

        enable_torque_head(robot)
        head_torque_enabled = True


# Release cameras and close windows
camera_right.release()
camera_left.release()
cv2.destroyAllWindows()
