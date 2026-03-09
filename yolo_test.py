from ultralytics import YOLO
import cv2
from grasper import Grasper
from camera import Camera
import time
import pybullet as p
from touchscreen_app import Touchscreen_app


# model = YOLO("yolo/yolov8x-worldv2.pt")
# model.set_classes(["tomato"])

# model = YOLO("yolo/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx")

# model = YOLO("yolo/yolov8s-oiv7.pt")
# model = YOLO("yolo/yolo12s_oidv7v2_best.pt")
model = YOLO("custom_dataset_models/yolo12n_custom_dataset_best.pt")
model.overrides['verbose'] = False   # True for logging in console
camera_right = Camera("right")
camera_left = Camera("left")
target_coord_diffs_r = camera_right.annotate(model, "Tomato")
target_coord_diffs_l = camera_left.annotate(model, "Tomato")


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


# grasper.init_position_full()
# grasper.open_gripper('right')
# grasper.open_gripper('left')
# grasper.move_head(0, -30)
# target_z, target_y = 20, 10
# grasper.move_head(target_z, target_y)

touch_app = Touchscreen_app()

box_id = p.createMultiBody(                                 # point where the eyesight begins on nicos forehead
        baseMass=0, # Set mass to 0 if it's only visual
        baseCollisionShapeIndex=-1, # No collision shape
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0.0, 0.0, 0.8]), # Visual shape only
        basePosition=[0, 0, 0]
    )

box_id2 = p.createMultiBody(                                # target point where nico is looking at on the tablet, where eyesight ends
        baseMass=0, # Set mass to 0 if it's only visual
        baseCollisionShapeIndex=-1, # No collision shape
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0.0, 0.0, 0.8]), # Visual shape only
        basePosition=[0, 0, 0]
    )

box_id3 = p.createMultiBody(
        baseMass=0, # Set mass to 0 if it's only visual
        baseCollisionShapeIndex=-1, # No collision shape
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01]*3, rgbaColor=[1, 0.0, 0.0, 0.8]), # Visual shape only
        basePosition=[0, 0, 0]
    )


# print(f"joint name: {p.getJointInfo(grasper.robot_id, grasper.eyesight_link_index)[1]}")

action = 'placing_object'

grasper.init_position_full()

while True:
    if action == 'placing_object':
        grasper.open_gripper('right')
        grasper.open_gripper('left')
        grasper.init_position_full()
        grasper.move_head(0, -30)
        grasper.open_gripper('right')
        grasper.open_gripper('left')
        touch_app.wait_for_grasp()
        action = 'grasping'

    elif action == 'grasping':
        actual_position = grasper.get_real_joint_angles()
        for i in range(len(grasper.joint_indices)):
            joint_name = grasper.joint_names[i]
            p.resetJointState(grasper.robot_id, grasper.joint_indices[i], grasper.nicodeg2rad(joint_name, actual_position[joint_name]))
        p.stepSimulation()
        
        target_coord_diffs_r = camera_right.annotate(model, "Tomato")
        target_coord_diffs_l = camera_left.annotate(model, "Tomato")

        if target_coord_diffs_r and target_coord_diffs_l:
                head_z = grasper.robot.getAngle("head_z")
                head_y = grasper.robot.getAngle("head_y")

                # print(f"head_z = {head_z}, head_y = {head_y}")

                x_dif = (target_coord_diffs_r[0] + target_coord_diffs_l[0]) / 2
                y_dif = (target_coord_diffs_r[1] + target_coord_diffs_l[1]) / 2
                # print(f"x_dif = {x_dif}, y_dif = {y_dif}")

                if abs(x_dif) < 5 and abs(y_dif) < 5:
                    print(f"Target is close to center")
                    grasper.move_head(head_z, head_y)
                    time.sleep(1)
                    target_pos = grasper.get_target_position()
                    p.resetBasePositionAndOrientation(box_id3, target_pos, [0, 0, 0, 1])

                    if 0.25 <= target_pos[0] <= 0.57 and -0.26 <= target_pos[1] <= 0.26:
                        print(f"Target position {target_pos} is inside of tablet")
                        target_pos_pred = grasper.get_xy2xy_prediction(target_pos[0], target_pos[1])
                        print(f"Target position prediction: {target_pos_pred}")
                        target_pos = [target_pos_pred[0], target_pos_pred[1], 0.05]
                    else:
                        print(f"Target position {target_pos} is outside of tablet")

                    grasper.pick_object(target_pos, [0, 0, 0], 'right', autozpos=True, autoori=True)
                    touch_app.wait_for_drop()
                    action = 'placing_object'
                else:
                    grasper.move_head(head_z + x_dif * 0.2, head_y + y_dif * 0.3)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

        # for i in range(grasper.num_joints+10):
        #     try:
        #         sight_link_pos = p.getLinkState(grasper.robot_id, i)[0]
        #         box_id = p.createMultiBody(
        #             baseMass=0, # Set mass to 0 if it's only visual
        #             baseCollisionShapeIndex=-1, # No collision shape
        #             baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03]*3, rgbaColor=[1, 0.0, 0.0, 0.8]), # Visual shape only
        #             basePosition=sight_link_pos
        #         )
        #     except Exception as e:
        #         print(f"Error creating box for link {i}: {e}")

        sight_link_pos = p.getLinkState(grasper.robot_id, grasper.eyesight_link_index)[0]
        p.resetBasePositionAndOrientation(box_id, sight_link_pos, [0, 0, 0, 1])

        target_pos = grasper.get_target_position()
        p.resetBasePositionAndOrientation(box_id2, target_pos, [0, 0, 0, 1])



    # print(f"z_diff = {sight_link_pos[2] - 0.05}")


camera_right.release()
camera_left.release()
cv2.destroyAllWindows()