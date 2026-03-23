import numpy as np
from scipy.spatial.transform import Rotation as R


class NicoCoordinateTransformer:
    def __init__(self, baseline_m=0.07):
        """
        :param baseline_m: Distance between left and right camera centers (meters).
                           Default is 0.07 (7cm).
        """
        self.baseline = baseline_m
        
        # OFFSETS FROM URDF
        self.torso_to_neck_xyz = np.array([-0.025, 0.0, 0.23])
        self.neck_to_head_xyz = np.array([0.0, 0.0, 0.105])
        self.head_to_eyesight_xyz = np.array([0.1, 0.0, 0.095])
        
        # OFFSET FROM EYESIGHT TO LEFT CAMERA (VERIFIED IN SIMULATOR)
        # X: -2cm backward, Y: +3.5cm (half baseline) left, Z: 0cm height
        self.eyesight_to_l_cam_xyz = np.array([-0.02, 0.035, 0.0])

    def _get_homogeneous_matrix(self, translation, rotation_matrix):
        """Helper to create 4x4 matrix"""
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation
        return T

    def get_cam_to_torso_matrix(self, pan_rad, tilt_rad):
        """
        Builds the kinematic chain: Torso -> Neck -> Head -> Eyesight -> Left Camera
        """
        # 1. Torso to Neck (Rotates around Z axis)
        r_pan = R.from_euler('z', pan_rad).as_matrix()
        T_torso_neck = self._get_homogeneous_matrix(self.torso_to_neck_xyz, r_pan)

        # 2. Neck to Head (Rotates around -Y axis based on URDF <axis xyz="0 -1 0"/>)
        # Therefore, we negate the tilt angle for standard Y-rotation
        r_tilt = R.from_euler('y', -tilt_rad).as_matrix()
        T_neck_head = self._get_homogeneous_matrix(self.neck_to_head_xyz, r_tilt)

        # 3. Head to Eyesight (Static link)
        T_head_sight = self._get_homogeneous_matrix(self.head_to_eyesight_xyz, np.eye(3))

        # 4. Eyesight to Left Camera (Static verified offset)
        T_sight_cam = self._get_homogeneous_matrix(self.eyesight_to_l_cam_xyz, np.eye(3))

        # Combine the chain
        # Final result: How to get from Camera coordinates to Torso coordinates
        T_torso_cam = T_torso_neck @ T_neck_head @ T_head_sight @ T_sight_cam
        return T_torso_cam

    def transform_cv_point_to_torso(self, cv_point, pan_deg, tilt_deg):
        """
        Converts an OpenCV [X, Y, Z] point to Robot Torso [X, Y, Z].
        
        :param cv_point: List or array [x, y, z] in meters from stereo code.
        :param pan_deg: Current motor angle in degrees (Pan).
        :param tilt_deg: Current motor angle in degrees (Tilt).
        :return: [X, Y, Z] in meters relative to robot torso center.
        """
        # STEP 1: Map OpenCV Axis to Robot Local Axis (Standard Computer Vision swap)
        # OpenCV: Z is depth (forward), X is right, Y is down
        # Robot Local (at camera): X is forward, Y is left, Z is up
        # Conversion logic:
        # Robot X = CV Z
        # Robot Y = -CV X
        # Robot Z = -CV Y
        p_robot_local = np.array([cv_point[2], -cv_point[0], -cv_point[1], 1.0])

        # STEP 2: Calculate transformation for current head angles
        T_matrix = self.get_cam_to_torso_matrix(np.radians(pan_deg), np.radians(tilt_deg))

        # STEP 3: Multiply matrix by the point
        p_torso = T_matrix @ p_robot_local

        return p_torso[:3] # Return X, Y, Z