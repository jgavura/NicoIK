import cv2
import numpy as np
from nico_coord_transformer import NicoCoordinateTransformer

# --- STEREO VISION CLASS ---
class StereoVision:
    def __init__(self, config_path):
        # 1. Load calibration data
        data = np.load(config_path)
        self.map1_l = data['map1_l']
        self.map2_l = data['map2_l']
        self.map1_r = data['map1_r']
        self.map2_r = data['map2_r']
        self.Q = data['Q']
        self.K_left = data['K_left']
        self.D_left = data['D_left']
        self.R1 = data['R1']
        self.P1 = data['P1']
        
        # 2. Setup SGBM parameters
        self.num_disp = 16 * 6
        self.block_size = 5
        self.l_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disp,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size**2,
            P2=32 * 3 * self.block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=200,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # 3. Setup WLS Filter
        self.r_matcher = cv2.ximgproc.createRightMatcher(self.l_matcher)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.l_matcher)
        self.wls_filter.setLambda(8000.0)
        self.wls_filter.setSigmaColor(1.5)

        # 4. Storage for debug data
        self.last_points_3d = None

        # 5. Set up cv2sim coordiante transformer
        self.transformer = NicoCoordinateTransformer(baseline_m=0.07)

    def process_frame(self, frame_l, frame_r):
        """ Rectifies images and updates 3D point cloud """
        rect_l = cv2.remap(frame_l, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
        
        disp_l = self.l_matcher.compute(gray_l, gray_r)
        disp_r = self.r_matcher.compute(gray_r, gray_l)
        filtered_disp = self.wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r)
        
        self.last_points_3d = cv2.reprojectImageTo3D(filtered_disp.astype(np.float32) / 16.0, self.Q)
        
        return rect_l, filtered_disp

    def get_visual_depth(self, filtered_disp):
        """ Creates a colormapped depth visualization """
        disp_float = filtered_disp.astype(np.float32) / 16.0
        disp_float[disp_float < 0] = 0
        disp_float[disp_float > self.num_disp] = self.num_disp
        
        disp_vis = (disp_float * (255.0 / self.num_disp)).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        disp_color[filtered_disp <= 0] = [0, 0, 0]
        return disp_color

    def mouse_callback(self, event, x, y, flags, param):
        """ Mouse handler to print distance info """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.last_points_3d is not None:
                # Get params
                head_z = param.get("head_z", 0.0)
                head_y = param.get("head_y", 0.0)
                off_x = param.get("off_x", 0)
                off_y = param.get("off_y", 0)
                scale = param.get("scale", 1.0)

                # 1. Reverse the scaling: window(x,y) -> cropped(x,y)
                # We divide by scale to get back to original pixel indices
                crop_x = x / scale
                crop_y = y / scale

                # 2. Reverse the cropping: cropped(x,y) -> full buffer(x,y)
                real_x = int(crop_x + off_x)
                real_y = int(crop_y + off_y)

                # 3. Get 3D point from original buffer
                cv_point = self.last_points_3d[real_y, real_x]
                
                print("-" * 30)
                print(f"Click: Window({x},{y}) -> Buffer({real_x},{real_y})")
                
                dist = np.sqrt(np.sum(cv_point**2))

                if np.isinf(dist) or dist > 20 or cv_point[2] <= 0:
                    print("  Status: INVALID POINT")
                else:
                    print("  [ CAMERA FRAME (Left Eye) ]")
                    print(f"  Z (Depth): {cv_point[2]:.3f} m")
                    print(f"  Direct Distance: {dist:.3f} m")
                    print(f"  3D Coordinates: X={cv_point[0]:.2f}, Y={cv_point[1]:.2f}")

                    head_z = param["head_z"]
                    head_y = param["head_y"]

                    torso_point = self.transformer.transform_cv_point_to_torso(
                        cv_point, head_z, head_y
                    )

                    print("  [ ROBOT TORSO FRAME ]")
                    print(f"    X (Forward): {torso_point[0]:.3f} m")
                    print(f"    Y (Left):    {torso_point[1]:.3f} m")
                    print(f"    Z (Up):      {torso_point[2]:.3f} m")
    
    def get_3d_from_raw_pixel(self, raw_u, raw_v, head_z, head_y):
        """
        Maps a point from a Raw (Fisheye) image to its 3D coordinate.
        :param raw_u: X coordinate in raw image
        :param raw_v: Y coordinate in raw image
        :return: [X, Y, Z] in Torso Base frame
        """
        # 1. Prepare the point for OpenCV (needs to be float32 and shaped (1, 1, 2))
        raw_pt = np.array([[[float(raw_u), float(raw_v)]]], dtype=np.float32)

        # 2. Transform the point from Raw to Rectified
        # We use K_left, D_left, R1, and P1 from your calibration
        # This calculates where that raw pixel 'landed' after rectification
        undistorted_pt = cv2.fisheye.undistortPoints(
            raw_pt, 
            self.K_left, 
            self.D_left, 
            R=self.R1, 
            P=self.P1
        )

        # 3. Extract the new coordinates
        rect_u = int(undistorted_pt[0][0][0])
        rect_v = int(undistorted_pt[0][0][1])

        # 4. Boundary check
        h, w = self.last_points_3d.shape[:2]
        if 0 <= rect_u < w and 0 <= rect_v < h:
            # 5. Get the 3D point from the pre-computed 3D buffer
            cv_point_3d = self.last_points_3d[rect_v, rect_u]
            
            # Check if depth is valid
            if np.isinf(cv_point_3d[2]) or cv_point_3d[2] <= 0:
                return None, None
            
            # 6. Transform to Torso Base using your existing transformer
            torso_point = self.transformer.transform_cv_point_to_torso(
                cv_point_3d, head_z, head_y
            )
            return torso_point, cv_point_3d
        
        return None, None
    
    def get_object_3d_position(self, frame_l, frame_r, raw_u, raw_v, head_z, head_y):
        """
        Calculates the 3D position of an object in sim space.
        
        :param frame_l: Raw left image (Fisheye)
        :param frame_r: Raw right image (Fisheye)
        :param raw_u: Detection X coordinate in raw_l image
        :param raw_v: Detection Y coordinate in raw_l image
        """
        # 1. Update the 3D buffer
        # It populates self.last_points_3d
        self.process_frame(frame_l, frame_r)

        # 2. Use the helper method to get the coordinates
        torso_p, cv_p = self.get_3d_from_raw_pixel(raw_u, raw_v, head_z, head_y)

        # 3. Output logic
        print("-" * 30)
        print(f"AI DETECTION at Raw Coordinates: [{raw_u}, {raw_v}]")
        
        if torso_p is not None:
            dist = np.sqrt(np.sum(cv_p**2))

            print("  [ CAMERA FRAME (Left Eye) ]")
            print(f"  Z (Depth): {cv_p[2]:.3f} m")
            print(f"  Direct Distance: {dist:.3f} m")
            print(f"  3D Coordinates: X={cv_p[0]:.2f}, Y={cv_p[1]:.2f}")

            print("  [ ROBOT TORSO FRAME ]")
            print(f"    X (Forward): {torso_p[0]:.3f} m")
            print(f"    Y (Left):    {torso_p[1]:.3f} m")
            print(f"    Z (Up):      {torso_p[2]:.3f} m")
            
            return torso_p
        else:
            print("  Status: INVALID 3D DATA at detected location.")
            return None