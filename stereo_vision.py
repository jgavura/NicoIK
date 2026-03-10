import cv2
import numpy as np

# --- STEREO VISION CLASS ---
class Stereo_vision:
    def __init__(self, config_path):
        # 1. Load calibration data
        data = np.load(config_path)
        self.map1_l = data['map1_l']
        self.map2_l = data['map2_l']
        self.map1_r = data['map1_r']
        self.map2_r = data['map2_r']
        self.Q = data['Q']
        
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
                point = self.last_points_3d[y, x]
                dist = np.sqrt(np.sum(point**2))
                
                print("-" * 30)
                print(f"DEBUG INFO at Pixel [{x}, {y}]:")
                if np.isinf(dist) or dist > 20 or point[2] <= 0:
                    print("  Status: INVALID POINT")
                else:
                    print(f"  Z (Depth): {point[2]:.3f} m")
                    print(f"  Direct Distance: {dist:.3f} m")
                    print(f"  3D Coordinates: X={point[0]:.2f}, Y={point[1]:.2f}, Z={point[2]:.2f}")