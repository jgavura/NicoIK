import cv2
import numpy as np
from stereo_vision import Stereo_vision


config_file = "stereo_intrinsics/stereo_config.npz"
sv = Stereo_vision(config_file)

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
            # --- ACTION: FREEZE AND COMPUTE ---
            print("\nFREEZING... Calculating 3D Depth...")
            is_frozen = True
            
            # Process the frames we just captured
            rect_l, filtered_disp = sv.process_frame(frame_l, frame_r)
            depth_map = sv.get_visual_depth(filtered_disp)
            
            # 1. Show the Rectified "Normal" Image
            cv2.namedWindow(win_rect)
            cv2.setMouseCallback(win_rect, sv.mouse_callback)
            cv2.imshow(win_rect, rect_l)
            
            # 2. Show the Heatmap Image
            cv2.namedWindow(win_depth)
            cv2.setMouseCallback(win_depth, sv.mouse_callback)
            cv2.imshow(win_depth, depth_map)
            
            print("Analysis Ready. You can click on BOTH windows to measure distance.")
        else:
            # --- ACTION: RESUME ---
            print("Resuming Live Feed...")
            is_frozen = False
            cv2.destroyWindow(win_rect)
            cv2.destroyWindow(win_depth)

# Cleanup
cap_l.release()
cap_r.release()
cv2.destroyAllWindows()