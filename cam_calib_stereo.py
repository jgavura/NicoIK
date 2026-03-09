import cv2
import numpy as np
import glob

# --- PARAMETERS ---
CHECKERBOARD = (8, 5)  # number of inner corners per chessboard row and column
square_size = 0.030    # meters (or consistent unit)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

# --- LOAD PREVIOUSLY SAVED INTRINSICS ---
data_left = np.load("stereo_intrinsics/camera_left_intrinsics.npz")
K_left = data_left["K"]
D_left = data_left["D"]

data_right = np.load("stereo_intrinsics/camera_right_intrinsics.npz")
K_right = data_right["K"]
D_right = data_right["D"]

# --- PREPARE 3D OBJECT POINTS ---
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float64)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# --- COLLECT OBJECT POINTS AND IMAGE POINTS ---
objpoints = []       # 3D points in real world space
imgpoints_left = []  # 2D points in left images
imgpoints_right = [] # 2D points in right images

# stereo image pairs folder
left_images = sorted(glob.glob("stereo_calib_9x6_30mm/left/*.jpg"))
right_images = sorted(glob.glob("stereo_calib_9x6_30mm/right/*.jpg"))

if len(left_images) != len(right_images):
    raise ValueError("Left and right image counts do not match!")

# detect corners for each pair
for fname_left, fname_right in zip(left_images, right_images):
    img_left = cv2.imread(fname_left)
    img_right = cv2.imread(fname_right)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret_left and ret_right:
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (3,3), (-1,-1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (3,3), (-1,-1), criteria)

        objpoints.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

print(f"Found {len(objpoints)} valid stereo pairs.")

# --- STEREO CALIBRATION ---
image_size = gray_left.shape[::-1]  # (width, height)

R = np.zeros((3,3))
T = np.zeros((3,1))

flags = cv2.fisheye.CALIB_FIX_INTRINSIC  # fix intrinsics, only compute extrinsics

objpoints = [op.reshape(-1,1,3).astype(np.float64) for op in objpoints]
imgpoints_left = [pts.reshape(-1,1,2).astype(np.float64) for pts in imgpoints_left]
imgpoints_right = [pts.reshape(-1,1,2).astype(np.float64) for pts in imgpoints_right]

# # debug:
# print(f'objpoints len: {len(objpoints)}')
# print(f'objpoints[0] shape: {objpoints[0].shape}')
# print(f'imgpoints_left len: {len(imgpoints_left)}')
# print(f'imgpoints_left[0] shape: {imgpoints_left[0].shape}')
# print(f'imgpoints_right len: {len(imgpoints_right)}')
# print(f'imgpoints_right[0] shape: {imgpoints_right[0].shape}')
# print(f'K_left: {K_left}')
# print(f'D_left: {D_left}')
# print(f'K_right: {K_right}')
# print(f'D_right: {D_right}')
# print(f'image_size: {image_size}')

ret, K_left, D_left, K_right, D_right, R, T, E, F = cv2.fisheye.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    K_left,
    D_left,
    K_right,
    D_right,
    image_size,
    R,
    T,
    flags=flags,
    criteria=criteria
)

print("Stereo calibration done.")
print("Rotation between cameras R:\n", R)
print("Translation between cameras T:\n", T)
print(f"Root mean square reprojection error: {ret:.4f} px")

# 1. Výpočet rektifikácie
# balance=0.0: zoom in to only see pixels that were in the original image (less stretching)
# balance=1.0: see everything (what you have now)
balance = 0.0 

R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
    K_left, D_left, 
    K_right, D_right, 
    image_size, R, T, 
    flags=0,
    balance=balance,
    fov_scale=0.6 # You can lower this (e.g., 0.8) to zoom in more
)

print(f'Q: {Q}')

# 2. Predvýpočet máp pre remap (toto žerie CPU čas, urobme to len raz)
map1_l, map2_l = cv2.fisheye.initUndistortRectifyMap(
    K_left, D_left, R1, P1, image_size, cv2.CV_16SC2)
map1_r, map2_r = cv2.fisheye.initUndistortRectifyMap(
    K_right, D_right, R2, P2, image_size, cv2.CV_16SC2)

# 3. Uloženie VŠETKÉHO do jedného súboru
np.savez("stereo_intrinsics/stereo_config.npz", 
         # Pôvodné matice (pre istotu)
         K_left=K_left, D_left=D_left, 
         K_right=K_right, D_right=D_right,
         R=R, T=T,
         # Rektifikačné matice (potrebné pre 3D)
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
         # Hotové mapy (pre rýchly remap)
         map1_l=map1_l, map2_l=map2_l,
         map1_r=map1_r, map2_r=map2_r)