import cv2
import numpy as np
import glob


SIDE = 'left'

# number of inner corners
CHECKERBOARD = (8,5)
square_size = 0.030  # 30 mm

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= square_size

objpoints = []  # 3D
imgpoints = []  # 2D

images = glob.glob(f'stereo_calib_9x6_30mm/{SIDE}/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        imgpoints.append(corners2)

N_OK = len(objpoints)
K = np.zeros((3,3))
D = np.zeros((4,1))
rvecs = []
tvecs = []

ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
)

print("Fisheye Intrinsic matrix K:\n", K)
print("Fisheye distortion coefficients D:\n", D)

np.savez(f"stereo_intrinsics/camera_{SIDE}_intrinsics.npz", K=K, D=D)


# reprojection error test
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
    
    # uisti sa, že oba array sú rovnakého tvaru
    imgpoints_i = imgpoints[i].reshape(-1,2)
    imgpoints2 = imgpoints2.reshape(-1,2)
    
    error = cv2.norm(imgpoints_i, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)
print("Mean reprojection error:", mean_error)


# # visual test
# for i, fname in enumerate(images):
#     img = cv2.imread(fname)
#     img_vis = img.copy()
    
#     # detegovanie a zobrazenie rohova
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
#     if ret:
#         corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
#         cv2.drawChessboardCorners(img_vis, CHECKERBOARD, corners2, ret)
    
#         # UNDISTORT IMAGE
#         h, w = img.shape[:2]
#         map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w,h), cv2.CV_16SC2)
#         undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    
#         # zobraz
#         cv2.imshow("Corners + Projection", img_vis)
#         cv2.imshow("Undistorted", undistorted)
#         print(f"Zobrazený obrázok {i+1}/{len(images)}: {fname}")
#         cv2.waitKey(0)  # stlačením klávesu pokračuješ na ďalší
# cv2.destroyAllWindows()