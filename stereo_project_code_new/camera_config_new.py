import cv2
import numpy as np

# 左相机内参
left_camera_matrix = np.array([[3.478981043947528e+02, 0, 2.989918917458671e+02],
                               [0., 3.503310379011750e+02, 2.175443948057622e+02],
                               [0., 0., 1.]])

# 左相机畸变系数:[k1, k2, p1, p2, k3]
left_distortion = np.array([[0.062048257677112, -0.022359891076587, 0.006320394366425, -0.004031254863361, -0.097243480832149]])

# 右相机内参
right_camera_matrix = np.array([[3.535031741808161e+02, 0, 3.147818160691025e+02],
                                [0., 3.551383746108855e+02, 2.144045991984501e+02],
                                [0., 0., 1.]])
# 右相机畸变系数:[k1, k2, p1, p2, k3]
right_distortion = np.array([[0.011586150547927, 0.212751002827209, 0.004802339749220, 0.005711433335803, -0.427515633033089]])


# 旋转矩阵
R = np.array([[0.999442536498922, -0.001685737598511, -0.033343283061955],
              [0.001726234365938, 0.999997807005380, 0.001185790598759],
              [0.033341211008518, -0.001242687884875, 0.999443254704991]])

# 平移向量
T = np.array([[-1.338089109645611e+02], [-0.260685198714778], [-1.321941884559895]])

size = (640, 480)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)