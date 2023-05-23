from scipy.spatial.distance import euclidean  # 用来计算端点之间的欧氏距离
import numpy as np
import imutils
import cv2
import camera_config_new as camera_configs
import math
import time

# 优化1
LOWER_BLUE = np.array([100, 43, 46])
UPPER_BLUE = np.array([124,255,255])
AREA_THRESHOLD = 500

# 优化2
def get_center(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy

def get_distance(threeD, cx, cy):
    distance = math.sqrt(threeD[cy][cx][0] ** 2 + threeD[cy][cx][1] ** 2 + threeD[cy][cx][2] ** 2)
    distance = distance / 1000.0  # mm -> m
    return distance

# 优化3
x_list, y_list, z_list, u_list, v_list = [], [], [], [], []

# 优化4
def process_contours(cnt, frameLL, img1_rectified, img2_rectified):
    area = cv2.contourArea(cnt)
    if area > AREA_THRESHOLD:
        cx, cy = get_center(cnt)
        cv2.circle(frameLL, (cx, cy), 5, (0, 0, 255), -1)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frameLL, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print("中心点坐标：", cx, cy)
        cv2.namedWindow('frameL', 1)
        cv2.imshow('frameL', frameLL)
        grayLL = cv2.cvtColor(img1_rectified,cv2.COLOR_BGR2GRAY)
        grayRR = cv2.cvtColor(img2_rectified,cv2.COLOR_BGR2GRAY)
        filteredImg, dispL = depth_map(grayLL, grayRR)
        threeD = cv2.reprojectImageTo3D(dispL, camera_configs.Q)
        x_list.append(threeD[cy][cx][0])
        y_list.append(threeD[cy][cx][1])
        z_list.append(threeD[cy][cx][2])
        u_list.append(cx)
        v_list.append(cy)
        distance = get_distance(threeD, cx, cy)
        distance_str = '像素点 (%d, %d) 离左相机的深度距离为 %0.3f 米' % (cx,cy, distance)
        print(distance_str)

def call_camera(cv=None, autox=None):
    camera = cv2.VideoCapture(camera_type, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if camera.isOpened() is False:
        print('摄像头调用失败')
        raise AssertionError
    else:
        while True:
            success, frame = camera.read()  # 返回捕获到的RGB
            # image = cv2.flip(frame, 1, dst=None)
            frameLL = frame[0:480, 0:640]
            frameRR = frame[0:480, 640:1280]
            hsv = cv2.cvtColor(frameLL, cv2.COLOR_BGR2HSV)
            cv2.imshow('Camera', frame)
            hsv = cv2.cvtColor(frameLL, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
            start = time.time()  # 计算检测时间
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                process_contours(cnt, frameLL, img1_rectified, img2_rectified)
            if (cv2.waitKey(1) > -1) or (cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1.0):  # 设置关闭条件
                success.release()
                cv2.destroyAllWindows()
                break

def reference_processing():
    # 加载相机
    call_camera()

if __name__ == '__main__':
    # camera_type = set_camera_type()  # 设置相机类型
    camera_type = 0
    reference_processing()
    # 计算欧氏距离与实际距离的比率
