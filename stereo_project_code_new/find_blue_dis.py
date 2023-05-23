from scipy.spatial.distance import euclidean  # 用来计算端点之间的欧氏距离
import numpy as np
import imutils
import cv2
import camera_config_new as camera_configs
import math
import time

x_list = []
y_list = []
z_list = []
u_list = []
v_list = []


def midpoint(ptA, ptB):  # 计算坐标中点函数
    return (ptA[0] + ptB[0]) // 2, (ptA[1] + ptB[1]) // 2


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
            lower_blue = np.array([100, 43, 46])
            upper_blue = np.array([124,255,255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            start = time.time()  # 计算检测时间
            # 寻找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                # 计算轮廓的面积
                area = cv2.contourArea(cnt)

                # 如果面积大于一定值，认为是感兴趣区域
                if area > 500:
                    # 计算轮廓的中心点
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # 画一个红色的点在中心点
                    cv2.circle(frameLL, (cx, cy), 5, (0, 0, 255), -1)

                    # 用蓝色的框体框住感兴趣区域
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frameLL, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # 输出中心点坐标
                    print("中心点坐标：", cx, cy)

                    cv2.namedWindow('frameL', 1)


                    cv2.imshow('frameL', frameLL)

                    img1_rectified = cv2.remap(frameLL, camera_configs.left_map1, camera_configs.left_map2,
                                               cv2.INTER_LINEAR)
                    img2_rectified = cv2.remap(frameRR, camera_configs.right_map1, camera_configs.right_map2,
                                               cv2.INTER_LINEAR)
                    grayLL = cv2.cvtColor(img1_rectified,cv2.COLOR_BGR2GRAY)
                    grayRR = cv2.cvtColor(img2_rectified,cv2.COLOR_BGR2GRAY)

                    filteredImg, dispL = depth_map(grayLL, grayRR)

                    # 将图片扩展至3d空间中，其z方向的值则为当前的深度值
                    threeD = cv2.reprojectImageTo3D(dispL, camera_configs.Q)

                    x_list.append(threeD[cy][cx][0])
                    y_list.append(threeD[cy][cx][1])
                    z_list.append(threeD[cy][cx][2])
                    u_list.append(cx)
                    v_list.append(cy)

                    distance = math.sqrt(threeD[cy][cx][0] ** 2 + threeD[cy][cx][1] ** 2 + threeD[cy][cx][2] ** 2)
                    distance = distance / 1000.0  # mm -> m
                    distance_str = '像素点 (%d, %d) 离左相机的深度距离为 %0.3f 米' % (cx,cy, distance)
                    print(distance_str)

                    end = time.time()  # 时间

                    print("轮廓检测所用时间：{:.3f}ms".format((end - start) * 1000))

            ####

            if (cv2.waitKey(1) > -1) or (cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1.0):  # 设置关闭条件
                success.release()
                cv2.destroyAllWindows()
                break


def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGML and WLS. Need rectified images, returns depth map ( left to right
    disparity )"""
    # SGML Parameters -----------------
    window_size = 3  # size default 3; 5; 7 for SGML reduced size image; 15 for SGML full size image (1300px and
    # above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # size default 3; 5; 7 for SGML reduced size image; 15 for SGML full size image (1300px and above); 5 Works
        # nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    return displ, filteredImg








def reference_processing():
    # 加载相机
    call_camera()


if __name__ == '__main__':


    # camera_type = set_camera_type()  # 设置相机类型
    camera_type = 0


    reference_processing()
    # 计算欧氏距离与实际距离的比率

