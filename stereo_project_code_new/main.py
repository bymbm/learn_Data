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

            cv2.imshow('Camera', frame)
            # 深度图
            #depthL = frame_process(frameL, frameR)
            cv2.namedWindow('frameL', 1)
            #cv2.imshow('depthL', depthL)
            #####
            start = time.time()  # 计算检测时间
            frameL = cv2.cvtColor(frameLL, cv2.COLOR_BGR2GRAY)  # 转换为灰度
            frameL = cv2.GaussianBlur(frameL, (5, 5), 1.5)  # 高斯滤波去噪点

            frameL = cv2.Canny(frameL, 75, 200)  # Canny边缘检测

            frameL = cv2.dilate(frameL, None, iterations=1)  # 扩张
            frameL = cv2.erode(frameL, None, iterations=1)
            ########## 腐蚀
            frameR = cv2.cvtColor(frameRR, cv2.COLOR_BGR2GRAY)  # 转换为灰度
            frameR = cv2.GaussianBlur(frameR, (5, 5), 1.5)  # 高斯滤波去噪点

            frameR = cv2.Canny(frameR, 75, 200)  # Canny边缘检测

            frameR = cv2.dilate(frameR, None, iterations=1)  # 扩张
            frameR = cv2.erode(frameR, None, iterations=1)  # 腐蚀
            # 轮廓检测
            cnts = cv2.findContours(frameL.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 检测出所有轮廓
            cnts = cnts[1] if imutils.is_cv3() else cnts[0]  # opencv4写法
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:]  # 排序得到前x个轮廓 可以根据图片自己设定

            # 遍历轮廓
            for c in cnts:
                # 计算轮廓近似
                peri = cv2.arcLength(c, True)
                # C表示输入的点集
                # epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
                # True表示封闭的
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                # 4个点的时候就拿出来 因为物品是矩阵形状
                if len(approx) == 4:
                    screenCnt = approx  # 保存下来
                    box = cv.BoxPoints(cv2.minAreaRect(screenCnt)) if imutils.is_cv2() else cv2.boxPoints(
                        cv2.minAreaRect(screenCnt))  # 得到四个最小矩阵的坐标点
                    cv2.drawContours(frameL, [box.astype("int")], -1, (255, 255, 0), 2)  # 在图中画出来
                    box = np.array(box, dtype="int")  # 转换类型
                    (tl, tr, br, bl) = box  # 得到左上 右上 左下 右下的坐标点
                    # 计算中点
                    (_, tltrY) = midpoint(tl, tr)
                    (_, blbrY) = midpoint(bl, br)
                    (_, tlblY) = midpoint(tl, bl)
                    (_, trbrY) = midpoint(tr, br)
                    (x, y) = midpoint(tl, br)
                    # cv2.circle(img, (center_x, center_y), 7, (0,255,0), -1)
                    cv2.circle(frameL, (x, y), 7, 128, -1)
                    print('center_x:', x, 'center_y:', y)
                    cv2.imshow('frameL', frameL)

                    img1_rectified = cv2.remap(frameLL, camera_configs.left_map1, camera_configs.left_map2,
                                               cv2.INTER_LINEAR)
                    img2_rectified = cv2.remap(frameRR, camera_configs.right_map1, camera_configs.right_map2,
                                               cv2.INTER_LINEAR)
                    grayLL = cv2.cvtColor(img1_rectified,cv2.COLOR_BGR2GRAY)
                    grayRR = cv2.cvtColor(img2_rectified,cv2.COLOR_BGR2GRAY)

                    filteredImg, dispL = depth_map(grayLL, grayRR)

                    # 将图片扩展至3d空间中，其z方向的值则为当前的深度值
                    threeD = cv2.reprojectImageTo3D(dispL, camera_configs.Q)

                    x_list.append(threeD[y][x][0])
                    y_list.append(threeD[y][x][1])
                    z_list.append(threeD[y][x][2])
                    u_list.append(x)
                    v_list.append(y)

                    distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2)
                    distance = distance / 1000.0  # mm -> m
                    distance_str = '像素点 (%d, %d) 离左相机的深度距离为 %0.3f 米' % (x, y, distance)
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







def object_length(point1, point2):
    # mm
    length = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)
    return length


def reference_processing():
    # 加载相机
    call_camera()


if __name__ == '__main__':
    cv2.namedWindow("depthL")

    # camera_type = set_camera_type()  # 设置相机类型
    camera_type = 0

    # 图像处理，得到关键点的信息
    reference_processing()
    # 计算欧氏距离与实际距离的比率

    # 实现实时测量

    # 单张图像测量  
    # off_time_processing()
    # cv2.waitKey(0)
