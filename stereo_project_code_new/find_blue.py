
import numpy as np

import cv2

def detect_blue_area():
    # 打开摄像头
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # 读取摄像头画面
        success, frame = camera.read()
        frameLL = frame[0:480, 0:640]
        frameRR = frame[0:480, 640:1280]

        # 转换颜色空间为HSV
        hsv = cv2.cvtColor(frameLL, cv2.COLOR_BGR2HSV)

        # 定义蓝色范围
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([150, 255, 255])

        # 创建蓝色区域掩码
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历轮廓
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

        # 显示画面
        cv2.imshow('frame', frame)

        # 按下q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源
    success.release()
    cv2.destroyAllWindows()

detect_blue_area()