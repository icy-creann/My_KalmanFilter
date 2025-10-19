import cv2
import numpy as np
from sys import exit

from Python.KF import KF

# This delay will affect the Kalman update rate
DELAY_MSEC = 100

# Arbitrary display params
WINDOW_NAME = 'Kalman Mousetracker [ESC to quit]'
WINDOW_SIZE = 500


class MouseInfo(object):
    '''
    A class to store X,Y points
    '''

    def __init__(self):

        self.x, self.y = -1, -1

    def __str__(self):

        return '%4d %4d' % (self.x, self.y)


def mouseCallback(event, x, y, flags, mouse_info):
    '''
    Callback to update a MouseInfo object with new X,Y coordinates
    '''

    mouse_info.x = x
    mouse_info.y = y


def drawCross(img, center, r, g, b):
    '''
    Draws a cross a the specified X,Y coordinates with color RGB
    '''

    d = 5
    t = 2

    color = (r, g, b)

    ctrx = center[0]
    ctry = center[1]

    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t,
             cv2.LINE_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t,
             cv2.LINE_AA)


def drawLines(img, points, r, g, b):
    '''
    Draws lines
    '''

    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))


def newImage():
    '''
    Returns a new image
    '''

    return np.zeros((500, 500, 3), dtype=np.uint8)


if __name__ == '__main__':

    # 创造一个新的图像
    img = newImage()
    # 创造一个新的窗口
    cv2.namedWindow(WINDOW_NAME)

    # 创造一个新的鼠标信息对象
    # 并将窗口的鼠标回调函数设置为mouseCallback
    mouse_info = MouseInfo()
    cv2.setMouseCallback(WINDOW_NAME, mouseCallback, mouse_info)

    while True:

        if mouse_info.x > 0 and mouse_info.y > 0:
            break

        cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(1) == 27:
            exit(0)

    # 这些变量用于存储鼠标轨迹和卡尔曼滤波结果
    measured_points = []
    kalman_points_x = []
    kalman_points_y = []

    #初始最优估计状态值
    x = mouse_info.x
    y = mouse_info.y
    #初始最优估计协方差矩阵，数值越大,初始估计越不准确，数值最终会回归稳定
    P = 0
    #过程噪声协方差矩阵,数值越大,预测结果越平滑
    Q = 0.1
    #测量噪声协方差矩阵，数值越大，测量结果越不准确
    R = 0.01
    # 初始化卡尔曼滤波器
    kf_x = KF(P=P, x=x, R=R, Q=Q)
    kf_y = KF(P=P, x=y, R=R, Q=Q)

    # 主循环
    while True:

        # 清空图像
        img = newImage()

        # 获取当前鼠标位置
        measured = (mouse_info.x, mouse_info.y)
        measured_points.append(measured)

        # 更新卡尔曼滤波器
        kf_x.update(mouse_info.x)
        kf_y.update(mouse_info.y)

        # 获取卡尔曼滤波结果
        x = kf_x.get_x()
        y = kf_y.get_x()
        kalman_points_x.append(x)
        kalman_points_y.append(y)
        estimated = (x, y)
        
        print(f"kf_x:{kf_x.get_x()}")
        print(f"kf_y:{kf_y.get_x()}")
        print("------------------")
        print(f"x_P:{kf_x.get_P()}")
        print(f"y_P:{kf_y.get_P()}")
        print("------------------")
        print(f"x_Q:{kf_x.get_Q()}")
        print(f"y_Q:{kf_y.get_Q()}")
        print("------------------")
        print(f"x_R:{kf_x.get_R()}")
        print(f"y_R:{kf_y.get_R()}")
        print("------------------")
        print(f"x_K:{kf_x.get_K()}")
        print(f"y_K:{kf_y.get_K()}")
        print("------------------")

        #添加估计点到轨迹
        estimated = [int(c) for c in estimated]
        kalman_points_x.append(x)
        kalman_points_y.append(y)
        kalman_points = list(zip(kalman_points_x, kalman_points_y))
        # 绘制卡尔曼滤波结果
        drawLines(img, kalman_points,   0,   255, 0)
        drawCross(img, estimated,       255, 255, 255)
        drawLines(img, measured_points, 255, 255, 0)
        drawCross(img, measured, 0,   0,   255)

        # 显示图像
        cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
            break