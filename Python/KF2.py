#############################################
# 二阶卡尔曼滤波器
# 该滤波适用于二维状态估计，可处理非线性问题
# author:lcy
#############################################

import numpy as np
import matplotlib.pyplot as plt


class KF2:
    #初始化
    #K:卡尔曼增益
    #x:最优估计状态值
    #P:最优估计协方差矩阵
    #R:测量噪声协方差矩阵    
    #Q:过程噪声协方差矩阵
    #H:观测矩阵

    def __init__(self, P, x, R=0, Q=0): 
        self.P = P
        self.x = x
        self.R = R
        self.Q = Q
        self.K = 0 
        self.H = np.eye(len(x))  # 观测矩阵