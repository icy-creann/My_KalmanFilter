#############################################
# 一阶卡尔曼滤波器
# 该滤波仅适用于一维状态估计，无法处理非线性问题
# author:lcy
#############################################

import numpy as np
import matplotlib.pyplot as plt

class KF:
    #初始化
    #K:卡尔曼增益
    #x:最优估计状态值
    #P:最优估计协方差矩阵
    #R:测量噪声协方差矩阵    
    #Q:过程噪声协方差矩阵

    def __init__(self, P, x, R=0, Q=0): 
        self.P = P
        self.x = x
        self.R = R
        self.Q = Q
        self.K = 0 


    # 更新卡尔曼增益
    def update_K(self):
        self.K = self.P / (self.P + self.R)

    #更新最优估计状态值
    #z:测量值
    def update_x(self,z):
        self.x = self.x + self.K * (z - self.x)

    #更新最优估计协方差矩阵
    def update_P(self):
        self.P = (1 - self.K) * self.P
        # 若有状态转移矩阵F，应改为self.x = F * self.x
        self.P = self.P + self.Q  # 过程噪声加入协方差        

    def update_R(self,R):
        self.R = R 

    #更新K卡尔曼系数、X最优估计值、P最优估计误差
    #z:测量值
    #R:测量噪声协方差矩阵
    def update(self, z):
        self.update_K()
        self.update_x(z)
        self.update_P()
    
    def get_x(self):
        return self.x

    def get_K(self):
        return self.K
    
    def get_P(self):
        return self.P
    
    def get_Q(self):
        return self.Q
    
    def get_R(self):
        return self.R
    
if __name__ == '__main__':
    # 生成模拟数据
    np.random.seed(1)
    true_values = np.linspace(5, 10, 100)
    z = true_values + np.random.normal(0, 1, 100) 
    # 初始化卡尔曼滤波器
    kf = KF(P=1, x=5, R=1, Q=0.1)
    kf_results = []
    for i in range(len(z)):
        kf.update(z[i])
        current_x = kf.get_x()  # 使用临时变量存储当前值
        kf_results.append(current_x)  # 保存每次结果
        print(current_x)
    
    # 绘制结果
    plt.plot(z, label='input')
    plt.plot(kf_results, label='KF output')
    plt.plot(true_values, label="expect") 

    # plt.plot([np.average(z)]*len(z), label='average')
    plt.legend()
    plt.show()

#设置最优估计状态值x和最优估计协方差矩阵p，测量协方差矩阵R和噪声协方差矩阵Q
#输入当前的测量值，更新卡尔曼滤波内部参数K，x，P，R
#用get_x获取当前最优估计状态值即滤波结果


