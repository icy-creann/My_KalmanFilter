import numpy as np

"""
卡尔曼滤波
使用前需先确定矩阵参数(Fk,Qk,Rk,Hk,X0->Xk,Pk)
"""


class lcy_KF:
    """
    Xk:状态向量
    zk:观测向量
    Fk:状态转移矩阵
    Pk:协方差矩阵
    Qk:过程噪声协方差矩阵
    Rk:观测噪声协方差矩阵
    Hk:观测矩阵
    Kk:卡尔曼增益
    """

    def __init__(self, F, Q, R, H, Xk, Pk, Zk = None): 
        self.Xk = Xk  # 状态向量
        self.Zk = Zk  # 观测向量
        self.F = F  # 状态转移矩阵
        self.Pk = Pk  # 协方差矩阵        
        self.Q = Q  # 过程噪声协方差矩阵
        self.R = R  # 观测噪声协方差矩阵
        self.H = H  # 观测矩阵
        self.Kk = np.zeros_like(Xk)  # 卡尔曼增益

    #先验估计
    def predict(self):
        self.Xk_ = self.F @ self.Xk
        self.Pk_ = self.F @ self.Pk @ self.F.T + self.Q

    def cal_Kk(self):
        Sk = self.H @ self.Pk_ @ self.H.T + self.R  # 计算残差协方差
        self.Kk = self.Pk_ @ self.H.T @ np.linalg.pinv(Sk)  # 计算卡尔曼增益

    #后验估计
    def update(self, z):
        self.Zk = z
        self.Xk = self.Xk_ + self.Kk @ (z - self.H @ self.Xk_)
        self.Pk = (np.eye(len(self.Xk)) - self.Kk @ self.H) @ self.Pk_

    """
    主滤波调用接口
    输入：观测值z
    输出：状态估计值Xk
    """
    def filter(self, z):
        self.predict()
        self.cal_Kk()
        self.update(z)
        return self.Xk

if __name__ == '__main__':
    #模拟输入数据
    input_datas = np.array([[1, 2], [3, 4], [5, 6]])


    # 定义参数
    Fk = np.array([[1, 1], [0, 1]])
    Qk = np.array([[0.1, 0], [0, 0.1]])
    Rk = np.array([[1, 0], [0, 1]])
    Hk = np.array([[1, 0], [0, 1]])
    Xk = np.array([[0], [0]])
    Pk = np.array([[1, 0], [0, 1]])
    Zk = np.array([[0], [0]])
    kf = lcy_KF(Fk, Qk, Rk, Hk, Xk, Pk, Zk)
    # 进行滤波
    output_datas = []
    for z in input_datas:
        Xk = kf.filter(z)
        output_datas.append(Xk)
    output_datas = np.array(output_datas).reshape(-1, 2)
    print("滤波结果：")
    print(output_datas)

