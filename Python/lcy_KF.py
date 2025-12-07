import numpy as np

"""
卡尔曼滤波
使用前需先确定矩阵参数(F,Q,R,H,X0->Xk,Pk)
"""


class lcy_KF:
    """
    Xk:状态向量
    Zk:观测向量
    F:状态转移矩阵
    P:协方差矩阵
    Q:过程噪声协方差矩阵
    R:观测噪声协方差矩阵
    H:观测矩阵
    Kk:卡尔曼增益
    """

    def __init__(self, F, Q, R, H, Xk, Pk, Zk = None): 
        self.Xk = np.array(Xk).reshape(-1, 1)  # 状态向量(nx1)
        self.Zk = Zk  # 观测向量(mx1)
        self.F = np.array(F)  # 状态转移矩阵(nxn)
        self.Pk = np.array(Pk)  # 协方差矩阵(nxn)        
        self.Q = np.array(Q)  # 过程噪声协方差矩阵(nxn)
        self.R = np.array(R)  # 观测噪声协方差矩阵(mxm)
        self.H = np.array(H)  # 观测矩阵(mxn)
        self.n = self.Xk.shape[0]  # 状态向量维度
        self.m = self.H.shape[0]  # 观测向量维度
        self.Kk = np.zeros((self.n, self.m))  # 卡尔曼增益(nxm)
        self.Xk_ = np.zeros_like(self.Xk)  # 先验状态向量(nx1)
        self.Pk_ = np.zeros_like(self.Pk)  # 先验协方差矩阵(nxn)
        self._check_dims()

    def _check_dims(self):
        """检查矩阵维度是否匹配"""
        if self.F.shape != (self.n, self.n):
            raise ValueError(f"状态转移矩阵F维度错误！应为({self.n},{self.n})，实际{self.F.shape}")
        if self.Q.shape != (self.n, self.n) or self.Pk.shape != (self.n, self.n):
            raise ValueError(f"协方差矩阵Q/Pk维度错误！应为({self.n},{self.n})")
        if self.H.shape != (self.m, self.n):
            raise ValueError(f"观测矩阵H维度错误！应为({self.m},{self.n})，实际{self.H.shape}")
        if self.R.shape != (self.m, self.m):
            raise ValueError(f"观测噪声R维度错误！应为({self.m},{self.m})，实际{self.R.shape}")

    #先验估计
    def predict(self):
        self.Xk_ = self.F @ self.Xk
        self.Pk_ = self.F @ self.Pk @ self.F.T + self.Q

    def cal_Kk(self):
        Sk = self.H @ self.Pk_ @ self.H.T + self.R  # 计算残差协方差
        self.Kk = self.Pk_ @ self.H.T @ np.linalg.pinv(Sk)  # 计算卡尔曼增益

    #后验估计
    def update(self, z):
        z = np.array(z).reshape(-1, 1)
        if z.shape[0] != self.m:
            raise ValueError(f"观测向量z维度错误！应为({self.m},1)，实际{z.shape}")
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
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    # 定义参数
    dt  = 0.01  # 时间间隔

    F = np.array([[1, 0, dt, 0],  # x(t)  = x(t-1) + vx(t-1)×dt
                  [0, 1, 0, dt],  # y(t)  = y(t-1) + vy(t-1)×dt
                  [0, 0, 1, 0],   # vx(t) = vx(t-1) 
                  [0, 0, 0, 1]])  # vy(t) = vy(t-1)                    #状态转移矩阵

    Q = np.diag([0.01, 0.01, 0.1, 0.1])                                # 过程噪声协方差矩阵

    R = np.diag([1, 1, 4, 4])                                          # 测量噪声协方差矩阵

    H = np.eye(4)                                                      # 观测矩阵

     # 初始状态
    Xk = np.array([0, 0, 50, 30])                                        # 初始状态向量
    Pk = np.eye(4) * 10  #                                             # 初始状态协方差矩阵
    kf = lcy_KF(F, Q, R, H, Xk, Pk, Zk=None)                           # 创建卡尔曼滤波器对象
    

    #模拟输入数据
    read_datas = pd.read_csv('./mouse_sim_kf_data.csv',header=None).values
    read_datas = read_datas[1:,:].astype(np.float32)
    input_datas = read_datas[:, -4:]
    output_datas = []
    for z in input_datas:
        Xk = kf.filter(z)
        output_datas.append(Xk)


    # 画图
    true_datas = read_datas[:, 1:5]
    input_datas = read_datas[:, -4:]
    output_datas = np.array(output_datas).reshape(-1, 4)
    t = read_datas[:, 0]
    # 绘制结果
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    #子图1:x/y位置对比
    plt.subplot(2, 2, 1)
    plt.plot(t, true_datas[:, 0], label='真实x位置', color='g')
    plt.plot(t, input_datas[:, 0], label='观测x位置', color='r', linestyle='dotted')
    plt.plot(t, output_datas[:, 0], label='滤波后x位置', color='b', linestyle='dashed')
    plt.legend()
    plt.xlabel('时间s')
    plt.ylabel('位置m')
    plt.title('x位置对比')
    plt.subplot(2, 2, 2)
    plt.plot(t, true_datas[:, 1], label='真实y位置', color='g')
    plt.plot(t, input_datas[:, 1], label='观测y位置', color='r', linestyle='dotted')
    plt.plot(t, output_datas[:, 1], label='滤波后y位置', color='b', linestyle='dashed')
    plt.legend()
    plt.xlabel('时间s')
    plt.ylabel('位置m')
    plt.title('y位置对比')
    #子图2:x/y速度对比
    plt.subplot(2, 2, 3)
    plt.plot(t, true_datas[:, 2], label='真实x速度', color='g')
    plt.plot(t, input_datas[:, 2], label='观测x速度', color='r', linestyle='dotted')
    plt.plot(t, output_datas[:, 2], label='滤波后x速度', color='b', linestyle='dashed')
    plt.legend()
    plt.xlabel('时间s')
    plt.ylabel('速度m/s')
    plt.title('x速度对比')
    plt.subplot(2, 2, 4)
    plt.plot(t, true_datas[:, 3], label='真实y速度', color='g')
    plt.plot(t, input_datas[:, 3], label='观测y速度', color='r', linestyle='dotted')
    plt.plot(t, output_datas[:, 3], label='滤波后y速度', color='b', linestyle='dashed')
    plt.legend()
    plt.xlabel('时间s')
    plt.ylabel('速度m/s')
    plt.title('y速度对比')
    plt.tight_layout()
    plt.show()



