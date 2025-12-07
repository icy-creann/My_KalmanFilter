import numpy as np

"""
扩展卡尔曼滤波(EKF)类
核心区别:用非线性函数+雅克比矩阵替代KF的线性矩阵F/H
使用方法:
1. 定义非线性状态转移函数f和雅克比函数jac_f
2. 定义非线性观测函数h和雅克比函数jac_h
3. 初始化EKF对象
4. 调用filter方法进行滤波
"""


class lcy_EKF:
    """
    Xk: 状态向量 (nx1)
    Zk: 观测向量 (mx1)
    f: 非线性状态转移函数 → X_k+1 = f(X_k)
    jac_f: 状态转移函数的雅克比矩阵计算函数 → F_jac = jac_f(X_k)
    h: 非线性观测函数 → Z_k = h(X_k)
    jac_h: 观测函数的雅克比矩阵计算函数 → H_jac = jac_h(X_k)
    Pk: 协方差矩阵 (nxn)
    Q: 过程噪声协方差矩阵 (nxn)
    R: 观测噪声协方差矩阵 (mxm)
    Kk: 卡尔曼增益 (nxm)
    """

    def __init__(self, f, jac_f, h, jac_h, Q, R, Xk, Pk, Zk=None):
        # 非线性函数
        self.f = f          # 状态转移非线性函数
        self.jac_f = jac_f  # 状态转移雅克比函数
        self.h = h          # 观测非线性函数
        self.jac_h = jac_h  # 观测雅克比函数
        
        # 噪声/协方差
        self.Q = Q  # 过程噪声协方差矩阵
        self.R = R  # 观测噪声协方差矩阵
        self.Pk = Pk  # 协方差矩阵
        
        # 状态/观测向量
        self.Xk = np.array(Xk).reshape(-1, 1)  # 状态向量 (nx1)
        self.Zk = Zk  # 观测向量
        
        # 卡尔曼增益
        self.n = self.Xk.shape[0]  # 状态维度
        self.m = self._get_obs_dim()  # 观测维度(从h函数推导)
        self.Kk = np.zeros((self.n, self.m))  # Kk维度:nxm

    def _get_obs_dim(self):
        """私有方法:推导观测维度(避免手动指定)"""
        # 用初始状态计算观测值,推导观测维度
        z_test = self.h(self.Xk)
        return np.array(z_test).reshape(-1, 1).shape[0]

    # 先验估计(非线性版本)
    def predict(self):
        """非线性预测:替代KF的线性F@Xk"""
        # 1. 非线性状态预测(先验状态)
        self.Xk_ = self.f(self.Xk)  # 替代 KF的 self.F @ self.Xk
        # 2. 计算状态转移雅克比矩阵(替代KF的F)
        F_jac = self.jac_f(self.Xk)
        # 3. 先验协方差预测(和KF逻辑一致,只是F换成F_jac)
        self.Pk_ = F_jac @ self.Pk @ F_jac.T + self.Q

    # 计算卡尔曼增益(雅克比版本)
    def cal_Kk(self):
        """计算卡尔曼增益:H换成观测雅克比矩阵"""
        # 1. 计算观测雅克比矩阵(替代KF的H)
        H_jac = self.jac_h(self.Xk_)
        # 2. 残差协方差(H换成H_jac)
        Sk = H_jac @ self.Pk_ @ H_jac.T + self.R
        # 3. 卡尔曼增益(H换成H_jac,伪逆保证鲁棒性)
        self.Kk = self.Pk_ @ H_jac.T @ np.linalg.pinv(Sk)

    # 后验估计(非线性版本)
    def update(self, z):
        """非线性更新:替代KF的线性H@Xk_"""
        self.Zk = np.array(z).reshape(-1, 1)
        obs_residual = self.Zk - self.h(self.Xk_)
        self.Xk = self.Xk_ + self.Kk @ obs_residual
        H_jac = self.jac_h(self.Xk_)
        self.Pk = (np.eye(self.n) - self.Kk @ H_jac) @ self.Pk_

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








##############使用示例###############



# 定义非线性状态转移函数f
def f(X):
    dt = 0.01
    damp = 0.01
    x, y, vx, vy, ax, ay = X[0,0], X[1,0], X[2,0], X[3,0], X[4,0], X[5,0]
    x_new = x + vx * dt + 0.5 * ax * dt**2
    y_new = y + vy * dt + 0.5 * ay * dt**2
    vx_new = vx * (1 - damp) + ax * dt
    vy_new = vy * (1 - damp) + ay * dt
    ax_new = ax
    ay_new = ay
    return np.array([[x_new], [y_new], [vx_new], [vy_new], [ax_new], [ay_new]])
# 定义状态转移函数的雅克比矩阵计算函数
def jac_f(X):
    dt = 0.01
    damp = 0.01
    return np.array([
        [1, 0, dt, 0, 0.5*dt**2, 0],
        [0, 1, 0, dt, 0, 0.5*dt**2],
        [0, 0, 1 - damp, 0, dt, 0],
        [0, 0, 0, 1 - damp, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
# 定义非线性观测函数h
def h(X):
    x, y, vx, vy = X[0,0], X[1,0], X[2,0], X[3,0]
    return np.array([[x], [y], [vx], [vy]])
# 定义观测函数的雅克比矩阵计算函数
def jac_h(X):
    return np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    # 定义参数
    dt = 0.01  # 时间间隔
    # 定义噪声协方差矩阵Q\R
    Q = np.diag([0.01, 0.01, 0.1, 0.1, 1.0, 1.0]) # 过程噪声协方差矩阵
    R = np.diag([1, 1, 4, 4])                     # 观测噪声协方差矩阵
    # 定义初始状态
    X0 = np.array([0, 0, 0, 0, 0, 0])             # 初始状态向量
    P0 = np.eye(6) * 10                          # 初始状态协方差矩阵

    ekf = lcy_EKF(f=f, jac_f=jac_f, h=h, jac_h=jac_h, Q=Q, R=R, Xk=X0, Pk=P0)

    read_datas = pd.read_csv('./mouse_sim_ekf_data.csv').values
    t = read_datas[:, 0]  # 时间
    true_datas = read_datas[:, 1:5]  # 真实值[x,y,vx,vy]
    input_datas = read_datas[:, 5:9]  # 观测值[x_meas,y_meas,vx_meas,vy_meas]

    output_datas = []
    for z in input_datas:
        Xk = ekf.filter(z)
        output_datas.append(Xk)

    # 格式转换
    output_datas = np.array(output_datas).reshape(-1, 6)  # 6维状态输出
    # 只取前4维(x,y,vx,vy)用于绘图
    output_datas_4d = output_datas[:, :4]


    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 8))

    # x位置对比
    plt.subplot(3, 2, 1)
    plt.plot(t, true_datas[:, 0], 'g-', label='真实x')
    plt.plot(t, input_datas[:, 0], 'r:', label='观测x')
    plt.plot(t, output_datas_4d[:, 0], 'b--', label='EKF滤波x')
    plt.legend()
    plt.xlabel('时间(秒)')
    plt.ylabel('x位置(像素)')
    plt.title('x位置对比(EKF)')

    # y位置对比
    plt.subplot(3, 2, 2)
    plt.plot(t, true_datas[:, 1], 'g-', label='真实y')
    plt.plot(t, input_datas[:, 1], 'r:', label='观测y')
    plt.plot(t, output_datas_4d[:, 1], 'b--', label='EKF滤波y')
    plt.legend()
    plt.xlabel('时间(秒)')
    plt.ylabel('y位置(像素)')
    plt.title('y位置对比(EKF)')

    # vx速度对比
    plt.subplot(3, 2, 3)
    plt.plot(t, true_datas[:, 2], 'g-', label='真实vx')
    plt.plot(t, input_datas[:, 2], 'r:', label='观测vx')
    plt.plot(t, output_datas_4d[:, 2], 'b--', label='EKF滤波vx')
    plt.legend()
    plt.xlabel('时间(秒)')
    plt.ylabel('vx速度(像素/秒)')
    plt.title('x速度对比(EKF)')

    # vy速度对比
    plt.subplot(3, 2, 4)
    plt.plot(t, true_datas[:, 3], 'g-', label='真实vy')
    plt.plot(t, input_datas[:, 3], 'r:', label='观测vy')
    plt.plot(t, output_datas_4d[:, 3], 'b--', label='EKF滤波vy')
    plt.legend()
    plt.xlabel('时间(秒)')
    plt.ylabel('vy速度(像素/秒)')
    plt.title('y速度对比(EKF)')



    # 图2D位置对比
    plt.subplot(3, 2, 5)
    plt.plot(true_datas[:, 0], true_datas[:, 1], 'g-', label='真实轨迹')
    plt.plot(input_datas[:, 0], input_datas[:, 1], 'r:', label='观测轨迹')
    plt.plot(output_datas_4d[:, 0], output_datas_4d[:, 1], 'b--', label='EKF滤波轨迹')
    plt.legend()
    plt.xlabel('x位置(像素)')
    plt.ylabel('y位置(像素)')
    plt.title('2D位置对比(EKF)')

    plt.tight_layout()
    plt.show()