import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- 1. 配置参数 ----------------------
n_steps = 100  # 生成100组数据
dt = 0.01      # 采样时间（100Hz）
t_total = n_steps * dt  # 总时间（1秒）

# 变加速参数
A_x = 200     # x方向加速度幅值（像素/秒²）
A_y = 150     # y方向加速度幅值（像素/秒²）
T = 5         # 加速度变化周期（秒）
process_noise = 5   # 加速度随机扰动（模拟手动操作的不规则性）
noise_x_y = 1.0     # x/y观测噪声标准差（像素）
noise_vx_vy = 2.0   # vx/vy观测噪声标准差（像素/秒）

# ---------------------- 2. 初始化状态 ----------------------
# 初始状态：x=0, y=0, vx=0, vy=0, ax=0, ay=0
x_true = 0.0
y_true = 0.0
vx_true = 0.0
vy_true = 0.0

# 存储数据的数组
true_data = np.zeros((n_steps, 4))  # 真实值：x, y, vx, vy
meas_data = np.zeros((n_steps, 4))  # 观测值：x, y, vx, vy
time_list = np.arange(n_steps) * dt # 时间轴（秒）

# ---------------------- 3. 生成变加速运动数据 ----------------------
for i in range(n_steps):
    t = time_list[i]  # 当前时间
    
    # 1. 计算变加速度（正弦变化+随机扰动）
    ax = A_x * np.sin(2 * np.pi * t / T) + np.random.normal(0, process_noise)
    ay = A_y * np.cos(2 * np.pi * t / T) + np.random.normal(0, process_noise)
    
    # 2. 更新速度（变加速：v = v_prev + a*dt）
    vx_true = vx_true + ax * dt
    vy_true = vy_true + ay * dt
    
    # 3. 更新位置（变加速：x = x_prev + v_prev*dt + 0.5*a*dt²）
    x_true = x_true + vx_true * dt + 0.5 * ax * dt**2
    y_true = y_true + vy_true * dt + 0.5 * ay * dt**2
    
    # 4. 保存真实值
    true_data[i] = [x_true, y_true, vx_true, vy_true]
    
    # 5. 生成观测值（真实值+高斯噪声）
    x_meas = x_true + np.random.normal(0, noise_x_y)
    y_meas = y_true + np.random.normal(0, noise_x_y)
    vx_meas = vx_true + np.random.normal(0, noise_vx_vy)
    vy_meas = vy_true + np.random.normal(0, noise_vx_vy)
    meas_data[i] = [x_meas, y_meas, vx_meas, vy_meas]

# ---------------------- 4. 整理成DataFrame ----------------------
df = pd.DataFrame({
    'time': time_list,
    'x_true': true_data[:, 0],
    'y_true': true_data[:, 1],
    'vx_true': true_data[:, 2],
    'vy_true': true_data[:, 3],
    'x_meas': meas_data[:, 0],
    'y_meas': meas_data[:, 1],
    'vx_meas': meas_data[:, 2],
    'vy_meas': meas_data[:, 3]
})

# ---------------------- 5. 保存数据 ----------------------
df.to_csv('mouse_sim_ekf_data.csv', index=False)
print("100组变加速鼠标数据已保存为 mouse_sim_ekf_data.csv")
print("\n数据前10行预览：")
print(df.head(10))

# ---------------------- 6. 可视化变加速效果 ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 负号显示
plt.figure(figsize=(14, 10))

# 子图1：x/y位置（变加速轨迹）
plt.subplot(3, 2, 1)
plt.plot(df['x_true'], df['y_true'], 'b-', label='真实轨迹')
plt.scatter(df['x_meas'], df['y_meas'], c='r', s=5, alpha=0.6, label='观测轨迹')
plt.xlabel('x位置（像素）')
plt.ylabel('y位置（像素）')
plt.legend()
plt.title('鼠标变加速运动轨迹：真实值 vs 观测值')

# 子图2：x位置随时间变化
plt.subplot(3, 2, 2)
plt.plot(df['time'], df['x_true'], 'b-', label='x真实值')
plt.plot(df['time'], df['x_meas'], 'r.', alpha=0.6, label='x观测值')
plt.xlabel('时间（秒）')
plt.ylabel('x位置（像素）')
plt.legend()
plt.title('x位置：变加速运动')

# 子图3：y位置随时间变化
plt.subplot(3, 2, 3)
plt.plot(df['time'], df['y_true'], 'b-', label='y真实值')
plt.plot(df['time'], df['y_meas'], 'r.', alpha=0.6, label='y观测值')
plt.xlabel('时间（秒）')
plt.ylabel('y位置（像素）')
plt.legend()
plt.title('y位置：变加速运动')

# 子图4：vx速度随时间变化
plt.subplot(3, 2, 4)
plt.plot(df['time'], df['vx_true'], 'b-', label='vx真实值')
plt.plot(df['time'], df['vx_meas'], 'r.', alpha=0.6, label='vx观测值')
plt.xlabel('时间（秒）')
plt.ylabel('vx速度（像素/秒）')
plt.legend()
plt.title('x方向速度（变加速）')

# 子图5：vy速度随时间变化
plt.subplot(3, 2, 5)
plt.plot(df['time'], df['vy_true'], 'b-', label='vy真实值')
plt.plot(df['time'], df['vy_meas'], 'r.', alpha=0.6, label='vy观测值')
plt.xlabel('时间（秒）')
plt.ylabel('vy速度（像素/秒）')
plt.legend()
plt.title('y方向速度（变加速）')

plt.tight_layout()
plt.show()