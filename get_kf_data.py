import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- 1. 配置参数 ----------------------
n_steps = 100  # 生成100组数据
dt = 0.01      # 采样时间（100Hz）
# 初始真实状态：x=0, y=0, vx=50像素/秒, vy=30像素/秒
x_true_init = 0.0
y_true_init = 0.0
vx_true_init = 50.0
vy_true_init = 30.0

# 噪声参数（模拟测量误差）
noise_x_y = 1.0    # x/y的观测噪声标准差（±1像素）
noise_vx_vy = 2.0  # vx/vy的观测噪声标准差（±2像素/秒）
process_noise = 0.1 # 真实值的微小扰动（模拟鼠标轻微变速）

# ---------------------- 2. 生成真实值和观测值 ----------------------
# 初始化数组存储数据
true_data = np.zeros((n_steps, 4))  # 真实值：x, y, vx, vy
meas_data = np.zeros((n_steps, 4))  # 观测值：x, y, vx, vy

# 初始状态
x_true = x_true_init
y_true = y_true_init
vx_true = vx_true_init
vy_true = vy_true_init

for i in range(n_steps):
    # 生成真实值（匀速运动+微小过程噪声）
    x_true = x_true + vx_true * dt + np.random.normal(0, process_noise)
    y_true = y_true + vy_true * dt + np.random.normal(0, process_noise)
    vx_true = vx_true + np.random.normal(0, process_noise)  # 速度微小波动
    vy_true = vy_true + np.random.normal(0, process_noise)
    
    # 生成观测值（真实值+高斯噪声）
    x_meas = x_true + np.random.normal(0, noise_x_y)
    y_meas = y_true + np.random.normal(0, noise_x_y)
    vx_meas = vx_true + np.random.normal(0, noise_vx_vy)
    vy_meas = vy_true + np.random.normal(0, noise_vx_vy)
    
    # 保存数据
    true_data[i] = [x_true, y_true, vx_true, vy_true]
    meas_data[i] = [x_meas, y_meas, vx_meas, vy_meas]

# ---------------------- 3. 整理成DataFrame（方便查看/使用） ----------------------
df_true = pd.DataFrame(true_data, columns=['x_true', 'y_true', 'vx_true', 'vy_true'])
df_meas = pd.DataFrame(meas_data, columns=['x_meas', 'y_meas', 'vx_meas', 'vy_meas'])
# 合并数据（加时间列）
df = pd.concat([
    pd.DataFrame({'time': np.arange(n_steps)*dt}),  # 时间轴（秒）
    df_true,
    df_meas
], axis=1)

# ---------------------- 4. 输出数据（前10行示例） ----------------------
print("生成的100组鼠标数据（前10行）：")
print(df.head(10))

# 保存为CSV（可选，方便后续使用）
df.to_csv('mouse_sim_data.csv', index=False)
print("\n数据已保存为 mouse_sim_data.csv")

# ---------------------- 5. 可视化对比（真实值 vs 观测值） ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.figure(figsize=(12, 8))

# 子图1：x/y坐标
plt.subplot(2, 2, 1)
plt.plot(df['time'], df['x_true'], 'b-', label='x真实值')
plt.plot(df['time'], df['x_meas'], 'r.', alpha=0.6, label='x观测值')
plt.xlabel('时间（秒）')
plt.ylabel('x坐标（像素）')
plt.legend()
plt.title('x坐标：真实值 vs 观测值')

plt.subplot(2, 2, 2)
plt.plot(df['time'], df['y_true'], 'b-', label='y真实值')
plt.plot(df['time'], df['y_meas'], 'r.', alpha=0.6, label='y观测值')
plt.xlabel('时间（秒）')
plt.ylabel('y坐标（像素）')
plt.legend()
plt.title('y坐标：真实值 vs 观测值')

# 子图2：vx/vy速度
plt.subplot(2, 2, 3)
plt.plot(df['time'], df['vx_true'], 'b-', label='vx真实值')
plt.plot(df['time'], df['vx_meas'], 'r.', alpha=0.6, label='vx观测值')
plt.xlabel('时间（秒）')
plt.ylabel('vx速度（像素/秒）')
plt.legend()
plt.title('vx速度：真实值 vs 观测值')

plt.subplot(2, 2, 4)
plt.plot(df['time'], df['vy_true'], 'b-', label='vy真实值')
plt.plot(df['time'], df['vy_meas'], 'r.', alpha=0.6, label='vy观测值')
plt.xlabel('时间（秒）')
plt.ylabel('vy速度（像素/秒）')
plt.legend()
plt.title('vy速度：真实值 vs 观测值')

plt.tight_layout()
plt.show()