"""
CS229 Lecture 1: Linear Regression
NumPy 实现：梯度下降 + 正规方程 + 可视化收敛曲线
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 生成模拟数据
# ============================================================
np.random.seed(42)
m = 100  # 样本数
X_raw = np.random.randn(m, 1) * 100  # 房间面积 (0 ~ 200)
y = 5 * X_raw.squeeze() + 50 + np.random.randn(m) * 20  # y = 5x + 50 + noise

# 加入偏置项 (x0 = 1)
X = np.c_[np.ones(m), X_raw]  # shape: (m, 2)

print(f"数据形状: X={X.shape}, y={y.shape}")
print(f"真实参数: θ_true = [50, 5]")


# ============================================================
# 2. MSE 损失函数
# ============================================================
def compute_mse(X, y, theta):
    """计算均方误差 J(θ) = (1/2m) * sum((h(x) - y)^2)"""
    m = len(y)
    h = X @ theta  # @ 是矩阵乘法
    return (1 / (2 * m)) * np.sum((h - y) ** 2)


# ============================================================
# 3. 梯度下降 (Batch GD)
# ============================================================
def gradient_descent(X, y, theta_init, alpha=0.001, n_iters=1000):
    """
    批量梯度下降
    θ_j := θ_j - α * (1/m) * sum((h(x^(i)) - y^(i)) * x_j^(i))
    """
    m = len(y)
    theta = theta_init.copy()
    history = []  # 记录每步的损失

    for i in range(n_iters):
        h = X @ theta
        error = h - y
        gradient = (1 / m) * (X.T @ error)  # 向量化梯度 (2,)
        theta = theta - alpha * gradient
        history.append(compute_mse(X, y, theta))

    return theta, history


# ============================================================
# 4. 正规方程 (Normal Equation)
# ============================================================
def normal_equation(X, y):
    """
    θ = (X^T X)^(-1) X^T y
    注意：需 X^T X 可逆，否则加正则化或用 pinv
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y


# ============================================================
# 5. 运行对比
# ============================================================
# 5.1 梯度下降
theta_init = np.zeros(X.shape[1])  # [0, 0]
theta_gd, history_gd = gradient_descent(X, y, theta_init, alpha=0.00005, n_iters=10000)

print("\n=== 梯度下降结果 ===")
print(f"最终 θ = {theta_gd}")
print(f"最终 MSE = {history_gd[-1]:.4f}")

# 5.2 正规方程
theta_ne = normal_equation(X, y)

print("\n=== 正规方程结果 ===")
print(f"解析 θ = {theta_ne}")
print(f"MSE = {compute_mse(X, y, theta_ne):.4f}")


# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 图1: 数据 + 拟合线
ax1 = axes[0]
ax1.scatter(X_raw, y, alpha=0.6, label='训练数据')
x_line = np.linspace(X_raw.min(), X_raw.max(), 100)
ax1.plot(x_line, theta_gd[0] + theta_gd[1] * x_line, 'r-', linewidth=2, label=f'GD: y={theta_gd[1]:.2f}x+{theta_gd[0]:.2f}')
ax1.plot(x_line, theta_ne[0] + theta_ne[1] * x_line, 'g--', linewidth=2, label=f'NE: y={theta_ne[1]:.2f}x+{theta_ne[0]:.2f}')
ax1.set_xlabel('房间面积 (sqft)')
ax1.set_ylabel('房价 ($1000)')
ax1.set_title('线性回归拟合')
ax1.legend()

# 图2: 收敛曲线
ax2 = axes[1]
ax2.plot(history_gd, 'b-', linewidth=1)
ax2.set_xlabel('迭代次数')
ax2.set_ylabel('MSE 损失')
ax2.set_title('梯度下降收敛曲线')
ax2.set_yscale('log')

# 图3: 不同学习率的收敛对比
ax3 = axes[2]
for alpha in [0.00001, 0.00005, 0.0001, 0.0005]:
    _, history = gradient_descent(X, y, theta_init, alpha=alpha, n_iters=2000)
    ax3.plot(history, label=f'α={alpha}', linewidth=1.5)
ax3.set_xlabel('迭代次数')
ax3.set_ylabel('MSE 损失')
ax3.set_title('学习率对比收敛速度')
ax3.set_yscale('log')
ax3.legend()

plt.tight_layout()
plt.savefig('../code/l01_convergence.png', dpi=150)
plt.show()
print("\n图片已保存到 code/l01_convergence.png")


# ============================================================
# 7. 验证：特征缩放的影响
# ============================================================
def gradient_descent_unscaled(X_raw, y, alpha=0.001, n_iters=1000):
    """未缩放特征的梯度下降"""
    X = np.c_[np.ones(len(y)), X_raw]
    theta = np.zeros(X.shape[1])
    history = []
    for _ in range(n_iters):
        h = X @ theta
        theta = theta - alpha * (1/len(y)) * (X.T @ (h - y))
        history.append(compute_mse(X, y, theta))
    return history

def gradient_descent_scaled(X_raw, y, alpha=0.1, n_iters=1000):
    """标准化特征的梯度下降"""
    mu = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    X_scaled = (X_raw - mu) / std
    X = np.c_[np.ones(len(y)), X_scaled]
    theta = np.zeros(X.shape[1])
    history = []
    for _ in range(n_iters):
        h = X @ theta
        theta = theta - alpha * (1/len(y)) * (X.T @ (h - y))
        history.append(compute_mse(X, y, theta))
    return history

history_unscaled = gradient_descent_unscaled(X_raw, y, alpha=0.001, n_iters=1000)
history_scaled = gradient_descent_scaled(X_raw, y, alpha=0.1, n_iters=1000)

plt.figure(figsize=(8, 4))
plt.plot(history_unscaled, label='未缩放 (α=0.001)', linewidth=2)
plt.plot(history_scaled, label='标准化 (α=0.1)', linewidth=2)
plt.xlabel('迭代次数')
plt.ylabel('MSE')
plt.title('特征缩放对收敛速度的影响')
plt.yscale('log')
plt.legend()
plt.savefig('../code/l01_feature_scaling.png', dpi=150)
plt.show()
print("特征缩放对比图已保存")
