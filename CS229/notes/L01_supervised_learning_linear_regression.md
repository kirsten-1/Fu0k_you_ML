# CS229 Lecture 1: 监督学习与线性回归

> **面试考点**：损失函数设计、梯度下降的收敛性、Learning Rate 的影响、正规方程与梯度下降的选择、特征缩放。

---

## 一、机器学习是什么

**Arthur Samuel (1959)**：不需要明确编程，让计算机具有学习能力。

**Tom Mitchell (1998)**：一个任务 T，用性能度量 P衡量，如果计算机程序在任务 T 上通过经验 E 提升性能 P，则称该程序从经验 E 中学习。

**机器学习分类**：
- **监督学习（Supervised Learning）**：给定输入-输出对 $(x^{(i)}, y^{(i)})$，学习 $f: x \to y$
  - 回归：$y$ 是连续值（房价、气温）
  - 分类：$y$ 是离散标签（猫/狗、垃圾邮件）
- **无监督学习（Unsupervised Learning）**：只有 $x^{(i)}$，找数据内部结构
  - 聚类（K-Means）、降维（PCA）、密度估计
- **强化学习（Reinforcement Learning）**：Agent 在环境中做决策，用 reward 信号反馈

---

## 二、监督学习形式化

给定训练集 $\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \ldots, (x^{(m)}, y^{(m)})\}$，其中 $m$ 是样本数。

学习的目标是找到函数 $h: \mathcal{X} \to \mathcal{Y}$（**hypothesis 假设函数**），使得 $h(x)$ 能尽可能准确地预测 $y$。

- 回归：$\mathcal{Y} = \mathbb{R}$
- 分类：$\mathcal{Y} = \{1, 2, \ldots, k\}$

---

## 三、线性回归（Linear Regression）

### 3.1 模型假设

假设输出是输入的线性函数：

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n = \theta^T x$$

其中 $\theta$ 是参数向量，$x$ 是特征向量（通常 $x_0 = 1$，所以写成 $\theta_0 + \cdots$）。

### 3.2 损失函数（Cost Function）

**均方误差（MSE）**：

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

- 系数 $\frac{1}{2}$ 是为了在求导时抵消平方的 2
- 系数 $\frac{1}{m}$ 是平均（也可写 $\frac{1}{2m}$ 配合 $\frac{1}{m}$ 平均）

> **面试考点**：为什么用 MSE 而不是 MAE？
> - MSE 处处可导，MAE 在零点不可导（不利于梯度下降）
> - MSE 对大误差的惩罚更重（误差平方放大量级），符合正态分布假设
> - 但 MSE 对离群点（outliers）敏感，MAE 更鲁棒

---

## 四、梯度下降（Gradient Descent）

### 4.1 核心思想

从一个初始 $\theta$ 出发，沿着损失函数下降最快的方向（负梯度方向）迭代更新参数：

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

其中 $\alpha > 0$ 是**学习率（Learning Rate）**。

### 4.2 批量梯度下降（Batch Gradient Descent）

对每个参数 $j$：

$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

**同时更新**所有 $\theta_j$（同步更新）。

**推导**：
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\theta^T x^{(i)} - y^{(i)})^2$$

对 $\theta_j$ 求偏导：
$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (\theta^T x^{(i)} - y^{(i)}) x_j^{(i)}$$

---

### 4.3 学习率 $\alpha$ 的影响

| $\alpha$ 太小 | $\alpha$ 太大 |
|--------------|--------------|
| 收敛极慢 | 可能越过最优点，发散 |
| | 甚至损失函数值上升 |

**如何判断收敛**：绘制 $J(\theta)$ 随迭代次数的曲线，下降趋于平缓时停止。

**自适应学习率**：实际中常用 learning rate schedule（前期大，逐渐衰减）或 Adagrad / Adam。

> **面试考点**：梯度下降一定会收敛到全局最优吗？
> - 对线性回归的凸函数（MSE 是凸的），梯度下降收敛到全局最优
> - 对非凸函数（如神经网络），只能保证收敛到局部最优
> - 随机梯度下降（SGD）有跳出局部最优的可能（噪声帮助探索）

---

## 五、正规方程（Normal Equations）

对线性回归，损失函数是凸的，可以直接求解析解：

$$\theta = (X^T X)^{-1} X^T y$$

其中 $X$ 是 $m \times (n+1)$ 的设计矩阵（每行一个样本，加一列1）。

**推导**（矩阵形式）：
$$J(\theta) = \frac{1}{2m} \| X\theta - y \|^2$$

令 $\nabla_\theta J(\theta) = X^T (X\theta - y) = 0$，得：
$$X^T X \theta = X^T y$$

---

### 5.1 正规方程 vs 梯度下降

| | 正规方程 | 梯度下降 |
|---|---|---|
| 复杂度 | $O(n^3)$（矩阵求逆） | $O(m \cdot n \cdot T)$ |
| 需要学习率 | ❌ | ✅ |
| 需要迭代 | ❌ | ✅ |
| 特征量大时 | 慢（$n > 10^4$） | 相对较快 |
| 非凸问题 | ❌ | ✅ 可用 |
| 数值稳定性 | 求逆可能病态 | 稳定 |

> **面试考点**：什么时候用正规方程，什么时候用梯度下降？
> - 特征数 $n \leq 10^4$ 且 $X^T X$ 可逆 → 正规方程更快
> - $n$ 很大（$10^6$+），或非凸问题 → 梯度下降
> - $X^T X$ 接近奇异（特征值很小）→ 数值不稳定，加正则化 $\lambda$

---

## 六、特征缩放（Feature Scaling）

当特征的量纲差异很大时（如房间面积 0-2000 vs 房间数量 1-5），损失函数的等高线是椭圆的，梯度下降路径曲折。

**方法**：

1. **均值归一化（Mean Normalization）**：
$$x_i := \frac{x_i - \mu_i}{\max_i - \min_i}$$

2. **Z-Score 标准化**：
$$x_i := \frac{x_i - \mu_i}{\sigma_i}$$

3. **Min-Max Scaling**：
$$x_i := \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}$$

特征缩放后，$x_i$ 的范围大约在 $[-1, 1]$，梯度下降会更快收敛。

---

## 七、多元线性回归（Multiple Linear Regression）

$$h_\theta(x) = \theta^T x = \sum_{j=0}^{n} \theta_j x_j$$

梯度下降更新（对每个 $j$）：

$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

---

## 八、概率视角（Probabilistic Interpretation）

假设 $y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$，其中误差 $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$（独立同分布的高斯噪声）。

则：
$$p(y^{(i)} | x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)$$

最大似然估计（MLE）：
$$\mathcal{L}(\theta) = \prod_{i=1}^{m} p(y^{(i)} | x^{(i)}; \theta)$$

取对数后化简，发现 $\log \mathcal{L}(\theta) = -J(\theta) + \text{const}$，即**最小化 MSE 等价于 MLE**。

> **面试考点**：为什么假设高斯噪声？
> - 中心极限定理：大量小独立因素叠加 → 正态分布
> - 数学上处理方便，对数似然是凸的
> - 也可假设拉普拉斯分布（得到 MAE 对应的损失）

---

## 九、本讲重点回顾

| 概念 | 掌握要点 |
|------|---------|
| 监督/无监督/强化学习 | 定义和典型算法 |
| MSE 损失函数 | 公式、可导性、对离群点敏感 |
| 梯度下降 | 更新公式、学习率影响、收敛判断 |
| 正规方程 | 解析解、$O(n^3)$ 复杂度、适用场景 |
| 特征缩放 | 为什么需要、常用方法 |
| 概率解释 | 高斯噪声假设 → MLE → MSE 等价 |

---

## 十、延伸面试题

1. 如果 $X^T X$ 不可逆怎么办？（病态矩阵、冗余特征、加正则化）
2. 梯度下降和随机梯度下降（SGD）的区别？Mini-Batch 是什么？
3. 如果学习率太大导致损失上升，有什么策略修复？
4. MSE 和 MAE 在实际中怎么选？

---

## 十一、代码实现

见 `code/l01_linear_regression.py`：NumPy 实现线性回归 + 梯度下降 + 正规方程对比。
