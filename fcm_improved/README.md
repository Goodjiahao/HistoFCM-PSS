# 基于直方图初始化与部分离策略的FCM不完整数据聚类改进方法

## 项目简介
本项目实现了一种改进的模糊C均值(FCM)聚类算法，针对不完整数据（含缺失值）的聚类问题，结合直方图初始化和部分距离策略，提高聚类准确性和稳定性。

## 算法原理
### 传统FCM算法
模糊C均值聚类算法通过最小化目标函数实现数据的软划分，允许每个数据点以不同隶属度属于多个聚类中心。目标函数定义为：

$$ J = \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2 $$

其中，$u_{ij}$为样本$x_i$对聚类中心$v_j$的隶属度，$m$为模糊系数，$c$为聚类数。

### 改进策略
1. **直方图初始化**：利用数据直方图的峰值点作为初始聚类中心，避免随机初始化导致的局部最优问题。
2. **部分距离策略**：对于含缺失值的数据，仅使用非缺失特征计算样本与聚类中心的距离，无需预先填充缺失值。

## 项目结构
```
fcm_improved/
├── fcm_base.py          # 基础FCM算法实现
├── fcm_histogram_init.py # 带直方图初始化的FCM
├── fcm_partial_distance.py # 处理不完整数据的FCM（主算法）
├── evaluation.py        # 算法评估与对比脚本
├── requirements.txt     # 依赖包列表
├── README.md            # 项目说明文档
├── 论文模板.md          # 论文模板
└── histograms/          # 生成的特征直方图图片
    ├── histogram_feature_0.png
    └── histogram_feature_1.png
```

## 环境要求
- Python 3.8+
- 依赖包：numpy, scipy, scikit-learn, matplotlib

## 安装与使用
1. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

2. 运行评估脚本（对比改进FCM、标准FCM和K-means）：
   ```
   python evaluation.py
   ```

## 实验结果
评估指标包括调整兰德指数(ARI)、归一化互信息(NMI)和运行时间。在含10%缺失值的合成数据集上，三种算法的性能对比如下：

| 算法 | 调整兰德指数(ARI) | 归一化互信息(NMI) | 运行时间(秒) |
|------|-------------------|-------------------|--------------|
| 改进FCM | 0.3093 | 0.4647 | 0.7219 |
| 标准FCM | 0.0000 | 0.0000 | 0.1295 |
| K-means | 0.8280 | 0.8388 | 1.3679 |

改进FCM在处理含缺失值数据时表现优于标准FCM，但K-means在当前合成数据集上表现最佳。建议在真实含缺失值数据集上进一步验证改进算法的优势。

## 使用示例与参数说明

### 基本使用
运行评估脚本将自动生成合成数据集（含缺失值），并对比三种聚类算法的性能：
```bash
python evaluation.py
```

### 可调参数
在`evaluation.py`中可以调整以下关键参数：
- `n_samples`: 样本数量（默认：1000）
- `n_features`: 特征数量（默认：2）
- `n_clusters`: 聚类数（默认：3）
- `missing_rate`: 缺失值比例（默认：0.1）
- `max_iter`: 最大迭代次数（默认：100）
- `m`: 模糊系数（默认：2.0）

## 结果解释
运行评估脚本后，将生成：
1. `clustering_comparison.png`: 三种算法的聚类结果对比图
2. `histograms/`目录: 每个特征的直方图分布图片
3. 控制台输出: 详细的性能指标（ARI、NMI、运行时间）

## 参考文献
1. Dunn, J. C. (1974). A fuzzy relative of the ISODATA process and its use in detecting compact well-separated clusters.
2. Bezdek, J. C. (1984). FCM: The fuzzy c-means clustering algorithm.
3. 基于直方图的模糊C均值聚类改进算法研究
4. 不完备数据FCM聚类和离群点检测方法研究