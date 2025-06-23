import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class FCM:
    def __init__(self, n_clusters=3, m=2, max_iter=100, tol=1e-5):
        self.n_clusters = n_clusters
        self.m = m  # 模糊系数
        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.u = None  # 隶属度矩阵
        
    def initialize_centers(self, X):
        # 随机初始化聚类中心
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices]
        
    def compute_membership(self, X):
        # 计算隶属度矩阵
        n_samples = X.shape[0]
        u = np.zeros((n_samples, self.n_clusters))
        
        for i in range(n_samples):
            for j in range(self.n_clusters):
                # 添加防止除零的极小值
                epsilon = 1e-10
                denominator = np.sum([(np.linalg.norm(X[i] - self.centers[jk]) + epsilon) ** (2/(self.m-1)) for jk in range(self.n_clusters)])
                u[i, j] = 1 / denominator
        return u
        
    def update_centers(self, X):
        # 更新聚类中心
        centers = np.zeros((self.n_clusters, X.shape[1]))
        for j in range(self.n_clusters):
            numerator = np.sum([(self.u[i, j] ** self.m) * X[i] for i in range(X.shape[0])], axis=0)
            denominator = np.sum([self.u[i, j] ** self.m for i in range(X.shape[0])])
            # 处理分母为零的情况
            epsilon = 1e-10
            if denominator < epsilon:
                centers[j] = self.centers[j]  # 使用之前的中心值
            else:
                centers[j] = numerator / denominator
        return centers
        
    def fit(self, X):
        # 初始化聚类中心
        self.centers = self.initialize_centers(X)
        
        for _ in range(self.max_iter):
            # 计算隶属度矩阵
            self.u = self.compute_membership(X)
            # 更新聚类中心
            new_centers = self.update_centers(X)
            # 检查收敛
            if np.linalg.norm(new_centers - self.centers) < self.tol:
                break
            self.centers = new_centers
        
    def predict(self, X):
        # 预测聚类结果
        self.compute_membership(X)
        return np.argmax(self.u, axis=1)

# 示例用法
if __name__ == "__main__":
    # 生成测试数据
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    # 创建并训练FCM模型
    fcm = FCM(n_clusters=4)
    fcm.fit(X)
    
    # 获取聚类结果
    y_pred = fcm.predict(X)
    
    # 可视化结果
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
    plt.scatter(fcm.centers[:, 0], fcm.centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title('FCM Clustering Results')
    plt.show()