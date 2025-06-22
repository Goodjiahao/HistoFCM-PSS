import numpy as np
from fcm_histogram_init import FCMHistogramInit

class FCMIncompleteData(FCMHistogramInit):
    def compute_membership(self, X):
        "使用部分距离策略计算隶属度矩阵，处理含缺失值(NaN)的数据"
        n_samples = X.shape[0]
        u = np.zeros((n_samples, self.n_clusters))
        
        for i in range(n_samples):
            # 获取样本i的非缺失特征索引
            valid_features = ~np.isnan(X[i])
            if not np.any(valid_features):
                # 若所有特征都缺失，设为均匀隶属度
                u[i] = 1 / self.n_clusters
                continue
            
            # 对每个聚类中心计算部分距离
            distances = []
            for j in range(self.n_clusters):
                # 仅使用非缺失特征计算距离
                x_valid = X[i, valid_features]
                center_valid = self.centers[j, valid_features]
                # 计算欧氏距离
                dist = np.linalg.norm(x_valid - center_valid)
                # 防止距离为0导致除零错误
                distances.append(dist if dist > 1e-10 else 1e-10)
            
            # 计算隶属度
            distances = np.array(distances)
            power_term = 2 / (self.m - 1)
            u[i] = 1 / np.sum((distances[:, np.newaxis] / distances) ** power_term, axis=0)
        return u
    
    def update_centers(self, X):
        "更新聚类中心时考虑缺失值"
        centers = np.zeros((self.n_clusters, X.shape[1]))
        for j in range(self.n_clusters):
            # 计算每个特征的有效样本权重和
            for feature_idx in range(X.shape[1]):
                # 找出该特征非缺失的样本
                valid_samples = ~np.isnan(X[:, feature_idx])
                if np.any(valid_samples):
                    # 使用权重平均计算中心
                    weights = self.u[valid_samples, j] ** self.m
                    centers[j, feature_idx] = np.sum(weights * X[valid_samples, feature_idx]) / np.sum(weights)
                else:
                    # 若所有样本该特征都缺失，保持原中心值
                    centers[j, feature_idx] = self.centers[j, feature_idx]
        return centers

# 示例用法：处理含缺失值的数据
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    
    # 生成测试数据并人为添加缺失值
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    # 随机设置10%的数据为缺失值
    np.random.seed(42)
    mask = np.random.rand(*X.shape) < 0.1
    X[mask] = np.nan
    
    # 创建并训练改进的FCM模型
    fcm = FCMIncompleteData(n_clusters=4)
    fcm.fit(X)
    
    # 获取聚类结果
    y_pred = fcm.predict(X)
    
    # 可视化结果
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
    plt.scatter(fcm.centers[:, 0], fcm.centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title('Improved FCM with Histogram Init and Partial Distance for Incomplete Data')
    plt.show()