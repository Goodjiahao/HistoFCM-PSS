import numpy as np
from fcm_base import FCM
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
font = {'family': plt.rcParams["font.family"][0]}

class FCMHistogramInit(FCM):
    def initialize_centers(self, X):
        "使用直方图峰值初始化聚类中心"
        n_features = X.shape[1]
        # 初始化中心数组 (n_clusters, n_features)
        centers = np.zeros((self.n_clusters, n_features))
        
        # 对每个特征维度计算直方图并寻找峰值
        for feature_idx in range(n_features):
            feature_data = X[:, feature_idx]
            non_nan_samples = feature_data[~np.isnan(feature_data)]
            
            if len(non_nan_samples) == 0:
                # 如果该特征所有值都是NaN，则使用0作为中心
                peak_centers = np.zeros(self.n_clusters)
            else:
                # 计算直方图
                hist, bin_edges = np.histogram(non_nan_samples, bins='auto')

                # 绘制并保存直方图
                plt.figure(figsize=(10, 6))
                plt.hist(non_nan_samples, bins=bin_edges, alpha=0.7, color='blue')
                plt.title(f'特征 {feature_idx} 的直方图分布', fontproperties=font)
                plt.xlabel('特征值', fontproperties=font)
                plt.ylabel('频率', fontproperties=font)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                os.makedirs('histograms', exist_ok=True)
                plt.savefig(f'histograms/histogram_feature_{feature_idx}.png', dpi=300, bbox_inches='tight')
                plt.close()

                # 寻找峰值
                peaks, _ = find_peaks(hist, height=0.1*np.max(hist))  # 阈值设为最大频率的10%
                # 取峰值对应的bin中心作为初始中心候选
                peak_centers = (bin_edges[peaks] + bin_edges[peaks+1]) / 2 if len(peaks) > 0 else []
                
                # 如果峰值数量小于聚类数，用随机样本补充
                if len(peak_centers) < self.n_clusters:
                    additional_samples = np.random.choice(non_nan_samples, self.n_clusters - len(peak_centers), replace=False)
                    peak_centers = np.concatenate([peak_centers, additional_samples])
                # 如果峰值数量大于聚类数，随机选择n_clusters个
                elif len(peak_centers) > self.n_clusters:
                    peak_centers = np.random.choice(peak_centers, self.n_clusters, replace=False)
            
            # 将当前特征的峰值中心分配给所有聚类
            centers[:, feature_idx] = peak_centers
        
        return centers

# 示例用法
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    
    # 生成测试数据
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    # 创建并训练改进的FCM模型
    fcm = FCMHistogramInit(n_clusters=4)
    fcm.fit(X)
    
    # 获取聚类结果
    y_pred = fcm.predict(X)
    
    # 可视化结果
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
    plt.scatter(fcm.centers[:, 0], fcm.centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title('FCM with Histogram Initialization Results')
    plt.show()