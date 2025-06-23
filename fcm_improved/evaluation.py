import numpy as np
import time
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from fcm_base import FCM
from fcm_partial_distance import FCMIncompleteData
import matplotlib.pyplot as plt
# 设置中文字体支持
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子以确保结果可复现
np.random.seed(789)

# 生成测试数据
def generate_incomplete_data(n_samples=300, n_centers=4, missing_rate=0.1):
    X, y_true = make_blobs(n_samples=n_samples, centers=n_centers, 
                          cluster_std=0.60, random_state=42)
    # 随机添加缺失值
    mask = np.random.rand(*X.shape) < missing_rate
    X[mask] = np.nan
    return X, y_true

# 评估不同算法的性能
def evaluate_algorithms(X, y_true, n_clusters=4):
    results = {}
    
    # 1. 改进的FCM算法
    start_time = time.time()
    fcm_improved = FCMIncompleteData(n_clusters=n_clusters)
    fcm_improved.fit(X)
    y_pred_fcm_improved = fcm_improved.predict(X)
    time_fcm_improved = time.time() - start_time
    
    # 2. 标准FCM算法（需要处理缺失值，这里用均值填充）
    X_filled = X.copy()
    for i in range(X_filled.shape[1]):
        mask = ~np.isnan(X_filled[:, i])
        X_filled[~mask, i] = np.mean(X_filled[mask, i])
    
    start_time = time.time()
    fcm_standard = FCM(n_clusters=n_clusters)
    fcm_standard.fit(X_filled)
    y_pred_fcm_standard = fcm_standard.predict(X_filled)
    time_fcm_standard = time.time() - start_time
    
    # 3. K-means算法（用于对比）
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_pred_kmeans = kmeans.fit_predict(X_filled)
    time_kmeans = time.time() - start_time
    
    # 计算评估指标
    results['Improved FCM'] = {
        'ARI': adjusted_rand_score(y_true, y_pred_fcm_improved),
        'NMI': normalized_mutual_info_score(y_true, y_pred_fcm_improved),
        'Time': time_fcm_improved
    }
    
    results['Standard FCM'] = {
        'ARI': adjusted_rand_score(y_true, y_pred_fcm_standard),
        'NMI': normalized_mutual_info_score(y_true, y_pred_fcm_standard),
        'Time': time_fcm_standard
    }
    
    results['K-means'] = {
        'ARI': adjusted_rand_score(y_true, y_pred_kmeans),
        'NMI': normalized_mutual_info_score(y_true, y_pred_kmeans),
        'Time': time_kmeans
    }
    
    return results, y_pred_fcm_improved, y_pred_fcm_standard, y_pred_kmeans

# 绘制结果对比图
def plot_comparison(X, y_true, y_pred_fcm_improved, y_pred_fcm_standard, y_pred_kmeans):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 改进的FCM结果
    axes[0].scatter(X[:, 0], X[:, 1], c=y_pred_fcm_improved, s=50, cmap='viridis')
    axes[0].set_title('改进FCM聚类结果')
    axes[0].set_xlabel('特征1')
    axes[0].set_ylabel('特征2')
    axes[0].text(0.05, 0.95, '● 不同颜色表示不同聚类类别\n● 本算法使用直方图初始化聚类中心\n● 支持处理含缺失值的数据', 
                 transform=axes[0].transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 标准FCM结果
    axes[1].scatter(X[:, 0], X[:, 1], c=y_pred_fcm_standard, s=50, cmap='viridis')
    axes[1].set_title('标准FCM聚类结果')
    axes[1].set_xlabel('特征1')
    axes[1].set_ylabel('特征2')
    axes[1].text(0.05, 0.95, '● 不同颜色表示不同聚类类别\n● 使用随机初始化聚类中心\n● 需要预先填充缺失值才能运行', 
                 transform=axes[1].transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # K-means结果
    axes[2].scatter(X[:, 0], X[:, 1], c=y_pred_kmeans, s=50, cmap='viridis')
    axes[2].set_title('K-means聚类结果')
    axes[2].set_xlabel('特征1')
    axes[2].set_ylabel('特征2')
    axes[2].text(0.05, 0.95, '● 不同颜色表示不同聚类类别\n● 传统硬聚类算法\n● 需要预先填充缺失值才能运行', 
                 transform=axes[2].transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('三种聚类算法结果对比', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为标题留出空间
    plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形避免显示

# 主函数
if __name__ == "__main__":
    # 生成含10%缺失值的数据
    X, y_true = generate_incomplete_data(missing_rate=0.1)
    
    # 评估算法
    results, y_pred_fcm_improved, y_pred_fcm_standard, y_pred_kmeans = evaluate_algorithms(X, y_true)
    
    # 打印评估结果
    print("Clustering Performance Comparison:")
    print("---------------------------------")
    for alg, metrics in results.items():
        print(f"{alg}:")
        print(f"  ARI: {metrics['ARI']:.4f}")
        print(f"  NMI: {metrics['NMI']:.4f}")
        print(f"  Time: {metrics['Time']:.4f} seconds")
        print()
    
    # 绘制对比图
    plot_comparison(X, y_true, y_pred_fcm_improved, y_pred_fcm_standard, y_pred_kmeans)