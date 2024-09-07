"""
使用 DBSACN 算法，预处理一遍 图像组。
选出具有代表性的中心。
"""
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN as dbscan

from catface_hav_v1.structs import Face


def calculate_cluster_centers(embeddings, labels):
    """
    计算聚类中心，实际就是 区平均值。
    将离群值也全都考虑进去。
    :param embeddings:
    :param labels:
    :return:
    """
    centers = []
    unique_labels = set(labels)
    for label in unique_labels:
        points = embeddings[labels == label]
        if label != -1:  # 排除噪声点
            center = points.mean(axis=0)
            centers.append({
                'embedding': center,
                'cnt': len(points)
            })
        else:  # 添加所有的离群值
            for point in points:
                centers.append({
                    'embedding': point,
                    'cnt': 0.8
                })
    return centers


class DBSCAN:
    _min_samples = 2

    def __init__(self, eps=None, k=4, distance_ratio=.7, **kwargs):
        self.eps = eps
        self.k = k
        self.distance_ratio = distance_ratio
        self.verbose = kwargs.get('verbose', False)

    def filtrate_embeddings(self, faces, **kwargs):
        """
        过滤 embedding，得到具有代表性的目标。
        :param faces: 依靠 check_embeddings() 具有兼容 embedding[] && Face[] 的能力，

        :param show_pca: 是否展示 pca 3D 的效果。
        :return:
        """
        show_pca_3D = kwargs.get('show_pca', False)

        embeddings = check_embeddings(faces)
        if len(embeddings) < 2:
            return embeddings

        # cal similarity && distance
        X = np.array(embeddings)
        sim_matrix = np.dot(X, X.T)

        sim_matrix -= np.min(sim_matrix)
        sim_matrix /= np.max(sim_matrix)

        distance_matrix = 1 - sim_matrix

        # dbscan
        if not self.eps:
            neighbors = NearestNeighbors(n_neighbors=self.k, metric='precomputed')
            neighbors_fit = neighbors.fit(distance_matrix)
            distances, indices = neighbors_fit.kneighbors(distance_matrix)

            # 对距离进行排序
            distances = np.sort(distances, axis=0)
            k_distances = distances[:, self.k - 1]  # 第k个最近邻居的距离

            # 根据k-距离图选择eps
            eps = k_distances[int(len(k_distances) * self.distance_ratio)]
            print(f"eps: {eps}")
        else:
            eps = self.eps

        db = dbscan(eps=eps, min_samples=self._min_samples, metric='precomputed').fit(distance_matrix)
        labels = db.labels_

        if self.verbose:
            img_labels = np.array(range(0, len(embeddings)))
            # 输出每个聚类的点位索引及其对应的 img_labels
            for cluster_label in set(labels):
                cluster_indices = np.where(labels == cluster_label)[0]
                cluster_labels = img_labels[cluster_indices]
                if cluster_label == -1:
                    # print(f'噪声点索引: {cluster_indices}')
                    print(f'噪声点对应的标签: {cluster_labels}')
                else:
                    # print(f'聚类 {cluster_label} 的点位索引: {cluster_indices}')
                    print(f'聚类 {cluster_label} 的点位对应的标签: {cluster_labels}')

        # 计算聚类中心
        centers = calculate_cluster_centers(X, labels)

        if show_pca_3D:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA

            centers_embedding = np.array([_['embedding'] for _ in centers if _['cnt'] >= 1])
            if len(centers_embedding) == 0:
                return centers
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X)
            centers_pca = pca.transform(centers_embedding)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # 为每个聚类设置不同的颜色
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='viridis', label='Data Points')
            # 绘制聚类中心
            ax.scatter(centers_pca[:, 0], centers_pca[:, 1], centers_pca[:, 2], s=100, c='red', marker='X',
                       label='Centers')

            # 图例和标签
            plt.colorbar(scatter, ax=ax)
            ax.set_title('PCA Reduction to 3D')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            ax.legend()

            plt.show()

        return centers


def check_embeddings(objs):
    """
    使 DBSCAN 同时兼容 处理 faces && 直接处理 embeddings
    :param objs:
    :return:
    """
    if len(objs) == 0:
        return []
    obj = objs[0]
    print(obj[:3])
    if isinstance(obj, np.ndarray):
        return objs
    elif isinstance(obj, Face):
        return [obj.embedding for obj in objs]
    else:
        raise ValueError(f"Unknown object {type(obj)}")








