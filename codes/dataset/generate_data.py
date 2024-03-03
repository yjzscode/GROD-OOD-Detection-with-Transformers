import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

config_path = "D:\Academic\毕设\codes\config\config0.yaml"
config = OmegaConf.load(config_path)


class generate_train_data:
    def __init__(self, d0, n, seed, K):
        super(generate_train_data, self).__init__()

        self.d0 = d0
        self.n = n
        self.seed = seed
        self.num_class = K

    def generate(self):

        # 设置随机种子
        torch.manual_seed(self.seed)

        # 生成 K 个簇的ID数据，一簇OOD数据
        # 簇的数量 self.num_class
        # 数据维度 self.d0
        # 总数据量 self.n
        cluster_data = []

        # 生成第 K+1 簇数据
        far_mean = torch.randn(self.d0) * 20  # 选择一个远离前 n 簇数据的均值
        far_std = torch.rand(self.d0) + 1.0  # 随机生成标准差
        far_cluster_size = torch.randint(low=50, high=100, size=(1,))  # 第 n+1 簇数据点数量
        far_cluster = torch.randn(far_cluster_size, self.d0) * far_std + far_mean
        # 清理数据：删除超出三个标准差范围之外的数据点
        mask = (far_cluster > (far_mean - 3 * far_std)) & (far_cluster < (far_mean + 3 * far_std))
        mask = mask.all(dim=1)
        far_cluster = far_cluster[mask]           
        
        # 生成 ID 数据
        n_ID = self.n - far_cluster_size
        mean_list = []
        std_list = []
        for i in range(self.num_class):
            mean = torch.randn(self.d0) * 5  # 随机生成每个簇的均值
            std = torch.rand(self.d0) + 1.0  # 随机生成每个簇的标准差
            mean_list.append(mean)
            std_list.append(std)

            cluster_size = int(n_ID // self.num_class)  # 每个簇的数据点数量
            cluster = torch.randn(cluster_size, self.d0) * std + mean
            
            # 清理数据：删除超出三个标准差范围之外的数据点
            mask = (cluster > (mean - 3 * std)) & (cluster < (mean + 3 * std))
            mask = mask.all(dim=1)
            cleaned_cluster = cluster[mask]           
            cluster_data.append(cleaned_cluster)
        
        # 加上OOD数据
        cluster_data.append(far_cluster)
        mean_list.append(far_mean)
        std_list.append(far_std)

        # 调整每个簇的数据数量，使得所有簇的数据总量为 N
        total_points = sum(len(cluster) for cluster in cluster_data)
        if total_points < self.n:
            # 增加数据点
            remaining_points = self.n - total_points
            for i in range(remaining_points):
                idx = i % (self.num_class + 1)  # 循环添加数据点以平衡各个簇的数据数量
                new_point = torch.randn(1, self.d0) * std_list[idx] + mean_list[idx]
                cluster_data[idx] = torch.cat([cluster_data[idx], new_point])
        return cluster_data


train_data_func = generate_train_data(config.d0, config.n * 0.8, config.seed_data, config.K)
test_data_func = generate_train_data(config.d0, config.n * 0.8, config.seed_data, config.K)

train_data = train_data_func()
test_data = test_data_func()

# 可视化数据 2维
plt.figure(figsize=(8, 6))
for cluster in train_data:
    plt.scatter(cluster[:, 0], cluster[:, 1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cleaned and Adjusted Data')
plt.show()

# 大于2维可以PCA
# Note: 随机种子固定，test会是train的一部分吗？
# 标签还没加!!!
