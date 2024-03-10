import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

config_path = "D:\Academic\毕设\codes\config\config0.yaml"
config = OmegaConf.load(config_path)
from torch.utils.data import Dataset
# (N,d0)

class generate_train_data:
    def __init__(self, d0, n, seed, K):
        super(generate_train_data, self).__init__()

        self.d0 = d0
        self.n = n
        self.seed = seed
        self.num_class = K

    def generate(self, mean_list, std_list):

        # 设置随机种子
        torch.manual_seed(self.seed)

        # 生成 K 个簇的ID数据，一簇OOD数据
        # 簇的数量 self.num_class
        # 数据维度 self.d0
        # 总数据量 self.n
        cluster_data = []
        label = []
 
        # 生成 ID 数据
        n_ID = self.n 
        for i in range(self.num_class):
            mean = mean_list[i]  # 随机生成每个簇的均值
            std = std_list[i] # 随机生成每个簇的标准差
    
            cluster_size = int(n_ID // self.num_class)  # 每个簇的数据点数量
            cluster = torch.randn(cluster_size, self.d0) * std + mean
            
            # 清理数据：删除超出三个标准差范围之外的数据点
            mask = (cluster > (mean - 3 * std)) & (cluster < (mean + 3 * std))
            mask = mask.all(dim=1)
            cleaned_cluster = cluster[mask]           
            cluster_data.append(cleaned_cluster)
            label.append([i+1]*cleaned_cluster[0])
        

        # 调整每个簇的数据数量，使得所有簇的数据总量为 N
        total_points = sum(len(cluster) for cluster in cluster_data)
        if total_points < self.n:
            # 增加数据点
            remaining_points = self.n - total_points
            for i in range(remaining_points):
                idx = i % (self.num_class)  # 循环添加数据点以平衡各个簇的数据数量
                new_point = torch.randn(1, self.d0) * std_list[idx] + mean_list[idx]
                cluster_data[idx] = torch.cat([cluster_data[idx], new_point])
                label.append([idx+1])
        return cluster_data, label
    
class generate_test_data:
    def __init__(self, d0, n, seed, K):
        super(generate_test_data, self).__init__()

        self.d0 = d0
        self.n = n
        self.seed = seed
        self.num_class = K

    def generate(self, mean_list, std_list):

        # 设置随机种子
        torch.manual_seed(self.seed)

        # 生成 K 个簇的ID数据，一簇OOD数据
        # 簇的数量 self.num_class
        # 数据维度 self.d0
        # 总数据量 self.n
        cluster_data = []
        label = []
        # 生成第 K+1 簇数据
        far_mean = mean_list[self.K]  
        far_std = std_list[self.K]    # 随机生成标准差
        far_cluster_size = int(self.n // (self.num_class + 1))
        far_cluster = torch.randn(far_cluster_size, self.d0) * far_std + far_mean
        # 清理数据：删除超出三个标准差范围之外的数据点
        mask = (far_cluster > (far_mean - 3 * far_std)) & (far_cluster < (far_mean + 3 * far_std))
        mask = mask.all(dim=1)
        far_cluster = far_cluster[mask]           
        
        # 生成 ID 数据
        n_ID = self.n - far_cluster_size
        for i in range(self.num_class):
            mean = mean_list[i]  # 随机生成每个簇的均值
            std = std_list[i] # 随机生成每个簇的标准差
    
            cluster_size = int(n_ID // self.num_class)  # 每个簇的数据点数量
            cluster = torch.randn(cluster_size, self.d0) * std + mean
            
            # 清理数据：删除超出三个标准差范围之外的数据点
            mask = (cluster > (mean - 3 * std)) & (cluster < (mean + 3 * std))
            mask = mask.all(dim=1)
            cleaned_cluster = cluster[mask]           
            cluster_data.append(cleaned_cluster)
            label.append([i+1]*cleaned_cluster[0])
        # 加上OOD数据
        cluster_data.append([self.num_class+1] * far_cluster[0])

        # 调整每个簇的数据数量，使得所有簇的数据总量为 N
        total_points = sum(len(cluster) for cluster in cluster_data)
        if total_points < self.n:
            # 增加数据点
            remaining_points = self.n - total_points
            for i in range(remaining_points):
                idx = i % (self.num_class + 1)  # 循环添加数据点以平衡各个簇的数据数量
                new_point = torch.randn(1, self.d0) * std_list[idx] + mean_list[idx]
                cluster_data[idx] = torch.cat([cluster_data[idx], new_point])
                label.append([idx+1])
        return cluster_data, label

#############################################################组合
class myDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data).unsqueeze(-1)  # n = 1 (N,d0,1)
        #5个数据的标签
        self.label = torch.tensor(label)
    
    #根据索引获取data和label
    def __getitem__(self,index):
        return self.data[index], self.label[index] #以元组的形式返回

    #获取数据集的大小
    def __len__(self):
        return len(self.data)
###########################################################
    
train_data_func = generate_train_data(config.d0, config.N * 0.8, config.seed_data, config.K)
test_data_func = generate_test_data(config.d0, config.N * 0.2, config.seed_data, config.K)

mean_list = []
std_list = []
for i in range(config.K):
    mean = torch.randn(config.d0) * 5  # 随机生成每个簇的均值
    std = torch.rand(config.d0) + 1.0  # 随机生成每个簇的标准差
    mean_list.append(mean)
    std_list.append(std)
mean = torch.randn(config.d0) * 200  # 随机生成每个簇的均值
std = torch.rand(config.d0) + 1.0  # 随机生成每个簇的标准差
mean_list.append(mean)
std_list.append(std)

train_data = train_data_func(mean_list, std_list)[0]
train_label = train_data_func(mean_list, std_list)[1]

test_data = test_data_func(mean_list, std_list)[0]
test_label = test_data_func(mean_list, std_list)[1]

data = myDataset(train_data, train_label)
print(f'data size is : {len(data)}')

print(data[1]) #获取索引为1的data和label

# 可视化数据 2维
plt.figure(figsize=(8, 6))
for cluster in train_data:
    plt.scatter(cluster[:, 0], cluster[:, 1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cleaned and Adjusted Data')
plt.show()
