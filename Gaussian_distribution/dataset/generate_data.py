import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import matplotlib.cm as cm 

# config_path = '/home/wang1/ZYJ/OOD/codes1_03.11/config/config0.yaml'
# config = OmegaConf.load(config_path)
from torch.utils.data import Dataset
# (N,d0)
torch.manual_seed(42)
class generate_train_data:
    def __init__(self, d0, n, seed, K):
        super(generate_train_data, self).__init__()

        self.d0 = d0
        self.n = n
        self.seed = seed
        self.num_class = K

    def __call__(self, mean_list, std_list):

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
            label.append([i]*cleaned_cluster.size()[0])
        

        # 调整每个簇的数据数量，使得所有簇的数据总量为 N
        total_points = sum(len(cluster) for cluster in cluster_data)
        if total_points < self.n:
            # 增加数据点
            remaining_points = int(self.n - total_points)
            for i in range(remaining_points):
                idx = i % (self.num_class)  # 循环添加数据点以平衡各个簇的数据数量
                new_point = torch.randn(1, self.d0) * std_list[idx] + mean_list[idx]
                cluster_data.append(new_point)
                label.append([idx])
        label = [num for row in label for num in row]
        label = torch.tensor(label)
        cluster_data = torch.cat(cluster_data) #(N,d)
        cluster_data = torch.unsqueeze(cluster_data,-1)
#         print(cluster_data)
        return cluster_data, label
    
class generate_test_data:
    def __init__(self, d0, n, seed, K):
        super(generate_test_data, self).__init__()

        self.d0 = d0
        self.n = n
        self.seed = seed
        self.num_class = K

    def __call__(self, mean_list, std_list):

        # 设置随机种子
        torch.manual_seed(self.seed)

        # 生成 K 个簇的ID数据，一簇OOD数据
        # 簇的数量 self.num_class
        # 数据维度 self.d0
        # 总数据量 self.n
        cluster_data = []
        label = []
        # 生成第 K+1 簇数据
        far_mean = mean_list[self.num_class]  
        far_std = std_list[self.num_class]    # 随机生成标准差
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
            label.append([i]*cleaned_cluster.size()[0])
        # 加上OOD数据
        cluster_data.append(far_cluster)
        label.append([self.num_class] * far_cluster.size()[0])

        # 调整每个簇的数据数量，使得所有簇的数据总量为 N
        total_points = sum(len(cluster) for cluster in cluster_data)
        if total_points < self.n:
            # 增加数据点
            remaining_points = int(self.n - total_points)
            for i in range(remaining_points):
                idx = i % (self.num_class + 1)  # 循环添加数据点以平衡各个簇的数据数量
                new_point = torch.randn(1, self.d0) * std_list[idx] + mean_list[idx]
                cluster_data.append(new_point)
                label.append([idx])
        cluster_data = torch.cat(cluster_data) # (N,d0)
        cluster_data = torch.unsqueeze(cluster_data,-1)
        label = [num for row in label for num in row]
        label = torch.tensor(label)
#         print(cluster_data, label)
        return cluster_data, label

#############################################################组合
class myDataset(Dataset):
    def __init__(self, data, label):
        self.data = data  # n = 1 (N,d0,1)
        #5个数据的标签
        self.label = label
    
    #根据索引获取data和label
    def __getitem__(self,index):
        return self.data[index], self.label[index] #以元组的形式返回

    #获取数据集的大小
    def __len__(self):
        return len(self.data)
###########################################################
    
train_data_func = generate_test_data(2, 10000, 42, 4)
test_data_func = generate_test_data(2, 1000, 42, 2)
mean_list = []

std_list = []
mean_list_g = []
std_list_g = []
for i in range(2):
    mean = torch.abs(torch.randn(2))   # 随机生成每个簇的均值
    std = i / 10 * torch.abs(torch.rand(2)) + 0.1  # 随机生成每个簇的标准差
    mean_list.append(mean)
    std_list.append(std)
    mean_list_g.append(mean)
    std_list_g.append(std)
mean = -torch.abs(torch.randn(2)) * 5  # 随机生成每个簇的均值
std = 0.2 * torch.abs(torch.rand(2)) + 0.1  # 随机生成每个簇的标准差
mean_list.append(mean)
std_list.append(std)
print(mean_list)

mean_g = torch.abs(torch.randn(2)) * 5  # 随机生成每个簇的均值
std_g = 0.3 * torch.abs(torch.rand(2)) + 0.1  # 随机生成每个簇的标准差
mean_list_g.append(mean_g)
std_list_g.append(std_g)

mean_g = torch.randn(2) * 4  # 随机生成每个簇的均值
mean_g[0] = torch.abs(mean_g[0])
mean_g[1] = - torch.abs(mean_g[1])
std_g = 0.4 * torch.abs(torch.rand(2)) + 0.1  # 随机生成每个簇的标准差
mean_list_g.append(mean_g)
std_list_g.append(std_g)

mean_g = torch.randn(2) * 3  # 随机生成每个簇的均值
mean_g[0] = -torch.abs(mean_g[0])
mean_g[1] = torch.abs(mean_g[1])
std_g = 0.3 * torch.abs(torch.rand(2)) + 0.1  # 随机生成每个簇的标准差
mean_list_g.append(mean_g)
std_list_g.append(std_g)

train_data = train_data_func(mean_list_g, std_list_g)[0]
train_label = train_data_func(mean_list_g, std_list_g)[1]

test_data = test_data_func(mean_list, std_list)[0]
test_label = test_data_func(mean_list, std_list)[1]

data = myDataset(train_data, train_label)
print(f'data size is : {len(data)}')

print(data[1]) #获取索引为1的data和label

# 可视化数据 2维
plt.figure(figsize=(8, 6),dpi=100)
plt.scatter(train_data[:, 0], train_data[:, 1],s = 5,alpha = 0.5, color=(130/256, 176/256, 210/256),label='Train')
plt.scatter(test_data[:, 0], test_data[:, 1],s = 5,alpha = 0.5, color=(190/256,184/256,220/256),label='Test')
plt.xlabel('X',fontsize=15)
plt.ylabel('Y',fontsize=15)
plt.legend(loc="best",markerscale=2.,numpoints=1,scatterpoints=1,fontsize=15)
plt.title('Add Rounded Clusters of OOD Data',fontsize=15)
plt.savefig(r'./dataset/rounded_generation.png')
plt.show()
