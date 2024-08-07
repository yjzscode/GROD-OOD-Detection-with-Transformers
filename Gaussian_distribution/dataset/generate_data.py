import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import matplotlib.cm as cm 

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

        # set random seed
        torch.manual_seed(self.seed)

        # ID K CLUSTERS, OOD one cluster
        # number of clusters: self.num_class
        # data dimention: self.d0
        # the amount of data: self.n
        cluster_data = []
        label = []
 
        # generate ID
        n_ID = self.n 
        for i in range(self.num_class):
            mean = mean_list[i]  
            std = std_list[i] 
    
            cluster_size = int(n_ID // self.num_class)  
            cluster = torch.randn(cluster_size, self.d0) * std + mean
            
            # Clean the data: remove data points outside the range of three standard deviations
            mask = (cluster > (mean - 3 * std)) & (cluster < (mean + 3 * std))
            mask = mask.all(dim=1)
            cleaned_cluster = cluster[mask]           
            cluster_data.append(cleaned_cluster)
            label.append([i]*cleaned_cluster.size()[0])
        

        # Adjust the amount of data in each cluster so that the total amount of data in all clusters is N
        total_points = sum(len(cluster) for cluster in cluster_data)
        if total_points < self.n:
            remaining_points = int(self.n - total_points)
            for i in range(remaining_points):
                idx = i % (self.num_class)  # Loop to add data points to balance the number of data points in each cluster
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


        torch.manual_seed(self.seed)

        cluster_data = []
        label = []
        # generate the (K+1)-th cluster
        far_mean = mean_list[self.num_class]  
        far_std = std_list[self.num_class]   
        far_cluster_size = int(self.n // (self.num_class + 1))
        far_cluster = torch.randn(far_cluster_size, self.d0) * far_std + far_mean
        # Clean the data: remove data points outside the range of three standard deviations
        mask = (far_cluster > (far_mean - 3 * far_std)) & (far_cluster < (far_mean + 3 * far_std))
        mask = mask.all(dim=1)
        far_cluster = far_cluster[mask]           
        
        # generate ID
        n_ID = self.n - far_cluster_size
        for i in range(self.num_class):
            mean = mean_list[i]  
            std = std_list[i] 
    
            cluster_size = int(n_ID // self.num_class)  
            cluster = torch.randn(cluster_size, self.d0) * std + mean
            
            
            mask = (cluster > (mean - 3 * std)) & (cluster < (mean + 3 * std))
            mask = mask.all(dim=1)
            cleaned_cluster = cluster[mask]           
            cluster_data.append(cleaned_cluster)
            label.append([i]*cleaned_cluster.size()[0])
        # append OOD
        cluster_data.append(far_cluster)
        label.append([self.num_class] * far_cluster.size()[0])

        
        total_points = sum(len(cluster) for cluster in cluster_data)
        if total_points < self.n:
            remaining_points = int(self.n - total_points)
            for i in range(remaining_points):
                idx = i % (self.num_class + 1)  
                new_point = torch.randn(1, self.d0) * std_list[idx] + mean_list[idx]
                cluster_data.append(new_point)
                label.append([idx])
        cluster_data = torch.cat(cluster_data) # (N,d0)
        cluster_data = torch.unsqueeze(cluster_data,-1)
        label = [num for row in label for num in row]
        label = torch.tensor(label)
#         print(cluster_data, label)
        return cluster_data, label

#############################################################
class myDataset(Dataset):
    def __init__(self, data, label):
        self.data = data  # n = 1 (N,d0,1)
        self.label = label
    
    def __getitem__(self,index):
        return self.data[index], self.label[index] 

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
    mean = torch.abs(torch.randn(2))   # Randomly generate the mean for each cluster
    std = i / 10 * torch.abs(torch.rand(2)) + 0.1  # Randomly generate the standard deviation for each cluster
    mean_list.append(mean)
    std_list.append(std)
    mean_list_g.append(mean)
    std_list_g.append(std)
mean = -torch.abs(torch.randn(2)) * 5 
std = 0.2 * torch.abs(torch.rand(2)) + 0.1  
mean_list.append(mean)
std_list.append(std)
print(mean_list)

mean_g = torch.abs(torch.randn(2)) * 5  
std_g = 0.3 * torch.abs(torch.rand(2)) + 0.1  
mean_list_g.append(mean_g)
std_list_g.append(std_g)

mean_g = torch.randn(2) * 4  
mean_g[0] = torch.abs(mean_g[0])
mean_g[1] = - torch.abs(mean_g[1])
std_g = 0.4 * torch.abs(torch.rand(2)) + 0.1 
mean_list_g.append(mean_g)
std_list_g.append(std_g)

mean_g = torch.randn(2) * 3  
mean_g[0] = -torch.abs(mean_g[0])
mean_g[1] = torch.abs(mean_g[1])
std_g = 0.3 * torch.abs(torch.rand(2)) + 0.1  
mean_list_g.append(mean_g)
std_list_g.append(std_g)

train_data = train_data_func(mean_list_g, std_list_g)[0]
train_label = train_data_func(mean_list_g, std_list_g)[1]

test_data = test_data_func(mean_list, std_list)[0]
test_label = test_data_func(mean_list, std_list)[1]

data = myDataset(train_data, train_label)
print(f'data size is : {len(data)}')

print(data[1]) 

# visualization
plt.figure(figsize=(8, 6),dpi=100)
plt.scatter(train_data[:, 0], train_data[:, 1],s = 5,alpha = 0.5, color=(130/256, 176/256, 210/256),label='Train')
plt.scatter(test_data[:, 0], test_data[:, 1],s = 5,alpha = 0.5, color=(190/256,184/256,220/256),label='Test')
plt.xlabel('X',fontsize=15)
plt.ylabel('Y',fontsize=15)
plt.legend(loc="best",markerscale=2.,numpoints=1,scatterpoints=1,fontsize=15)
plt.title('Add Rounded Clusters of OOD Data',fontsize=15)
plt.savefig(r'./dataset/rounded_generation.png')
plt.show()
