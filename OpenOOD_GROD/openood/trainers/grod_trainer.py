import faiss.contrib.torch_utils
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config
from einops import repeat

torch.autograd.set_detect_anomaly(True)
class GRODTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net
        self.train_loader = train_loader
        self.config = config

        self.n_cls = config.dataset.num_classes


        self.optimizer = torch.optim.AdamW(
            params=net.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max = 10
        )

        self.head = self.net.head
        self.head1 = self.net.head1
        self.alpha = config.trainer.alpha
        self.nums_rounded = config.trainer.nums_rounded
        self.gamma = config.trainer.gamma
        
        self.k = self.net.k


    def train_epoch(self, epoch_idx):
        # adjust_learning_rate(self.config, self.optimizer, epoch_idx - 1)
        # self.net = nn.DataParallel(self.net)
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):

            batch = next(train_dataiter)
            data = batch['data'].to(self.device)
            target = batch['label'].to(self.device)                        
            if data.size(0) / self.n_cls > 2:
                sub_datasets_in_mu = torch.zeros((self.n_cls, 768)).to(self.device) #(K,f)
                
                dataset_in_mu = torch.zeros(768).to(self.device) #(f)
                sub_datasets_in_cov = torch.zeros((self.n_cls, 768, 768)).to(self.device)
                sub_datasets_in_distances = torch.zeros(self.n_cls).to(self.device)
                data_in, feat_lda, feat_pca = self.net(data, target)
                
                # data_in = self.head1(data_in)
                data = data_in

                # generate rounded ood data
                sub_datasets_in = [Subset(data_in, torch.where(target == i)[0]) for i in range(self.n_cls)]
                sub_datasets_lda = [Subset(feat_lda, torch.where(target == i)[0]) for i in range(self.n_cls)]
                argmax = torch.zeros((self.n_cls, feat_lda.size()[1]))
                argmin = torch.zeros((self.n_cls, feat_lda.size()[1]))
                k = 0
                for i in range(self.n_cls):
                    if len(sub_datasets_lda[i]) <= 1:
                        pass
                    else:
                        k = k + 1
                        dataloader = DataLoader(sub_datasets_lda[i], batch_size=64, shuffle=False)
                        for batch in dataloader:
                            tensor_data_lda = batch
                        dataloader = DataLoader(sub_datasets_in[i], batch_size=64, shuffle=False)
                        for batch in dataloader:
                            tensor_data_in = batch
                        argmax[i] = torch.argmax(tensor_data_lda,dim=0) #feat_dim
                        argmax[i] = torch.argmin(tensor_data_lda,dim=0) #feat_dim

                        for j in range(feat_lda.size()[1]):
                            # print(argmax[i][j].item())
                            if k == 1 and j==0:
                                data_rounded_category = tensor_data_in[int(argmax[i][j].item())].unsqueeze(0)
                                data_rounded_category_1 = tensor_data_in[int(argmin[i][j].item())].unsqueeze(0)
                            else:
                                data_rounded_category = torch.cat((data_rounded_category, tensor_data_in[int(argmax[i][j].item())].unsqueeze(0)),dim=0)
                                data_rounded_category_1 = torch.cat((data_rounded_category_1, tensor_data_in[int(argmin[i][j].item())].unsqueeze(0)),dim=0)

                        mean =  torch.mean(tensor_data_in, dim = 0)
                        cov0 = self.calculate_covariance_matrix(tensor_data_in)+1e-4 * torch.eye(mean.size(0)).to(self.device)
                        L = torch.linalg.cholesky(cov0)

                        # 求解下三角矩阵 L 的逆
                        L_inv = torch.linalg.inv(L)

                        # 利用下三角矩阵的逆求解对称正定矩阵 A 的逆
                        cov = torch.mm(L_inv.t(), L_inv)
                        # cov = torch.linalg.inv(self.calculate_covariance_matrix(tensor_data_in)+1e-7 * torch.eye(mean.size(0)).to(self.device))
                        # print(torch.mm(cov0, cov))
                        if torch.max(torch.abs(sub_datasets_in_mu[i,:]))<1e-7:
                            sub_datasets_in_cov[i,:,:] = cov
                            sub_datasets_in_mu[i,:] = mean
                            # print(self.mahalanobis(tensor_data_in, sub_datasets_in_mu, sub_datasets_in_cov))
                            sub_datasets_in_distances[i] = torch.max(self.mahalanobis(tensor_data_in, sub_datasets_in_mu.clone(), sub_datasets_in_cov.clone())[:,i])                                                
                            # print("1",sub_datasets_in_distances[i])
                        
                        sub_datasets_in_cov[i,:,:] = 0.1 * cov.detach().clone().to(self.device) + 0.9 * sub_datasets_in_cov[i,:,:].detach().clone()
                        # print(sub_datasets_in_cov)
                        sub_datasets_in_mu[i,:] = 0.1 * mean.detach().clone().to(self.device) + 0.9 * sub_datasets_in_mu[i,:].detach().clone()
                        dists = self.mahalanobis(tensor_data_in, sub_datasets_in_mu.clone(), sub_datasets_in_cov.clone())[:,i]
                        dist = torch.max(dists)
                        # print(dists.max(), dists.min())
                        sub_datasets_in_distances[i] = 0.1 * dist.to(self.device).detach().clone() + 0.9 * sub_datasets_in_distances[i].detach().clone()
                        # print(i, sub_datasets_in_distances[i])
                        sub_datasets_in_mean =  repeat(sub_datasets_in_mu.clone()[i,:], "f -> b f", 
                                                    f = tensor_data_in.size(1), b = feat_lda.size()[1])
                    
                        A = data_rounded_category[-feat_lda.size()[1]:]
                        A_1 = data_rounded_category_1[- feat_lda.size()[1]:]
                        vector = F.normalize(A.to(self.device) - sub_datasets_in_mean.to(self.device), dim = 1)
                        vector_1 = F.normalize(A_1.to(self.device) - sub_datasets_in_mean.to(self.device), dim = 1)
                        A = A + self.alpha * vector.to(self.device) #(feat_dim, 768)
                        A_1 = A_1 + self.alpha * vector_1.to(self.device) #(feat_dim, 768)
                        if k == 1:
                            mean_matrix_0 = A
                            mean_matrix_1 = A_1
                        else:
                            mean_matrix_0 = torch.cat((mean_matrix_0, A), dim = 0) #(num, 768)
                            mean_matrix_1 = torch.cat((mean_matrix_1, A_1), dim = 0) #(num, 768)
                mean_matrix = torch.cat((mean_matrix_0, mean_matrix_1), dim = 0)
                # print(mean_matrix.size())
                std = 1 / 3 * self.alpha
                mu = mean_matrix.T.unsqueeze(2).to(self.device) #(768,num,1)
                rand_data = torch.randn(mean_matrix.size(1), self.nums_rounded).to(self.device) #(768,nums_rounded)
                gaussian_data = mu + std * rand_data.unsqueeze(1) #(768, num, nums_rounded)
                # print(gaussian_data.size())
                nums = gaussian_data.size(1)
                nums_rounded = gaussian_data.size(2)
                reshaped_rounded_data = gaussian_data.permute(1, 2, 0).contiguous().view(nums * nums_rounded, mean_matrix.size(1)) # (num* nums_rounded, 768)
                data = torch.cat((data, reshaped_rounded_data), dim = 0)
                # print(reshaped_rounded_data.size())

                # generate PCA ood data
                argmax = torch.zeros(feat_pca.size()[1])
                argmax = torch.argmax(feat_pca,dim=0) #feat_dim
                
                argmin = torch.zeros(feat_pca.size()[1])
                argmin = torch.argmin(feat_pca,dim=0) #feat_dim

                for j in range(feat_pca.size()[1]):
                    if j==0:
                        pcadata_rounded_category = data_in[int(argmax[j].item())].unsqueeze(0)
                        pcadata_rounded_category_1 = data_in[int(argmin[j].item())].unsqueeze(0)
                    else:
                        
                        pcadata_rounded_category = torch.cat((pcadata_rounded_category, data_in[int(argmax[j].item())].unsqueeze(0)),dim=0)
                        pcadata_rounded_category_1 = torch.cat((pcadata_rounded_category_1, data_in[int(argmin[j].item())].unsqueeze(0)),dim=0)
                if train_step == 1:
                    dataset_in_mu = torch.mean(data_in, dim = 0)
                dataset_in_mu = 0.1 * torch.mean(data_in.detach().clone(), dim = 0) + 0.9 * dataset_in_mu.detach().clone() 
                
                dataset_in_mu =  repeat(dataset_in_mu.squeeze(), "f -> b f", 
                                                f = data_in.size(1), b = feat_lda.size()[1])
                # print(data_rounded_category.size())
                B = pcadata_rounded_category
                B_1 = pcadata_rounded_category_1
                # print(A.size())
                pcavector = F.normalize(B.clone() - dataset_in_mu, dim = 1)
                pcavector_1 = F.normalize(B_1.clone() - dataset_in_mu, dim = 1)
                B = torch.add(B, self.alpha * pcavector) #(feat_dim, 768)
                B_1 = torch.add(B_1, self.alpha * pcavector_1) #(feat_dim, 768)
                mean_matrix_0 = B
                mean_matrix_1 = B_1
                # print(A.size())
                mean_matrix = torch.cat((mean_matrix_0, mean_matrix_1), dim = 0)
                # mean_matrix = mean_matrix_0
                std = 1 / 3 * self.alpha
                mu = mean_matrix.T.unsqueeze(2).to(self.device) 
                rand_data = torch.randn(mean_matrix.size(1), self.nums_rounded).to(self.device) 
                gaussian_data = mu + std * rand_data.unsqueeze(1) #(768, num, nums_rounded)
                # print(gaussian_data.size())
                nums = gaussian_data.size(1)
                nums_rounded = gaussian_data.size(2)
                reshaped_rounded_data = gaussian_data.permute(1, 2, 0).contiguous().view(nums * nums_rounded, mean_matrix.size(1)) # (num* nums_rounded, 768)
                # print(reshaped_rounded_data.size(),data.size())
                data = torch.cat((data, reshaped_rounded_data), dim = 0)
                # target = torch.cat((target, (self.n_cls) * torch.ones(nums * nums_rounded).to(self.device)), dim = 0)
                # print(data.size())
                # filt id like data
                # print(sub_datasets_in_distances)
                data_add = data[data_in.size(0):]   
                # print(data_add.size())
                
                distances = self.mahalanobis(data_add, sub_datasets_in_mu, sub_datasets_in_cov).to(self.device) #(n,k)
                
                # print(distances, sub_datasets_in_distances)
                # 计算每个样本点的最小距离和对应的类别索引
                min_distances, min_distances_clas = torch.min(distances, dim=1)                       
                # 获取每个样本点对应的子数据集距离
                sub_distances = sub_datasets_in_distances[min_distances_clas.to(self.device)]
                
                k_init = (torch.mean(min_distances / sub_distances) - 1) * 10
                # 找到满足条件的索引
                mask = min_distances > (1 + k_init * self.k.to(self.device)[0]) * sub_distances
                
                
                # # 找到满足条件的索引
                # mask = min_distances > (1 + self.k.to(self.device)[0]) * sub_distances
                # print(min_distances,sub_distances)
                # 使用布尔索引删除满足条件的数据点
                cleaned_data_add = data_add[mask.to(self.device)]
                
                if cleaned_data_add.size(0) > data_in.size(0) // self.n_cls + 2: #ood太多则随机删除
                    delete_num = cleaned_data_add.size(0) - (data_in.size(0) // self.n_cls + 2)
                    indices = torch.randperm(cleaned_data_add.size(0))[:(data_in.size(0) // self.n_cls + 2)].to(self.device)
                    cleaned_data_add_de = cleaned_data_add[indices]
                else: 
                    cleaned_data_add_de = cleaned_data_add
                    
                
                data = torch.cat((data[:data_in.size(0)], cleaned_data_add_de), dim = 0)


                target = torch.cat((target, (self.n_cls) * torch.ones(cleaned_data_add_de.size(0)).to(self.device)), dim = 0)
                
            else:
                sub_datasets_in_mu = torch.zeros((self.n_cls, 768)).to(self.device) #(K,f)
                dataset_in_mu = torch.zeros(768).to(self.device) #(f)
                sub_datasets_in_cov = torch.zeros((self.n_cls, 768, 768)).to(self.device)
                sub_datasets_in_distances = torch.zeros(self.n_cls).to(self.device)
                data_in, feat_lda, feat_pca = self.net(data, target)
                # data_in = self.head1(data_in)
                data = data_in
                # generate PCA ood data
                argmax = torch.zeros(feat_pca.size()[1])
                argmax = torch.argmax(feat_pca,dim=0) #feat_dim
                
                argmin = torch.zeros(feat_pca.size()[1])
                argmin = torch.argmin(feat_pca,dim=0) #feat_dim

                for j in range(feat_pca.size()[1]):
                    if j==0:
                        pcadata_rounded_category = data_in[int(argmax[j].item())].unsqueeze(0)
                        pcadata_rounded_category_1 = data_in[int(argmin[j].item())].unsqueeze(0)
                    else:
                        
                        pcadata_rounded_category = torch.cat((pcadata_rounded_category, data_in[int(argmax[j].item())].unsqueeze(0)),dim=0)
                        pcadata_rounded_category_1 = torch.cat((pcadata_rounded_category_1, data_in[int(argmin[j].item())].unsqueeze(0)),dim=0)
                if train_step == 1:
                        dataset_in_mu = torch.mean(data_in, dim = 0)
                dataset_in_mu = 0.1 * torch.mean(data_in.detach().clone(), dim = 0) + 0.9 * dataset_in_mu.detach().clone() 
                cov0 = self.calculate_covariance_matrix(data_in)+1e-4 * torch.eye(dataset_in_mu.size(0)).to(self.device)
                L = torch.linalg.cholesky(cov0)

                # 求解下三角矩阵 L 的逆
                L_inv = torch.linalg.inv(L)

                # 利用下三角矩阵的逆求解对称正定矩阵 A 的逆
                cov = torch.mm(L_inv.t(), L_inv)
                cov = torch.unsqueeze(cov, dim=0)
                dataset_in_mean =  repeat(dataset_in_mu.squeeze(), "f -> b f", 
                                                f = data_in.size(1), b = feat_lda.size()[1])
                # dataset_in_mu = torch.unsqueeze(dataset_in_mu, dim=0)
                # print(data_rounded_category.size())
                B = pcadata_rounded_category
                B_1 = pcadata_rounded_category_1
                # print(A.size())
                pcavector = F.normalize(B.clone() - dataset_in_mean, dim = 1)
                pcavector_1 = F.normalize(B_1.clone() - dataset_in_mean, dim = 1)
                B = torch.add(B, self.alpha * pcavector) #(feat_dim, 768)
                B_1 = torch.add(B_1, self.alpha * pcavector_1) #(feat_dim, 768)
                mean_matrix_0 = B
                mean_matrix_1 = B_1
                # print(A.size())
                mean_matrix = torch.cat((mean_matrix_0, mean_matrix_1), dim = 0)
                # mean_matrix = mean_matrix_0
                std = 1 / 3 * self.alpha
                mu = mean_matrix.T.unsqueeze(2).to(self.device) 
                rand_data = torch.randn(mean_matrix.size(1), self.nums_rounded).to(self.device) 
                gaussian_data = mu + std * rand_data.unsqueeze(1) #(768, num, nums_rounded)
                # print(gaussian_data.size())
                nums = gaussian_data.size(1)
                nums_rounded = gaussian_data.size(2)
                reshaped_rounded_data = gaussian_data.permute(1, 2, 0).contiguous().view(nums * nums_rounded, mean_matrix.size(1)) # (num* nums_rounded, 768)
                # print(reshaped_rounded_data.size(),data.size())
                data = torch.cat((data, reshaped_rounded_data), dim = 0)
                # target = torch.cat((target, (self.n_cls) * torch.ones(nums * nums_rounded).to(self.device)), dim = 0)
                # print(data.size())
                # filt id like data
                # print(sub_datasets_in_distances)
                data_add = data[data_in.size(0):]   
                # print(data_add.size())
                
                distances_add = self.mahalanobis(data_add, dataset_in_mu, cov).to(self.device).squeeze() #(n,1)
                distance = torch.max(self.mahalanobis(data[:data_in.size(0)], dataset_in_mu, cov).to(self.device))
                k_init = (torch.mean(distances_add) / distance - 1) * 10
                # 找到满足条件的索引
                mask = distances_add > (1 + k_init * self.k.to(self.device)[0]) * distance
                # print(distances_add, distance)
                
                # cleaned_data_add = data_add[mask.to(self.device)]
                cleaned_data_add = data_add
                
                if cleaned_data_add.size(0) > data_in.size(0) // self.n_cls + 2: #ood太多则随机删除
                    delete_num = cleaned_data_add.size(0) - (data_in.size(0) // self.n_cls + 2)
                    indices = torch.randperm(cleaned_data_add.size(0))[:(data_in.size(0) // self.n_cls + 2)].to(self.device)
                    cleaned_data_add_de = cleaned_data_add[indices]
                else: 
                    cleaned_data_add_de = cleaned_data_add
                    
                
                data = torch.cat((data[:data_in.size(0)], cleaned_data_add_de), dim = 0)
                # data = data[:data_in.size(0)]


                target = torch.cat((target, (self.n_cls) * torch.ones(cleaned_data_add_de.size(0)).to(self.device)), dim = 0)
                # target = target
                

            output = self.head(data)
            output = F.normalize(output, dim=1)
            loss1 = F.cross_entropy(output, target.to(torch.long))

            label_matrix = output
            biclas = torch.zeros(label_matrix.size(0), 2)
            biclas[:,-1] = label_matrix[:,-1]
            biclas[:,0] = torch.sum(label_matrix[:,:-1],-1)
            label_biclas = torch.where(
                torch.gt(target, self.n_cls-0.5),                
                torch.ones(data.size()[0]).to(self.device),
                torch.zeros(data.size()[0]).to(self.device),
                )
            loss2 = F.cross_entropy(biclas.to(self.device), label_biclas.to(torch.int64).to(self.device))
            loss3 = torch.sum(sub_datasets_in_distances) / self.n_cls

            loss = (1 - self.gamma) * loss1 + self.gamma * loss2 #+ self.gamma * loss3
            
            # backward
            self.optimizer.zero_grad()
            # print(loss1,loss2)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
    
    def mahalanobis(self, x, support_mean, inv_covmat): #(n,d), (k,d), (k,d,d)
        # 获取输入的张量维度信息
        n = x.size(0)
        d = x.size(1)

        # 将输入张量和支持向量均值移到 GPU 上
        x = x.cuda()
        support_mean = support_mean.cuda()

        maha_dists = []
        for i in range(inv_covmat.size(0)):
            class_inv_cov = inv_covmat[i].detach()
            support_class = support_mean[i].detach()
        
            x_mu = x - support_class.unsqueeze(0).expand(n, d)            
            # 将类别协方差矩阵移到 GPU 上
            class_inv_cov = class_inv_cov.cuda()
            # print(torch.max(torch.abs(class_inv_cov)))

            # 计算 Mahalanobis 距离
            left = torch.matmul(x_mu, class_inv_cov)
            # print(left.size())
            mahal = torch.matmul(left, x_mu.t()).diagonal()
            # print('2', mahal) 
            # print(torch.matmul(left, x_mu.t()))
            maha_dists.append(mahal)

        # 将结果转换为 PyTorch 张量
        return torch.stack(maha_dists).t()
    
    def calculate_covariance_matrix(self, data):
        # 计算数据的均值
        mean = torch.mean(data, dim=0)
        mean = mean.unsqueeze(0).expand(data.size(0), data.size(1))
        # 将数据减去均值，得到去中心化的数据
        centered_data = data - mean

        # 计算协方差矩阵
        covariance_matrix = torch.mm(centered_data.t(), centered_data) / (centered_data.size(0) - 1 + 1e-7)
        # print(covariance_matrix.size())
        return covariance_matrix
