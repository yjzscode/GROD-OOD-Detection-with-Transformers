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
        self.stat_smooth = config.trainer.stat_smooth
        self.batch_size = config.dataset.train.batch_size
        self.threshold = config.trainer.warmup_threshold
        
        self.k = self.net.k


    def train_epoch(self, epoch_idx):

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        sub_datasets_in_mu = torch.zeros((self.n_cls, 768)).to(self.device) #(K,f)
        dataset_in_mu = torch.zeros(768).to(self.device) #(f)
        dataset_in_cov = torch.zeros(768, 768).to(self.device)
        sub_datasets_in_cov = torch.zeros((self.n_cls, 768, 768)).to(self.device)
        sub_datasets_in_distances = torch.zeros(self.n_cls).to(self.device)
        
        torch.autograd.detect_anomaly(True)        
        
        #### Warmup: first x step without lda ood, compute mu and cov for each class instead ###
        warmup = int(self.threshold * self.n_cls / self.batch_size)
        data_warmup = None
        print("Warmup...")
        if warmup == 0:
            pass
        else:
            for train_step in tqdm(range(1,
                                        warmup + 1),
                                desc='Epoch {:03d}: '.format(epoch_idx),
                                position=0,
                                leave=True,
                                disable=not comm.is_main_process()):
                with torch.no_grad():
                    batch = next(train_dataiter)
                    data = batch['data'].to(self.device)
                    # print(data.size())
                    target = batch['label'].to(self.device)       
                    
                    data_in, feat_lda, feat_pca = self.net(data, target)              
                    
                    if train_step == 1:
                        data_warmup = data_in
                    else:    
                        data_warmup = torch.cat((data_warmup, data_in), dim=0) 
        
        if warmup == 0:
            pass
        else:        
            dataset_in_mu = torch.mean(data_warmup, dim = 0)
            cov0 = torch.tensor(self.calculate_covariance_matrix(data_warmup).detach() + 1e-4 * torch.eye(dataset_in_mu.size(0)).to(self.device).detach(), dtype = torch.double)
            L = torch.linalg.cholesky(cov0).detach()
            L_inv = torch.linalg.inv(L).detach()
            dataset_in_cov = torch.tensor(torch.mm(L_inv.t(), L_inv).unsqueeze(0).detach(), dtype=torch.float)
            
            sub_datasets_in = [Subset(data_warmup, torch.where(target == i)[0]) for i in range(self.n_cls)]      
                

            for i in range(len(sub_datasets_in)):
                dataloader = DataLoader(sub_datasets_in[i], batch_size=int(self.threshold * self.n_cls), shuffle=False)
                for batch in dataloader:
                    tensor_data_in = batch
                
                    mean =  torch.mean(tensor_data_in, dim = 0)
                    cov0 = (self.calculate_covariance_matrix(tensor_data_in)+1e-4 * torch.eye(mean.size(0)).to(self.device)).detach()
                    L = torch.linalg.cholesky(cov0).detach()
                    L_inv = torch.linalg.inv(L).detach()

                    # Solve the inverse of a symmetric positive definite matrix A using the inverse of a lower triangular matrix
                    cov = torch.mm(L_inv.t(), L_inv)

                    sub_datasets_in_cov[i,:,:] = cov.detach()
                    sub_datasets_in_mu[i,:] = mean.detach()      
                    sub_datasets_in_distances[i] = torch.max(self.mahalanobis(tensor_data_in, sub_datasets_in_mu.clone(), sub_datasets_in_cov.clone())[:,i]).detach()                                              
        #### Warmup: first x step without lda ood, compute mu and cov for each class instead ###     
        
        self.net.train()
        for train_step in tqdm(range(warmup + 1,
                                     len(train_dataiter)),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):

            batch = next(train_dataiter)
            data = batch['data'].to(self.device)
            target = batch['label'].to(self.device)       
            
            data_in, feat_lda, feat_pca = self.net(data, target)  
            
            
            data = data_in
            data_in = data_in.detach()
            feat_lda = feat_lda.detach()
            feat_pca = feat_pca.detach()

            # generate rounded ood data
            sub_datasets_in = [Subset(data_in, torch.where(target == i)[0]) for i in range(self.n_cls)]
            sub_datasets_lda = [Subset(feat_lda, torch.where(target == i)[0]) for i in range(self.n_cls)]   
            
            # Count the number of samples in each sub-dataset
            dataset_lengths = torch.tensor([len(subset) for subset in sub_datasets_lda])
            mask = dataset_lengths > 2
            lda_class = len(dataset_lengths[mask])
            
            
            reshaped_rounded_data, dataset_in_mu = self.grod_generate_pca(data_in, feat_lda, feat_pca, train_step, dataset_in_mu)
            
            data = torch.cat((data, reshaped_rounded_data), dim = 0)
            
            if lda_class > 0:
                reshaped_rounded_data, sub_datasets_in_mu, sub_datasets_in_cov, sub_datasets_in_distances = self.grod_generate_lda(feat_lda, sub_datasets_in, sub_datasets_lda, sub_datasets_in_mu, sub_datasets_in_cov, sub_datasets_in_distances, lda_class)
        
                data = torch.cat((data, reshaped_rounded_data), dim = 0)
        
            data = torch.cat((data, reshaped_rounded_data), dim = 0)
            # print(reshaped_rounded_data.size())

                

            data_add = data[data_in.size(0):]   
            # print(data_add.size())
                
            
            distances = self.mahalanobis(data_add, sub_datasets_in_mu, sub_datasets_in_cov).to(self.device) #(n,k)
            
            # Calculate the minimum distance and corresponding category index of each sample point
            min_distances, min_distances_clas = torch.min(distances, dim=1)                       
            # Get the sub-dataset distance corresponding to each sample point
            sub_distances = sub_datasets_in_distances[min_distances_clas.to(self.device)]
            
            ### soft label of outliers ###
            target_add = torch.zeros((data_add.size(0), self.n_cls + 1)).to(self.device) #(n, K+1)
            extend = torch.clamp((sub_datasets_in_distances / (distances + 1e-5)).detach(), -1e5, 1e5)
            # print(extend.max())
            # print(extend.size(), extend)
            target_add[:,:-1] = torch.exp(- (1 - extend))
            # extend_ood = torch.gather(extend, 1, min_distances_clas.unsqueeze(1)).squeeze(1).to(self.device)
            extend_ood, _ = torch.max(extend, dim=1)
            # print(extend_ood)
            assert not torch.isnan(extend).any(), "NaN values found in `extend`"
            assert not torch.isinf(extend).any(), "Inf values found in `extend`"
            assert not torch.isnan(min_distances_clas).any(), "NaN values found in `min_distances_clas`"
            target_add[:,-1] = torch.exp(1 - extend_ood)
            # print(target_add[:,:-1], target_add[:,-1])
            ### soft label of outliers ###
            
            
            
            k_init = (torch.mean(min_distances / sub_distances) - 1) * 10
            
            mask = min_distances > (1 + k_init * self.k.to(self.device)[0]) * sub_distances
            # Use Boolean indexing to remove data points that meet a condition
            cleaned_data_add = data_add[mask.to(self.device)]
            cleaned_target_add = target_add[mask.to(self.device)]
                
            if cleaned_data_add.size(0) > data_in.size(0) // self.n_cls + 2:
                indices = torch.randperm(cleaned_data_add.size(0))[:(data_in.size(0) // self.n_cls + 2)].to(self.device)
                cleaned_data_add_de = cleaned_data_add[indices]
                cleaned_target_add_de = cleaned_target_add[indices]
            else: 
                cleaned_data_add_de = cleaned_data_add
                cleaned_target_add_de = cleaned_target_add
                
            data = torch.cat((data[:data_in.size(0)], cleaned_data_add_de), dim = 0)
            
            target = F.one_hot(target, num_classes=self.n_cls + 1)
            
            target = torch.cat((target, cleaned_target_add_de), dim = 0)
            
                

            output = self.head(data)
            # output = F.normalize(output, dim=1)
            loss1 = F.cross_entropy(output, target)

            label_matrix = output
            biclas = torch.zeros(label_matrix.size(0), 2)
            biclas[:,-1] = label_matrix[:,-1]
            biclas[:,0] = torch.sum(label_matrix[:,:-1],-1)
            
            label_biclas = torch.zeros(target.size(0), 2)
            label_biclas[:,-1] = target[:,-1]
            label_biclas[:,0] = torch.sum(target[:,:-1],-1)
            
            loss2 = F.cross_entropy(biclas.to(self.device), label_biclas.to(self.device))
            print(loss1, loss2, cleaned_target_add_de) 
            loss = (1 - self.gamma) * loss1 + self.gamma * loss2 
            
            # backward
            self.optimizer.zero_grad()
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
    
    def grod_generate_pca(self, data_in, feat_lda, feat_pca, train_step, dataset_in_mu):
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
        
        ### mu and std smoothing ###
        if train_step == 1:
            dataset_in_mu = torch.mean(data_in, dim = 0)
        else:
            dataset_in_mu = (1 - self.stat_smooth) * torch.mean(data_in.detach().clone(), dim = 0) + self.stat_smooth * dataset_in_mu.detach().clone() 
        ### mu and std smoothing ###
        
        # dataset_in_mu = torch.mean(data_in.detach().clone(), dim = 0)
        
        dataset_in_mu =  repeat(dataset_in_mu.squeeze(), "f -> b f", 
                                        f = data_in.size(1), b = feat_lda.size()[1])
        # print(data_rounded_category.size())
        B = pcadata_rounded_category.detach()
        B_1 = pcadata_rounded_category_1.detach()
        # print(A.size())
        pcavector = F.normalize(B.clone() - dataset_in_mu, dim = 1)
        pcavector_1 = F.normalize(B_1.clone() - dataset_in_mu, dim = 1)
        B = torch.add(B, self.alpha * pcavector).detach() #(feat_dim, 768)
        B_1 = torch.add(B_1, self.alpha * pcavector_1).detach() #(feat_dim, 768)
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
        return reshaped_rounded_data, dataset_in_mu
        
    def grod_generate_lda(self, feat_lda, sub_datasets_in, sub_datasets_lda, sub_datasets_in_mu, sub_datasets_in_cov, sub_datasets_in_distances, lda_class):   
        dataset_lengths = torch.tensor([len(subset) for subset in sub_datasets_lda])
        # Get the index of sub-datasets with the largest amount of data
        top_indices = sorted(range(len(dataset_lengths)), key=lambda i: dataset_lengths[i], reverse=True)[:lda_class]

        
        arg_max = torch.zeros((len(sub_datasets_lda), feat_lda.size()[1]))
        arg_min = torch.zeros((len(sub_datasets_lda), feat_lda.size()[1]))
        k = 0
        for i in top_indices:
            k = k + 1
            dataloader = DataLoader(sub_datasets_lda[i], batch_size=64, shuffle=False)
            for batch in dataloader:
                tensor_data_lda = batch
            dataloader = DataLoader(sub_datasets_in[i], batch_size=64, shuffle=False)
            for batch in dataloader:
                tensor_data_in = batch
            arg_max[i] = torch.argmax(tensor_data_lda, dim=0) #feat_dim                   
            arg_min[i] = torch.argmin(tensor_data_lda, dim=0) #feat_dim

            for j in range(feat_lda.size()[1]):
                # print(argmax[i][j].item())
                if k == 1 and j==0:
                    data_rounded_category = tensor_data_in[int(arg_max[i][j].item())].unsqueeze(0)
                    data_rounded_category_1 = tensor_data_in[int(arg_min[i][j].item())].unsqueeze(0)
                else:
                    data_rounded_category = torch.cat((data_rounded_category, tensor_data_in[int(arg_max[i][j].item())].unsqueeze(0)),dim=0)
                    data_rounded_category_1 = torch.cat((data_rounded_category_1, tensor_data_in[int(arg_min[i][j].item())].unsqueeze(0)),dim=0)

            if tensor_data_in.size(0) > 1:
                
                mean =  torch.mean(tensor_data_in, dim = 0)
                cov0 = (self.calculate_covariance_matrix(tensor_data_in)+1e-4 * torch.eye(mean.size(0)).to(self.device)).detach()
                L = torch.linalg.cholesky(cov0).detach()
                L_inv = torch.linalg.inv(L).detach()

                # Solve the inverse of a symmetric positive definite matrix A using the inverse of a lower triangular matrix
                cov = torch.mm(L_inv.t(), L_inv)
                ### mu and std smoothing ###
                if torch.max(torch.abs(sub_datasets_in_mu[i,:]))<1e-7:
                    sub_datasets_in_cov[i,:,:] = cov.detach()
                    sub_datasets_in_mu[i,:] = mean.detach()                        
                    sub_datasets_in_distances[i] = torch.max(self.mahalanobis(tensor_data_in, sub_datasets_in_mu.clone(), sub_datasets_in_cov.clone())[:,i]).detach()                                                                     
                else:
                    sub_datasets_in_cov[i,:,:] = (1 - self.stat_smooth) * cov.detach().clone().to(self.device) + self.stat_smooth * sub_datasets_in_cov[i,:,:].detach().clone()
                    sub_datasets_in_mu[i,:] = (1 - self.stat_smooth) * mean.detach().clone().to(self.device) + self.stat_smooth * sub_datasets_in_mu[i,:].detach().clone()
                    dists = self.mahalanobis(tensor_data_in, sub_datasets_in_mu.clone(), sub_datasets_in_cov.clone())[:,i]
                    dist = torch.max(dists)
                    sub_datasets_in_distances[i] = (1 - self.stat_smooth) * dist.to(self.device).detach().clone() + self.stat_smooth * sub_datasets_in_distances[i].detach().clone()
                ### mu and std smoothing ###
            
            
            sub_datasets_in_mean =  repeat(sub_datasets_in_mu.clone()[i,:], "f -> b f", 
                                        f = tensor_data_in.size(1), b = feat_lda.size()[1])
            
            A = data_rounded_category[-feat_lda.size()[1]:].detach()
            A_1 = data_rounded_category_1[- feat_lda.size()[1]:].detach()
            vector = F.normalize(A.to(self.device) - sub_datasets_in_mean.to(self.device), dim = 1)
            vector_1 = F.normalize(A_1.to(self.device) - sub_datasets_in_mean.to(self.device), dim = 1)
            A = A + self.alpha * vector.detach().to(self.device) #(feat_dim, 768)
            A_1 = A_1 + self.alpha * vector_1.detach().to(self.device) #(feat_dim, 768)
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
            
            return reshaped_rounded_data, sub_datasets_in_mu, sub_datasets_in_cov, sub_datasets_in_distances
    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
    
    def mahalanobis(self, x, support_mean, inv_covmat): #(n,d), (k,d), (k,d,d)
        n = x.size(0)
        d = x.size(1)

        x = x.cuda()
        support_mean = support_mean.cuda()

        maha_dists = []
        for i in range(inv_covmat.size(0)):
            class_inv_cov = inv_covmat[i].detach()
            support_class = support_mean[i].detach()
        
            x_mu = x - support_class.unsqueeze(0).expand(n, d)            
            class_inv_cov = class_inv_cov.cuda()

            # Mahalanobis distances
            left = torch.matmul(x_mu, class_inv_cov)
            # print(x_mu.size(), class_inv_cov.size(), left.size())
            mahal = torch.matmul(left, x_mu.t()).diagonal()
            maha_dists.append(mahal)

        return torch.stack(maha_dists).t()
    
    def calculate_covariance_matrix(self, data):
        mean = torch.mean(data, dim=0)
        mean = mean.unsqueeze(0).expand(data.size(0), data.size(1))
        centered_data = data - mean

        covariance_matrix = torch.mm(centered_data.t(), centered_data) / (centered_data.size(0) - 1 + 1e-7)

        return covariance_matrix
