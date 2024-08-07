import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Transformer
from dataset import generate_train_data, generate_test_data, myDataset
from einops import repeat
from tqdm import tqdm
import random
from datetime import datetime
import time
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torch.optim import AdamW
from torch.optim import lr_scheduler
import pandas as pd
import torchvision
from utils import (
    AverageMeter,
    accuracy,
    save_log,
    LOGITS,
)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")


class Test:
    def __init__(
        self,
        config_path: str,
    ):
        config = OmegaConf.load(config_path)

        if hasattr(config, "seed"):
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)

        ##### Create Dataloaders.
        test_data_func = generate_test_data(config.d0, config.N * 0.2, config.seed_data, config.K)
        
        mean_list = []
        std_list = []
        for i in range(config.K):
            mean = torch.abs(torch.randn(config.d0))   
            std = i / 10 * torch.abs(torch.rand(config.d0)) + 0.1  
            mean_list.append(mean)
            std_list.append(std)
        mean = -torch.abs(torch.randn(config.d0)) * 5  
        std = 0.2 * torch.abs(torch.rand(config.d0)) + 0.1  
        mean_list.append(mean)
        std_list.append(std)


        test_data = test_data_func(mean_list, std_list)[0]
        test_label = test_data_func(mean_list, std_list)[1]
        OOD_NUM = 0
        for i in range(test_label.size(0)):
            if test_label[i]==2:
                OOD_NUM += 1 
        self.OOD_NUM = OOD_NUM

        testset = myDataset(test_data, test_label) #(b,d0,n=1)

        test_loader = DataLoader(
            dataset=testset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        self.test_loader = test_loader
        
        self.config = config

        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        para = torch.load('/outputs/w_15_seed_44/20240322_10:37_d0_2_w_15_0.1_y_rouned/weights/best_model.pth')
        
        model=model = Transformer( self.config.m_V, 
                 self.config.m_h, 
                 self.config.d, 
                 self.config.n, 
                 self.config.r, 
                 self.config.h, 
                 self.config.d0, 
                 self.config.K, 
                 self.config.mode_c, 
                 self.config.mode_E, 
                 self.config.lambda0, 
                 self.config.T, 
                 self.config.l,
                 self.config.batch_size,
                 self.device
                 )
        model.load_state_dict(para)
        self.model = model.to(self.device)
    def scoring_function(self, x, mode_E='e', T=1):#(b, K+1)
        if mode_E == 'e': 
            sum = torch.sum(torch.exp(x), dim = 1)
            sum = repeat(sum, "b -> b d", b = x.size()[0], d = x.size()[1])
            x = torch.exp(x) / (sum+1e-7)
            output = torch.max(x[:,:-1], dim = 1)[0] #b
            # print(output)
        elif mode_E == 'e_T':
            sum = torch.sum(torch.exp(x / T), dim = 1)
            sum = repeat(sum, "b -> b d", b = x.size()[0], d = x.size()[1])
            x = torch.exp(x / T) / sum
            output = torch.max(x[:,:-1], dim = 1)[0] #b
        elif mode_E == 'log_T':
            output = T * torch.log(torch.sum(torch.exp(x[:,:-1] / T), dim = 1))
            print(output)
        else:
            print('no such kind of scoring function')
        return output #b               
    def test_per_epoch(self):
        test_loss_recorder = AverageMeter()
        test_acc_recorder = AverageMeter()
        with torch.no_grad():
            self.model.eval()
            OOD_E = 0
            a,b,c = 0,0,0
            d,e,f = 0,0,0
            for data, label in tqdm(self.test_loader):
            
                testdata = data.to(self.device)
                label = label.to(self.device)
                output = self.model(testdata)[0]
                
                # print(label)
                # print(output)
                # print(label-output)
                label_matrix = self.model(testdata)[1]#[LOGITS]
                score0 = self.scoring_function(label_matrix)
                for i in range(score0.size()[0]):
                    
                    if label[i] == 0:
                        a+=score0[i]
                        d+=1
                    elif label[i] == 1:
                        b+=score0[i]
                        e+=1
                    elif label[i] == 2:
                        c+=score0[i]
                        f+=1
                
                score = repeat(score0, "b-> b d", b = label_matrix.size(0), d = label_matrix.size(1)) 
                OOD = torch.zeros(label_matrix.size())
                OOD[:,-1] = torch.ones(label_matrix.size(0))
                label_matrix[:,-1] = torch.zeros(label_matrix.size(0))
                biclas_matrix = torch.where(
                    torch.lt(score, 0.1),         
                    OOD.to(self.device), 
                    label_matrix
                    )
                max_value = torch.max(biclas_matrix[:,:-1], dim = 1)[0] # b
                hat_c = torch.argmax(biclas_matrix[:,:-1], dim = 1) 
                output = torch.where(
                    torch.lt(max_value, 1e-7),
                    (self.config.K ) * torch.ones(label_matrix.size(0)).to(self.device),
                    torch.tensor(hat_c)
                    ) #b 
                for i in range(label.size()[0]):
                    if label[i]==2 and output[i] != 2:
                        OOD_E += 1
                label_matrix = biclas_matrix
                loss1 = F.cross_entropy(label_matrix, label) # cross-entropy loss

                label_OOD = torch.where(
                    torch.gt(label,self.config.K),                
                    torch.ones(testdata.size()[0]).to(self.device),
                    torch.zeros(testdata.size()[0]).to(self.device),
                    )
                output_OOD = torch.where(
                    torch.gt(output,self.config.K),                
                    torch.ones(testdata.size()[0]).to(self.device),
                    torch.zeros(testdata.size()[0]).to(self.device),
                    )
                loss2 = torch.sum(torch.abs(label_OOD - output_OOD)) #extra OOD loss
                loss = self.config.gamma * loss1 + (1 - self.config.gamma) * loss2

                acc = accuracy(label_matrix, label)[0]  
                test_loss_recorder.update(loss.item(), label_matrix.size(0))
                test_acc_recorder.update(acc.item(), label_matrix.size(0))
            
            test_loss = test_loss_recorder.avg
            test_acc = test_acc_recorder.avg
            print(a/(d+1e-7), b/(e+1e-7), c/(f+1e-7))
            print((self.OOD_NUM - OOD_E) / self.OOD_NUM)
            print(test_acc)
            return test_loss, test_acc, 




