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


class Trainer:
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
        if config.generated == 'N':
            train_data_func = generate_train_data(config.d0, config.N * 0.8, config.seed_data, config.K)
        elif config.generated == 'Y':   
            train_data_func = generate_test_data(config.d0, config.N * 2.0, config.seed_data, config.K + 2)
        test_data_func = generate_test_data(config.d0, config.N * 0.2, config.seed_data, config.K)
        
        mean_list = []
        std_list = []
        mean_list_g = []
        std_list_g = []
        for i in range(config.K):
            mean = torch.abs(torch.randn(config.d0))   
            std = i / 10 * torch.abs(torch.rand(config.d0)) + 0.1  
            mean_list.append(mean)
            std_list.append(std)
            mean_list_g.append(mean)
            std_list_g.append(std)
        mean = -torch.abs(torch.randn(config.d0)) * 5  
        std = 0.2 * torch.abs(torch.rand(config.d0)) + 0.1  
        mean_list.append(mean)
        std_list.append(std)

        mean_g = torch.abs(torch.randn(config.d0)) * 5  
        std_g = 0.3 * torch.abs(torch.rand(config.d0)) + 0.1  
        mean_list_g.append(mean_g)
        std_list_g.append(std_g)

        mean_g = torch.randn(config.d0) * 4  
        mean_g[0] = torch.abs(mean_g[0])
        mean_g[1] = - torch.abs(mean_g[1])
        std_g = 0.4 * torch.abs(torch.rand(config.d0)) + 0.1  
        mean_list_g.append(mean_g)
        std_list_g.append(std_g)

        mean_g = torch.randn(config.d0) * 3  
        mean_g[0] = -torch.abs(mean_g[0])
        mean_g[1] = torch.abs(mean_g[1])
        std_g = 0.3 * torch.abs(torch.rand(config.d0)) + 0.1  
        mean_list_g.append(mean_g)
        std_list_g.append(std_g)


        
        if config.generated == 'N':
            train_data = train_data_func(mean_list, std_list)[0]
            train_label = train_data_func(mean_list, std_list)[1]
        elif config.generated == 'Y':   
            train_data = train_data_func(mean_list_g, std_list_g)[0]
            train_label = train_data_func(mean_list_g, std_list_g)[1]
            train_label = torch.where(torch.gt(train_label, 2.0), 
                                      2 * torch.ones(train_label.size()), 
                                      train_label).to(torch.int64) 

        test_data = test_data_func(mean_list, std_list)[0]
        test_label = test_data_func(mean_list, std_list)[1]
        
        trainset = myDataset(train_data, train_label)
        testset = myDataset(test_data, test_label) #(b,d0,n=1)

        train_loader = DataLoader(
            dataset=trainset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

        test_loader = DataLoader(
            dataset=testset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.config = config

        ##### Create folders for the outputs.
        postfix = time.strftime("%Y%m%d_%H:%M")
        if hasattr(config, "postfix") and config.postfix != "":
            postfix += "_" + config.postfix

        self.output_path = os.path.join(config.output_path, postfix)

        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "weights"), exist_ok=True)
        self.logging = open(os.path.join(self.output_path, "logging.txt"), "w+")

        OmegaConf.save(config=config, f=os.path.join(self.output_path, "config.yaml"))

        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        model = Transformer( self.config.m_V, 
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
        self.model = model.to(self.device)

        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=config.learning_rate,#5e-4
            weight_decay=config.weight_decay,
        )

        self.scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=config.T_max,
                    eta_min=config.min_learning_rate,
                )
        
        self.epochs = config.epochs
        self.patience = config.patience

    def train(
        self,
    ):
        best_epoch = 0.0
        best_test_acc = 0.0

        time_start = time.time()

        msg = "[{}] Total training epochs : {}".format(
            datetime.now().strftime("%A %H:%M"), self.epochs
        )
        save_log(self.logging, msg)


        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch()

            test_loss, test_acc = self.test_per_epoch(model=self.model,ep=epoch)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.output_path, "weights", "model_epoch{}.pth".format(epoch)
                    ),
                )
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_path, "weights", "best_model.pth"),
                )

            msg = "[{}] Epoch {:03d} \
                \n Train loss: {:.5f},   Train acc: {:.3f}%;\
                \n Test loss: {:.5f},   Test acc: {:.3f}%;  \
                \n Best test acc: {:.3f}%;\
                ".format(                
                datetime.now().strftime("%A %H:%M"),
                epoch,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                best_test_acc,
            )
            save_log(self.logging, msg)

            if (epoch - best_epoch) > self.patience: 
                break
        
        msg = "[{}] Best test acc:{:.3f}% @ epoch {} \n".format(
            datetime.now().strftime("%A %H:%M"), best_test_acc, best_epoch
        )
        save_log(self.logging, msg)

        time_end = time.time()
        msg = "[{}] run time: {:.1f}s, {:.2f}h\n".format(
            datetime.now().strftime("%A %H:%M"),
            time_end - time_start,
            (time_end - time_start) / 3600,
        )
        save_log(self.logging, msg)

    def train_one_epoch(self):
        train_loss_recorder = AverageMeter()
        train_acc_recorder = AverageMeter()

        self.model.train()
        acc_count = 0
        for data, label in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            traindata = data.to(self.device)
#             print(traindata.shape)
            label = label.to(self.device)
            output = self.model(traindata)[0]

            # print(output)
            label_matrix = self.model(traindata)[1]#[LOGITS]
#             print(label_matrix)
            biclas = torch.zeros(label_matrix.size(0), 2)
            biclas[:,-1] = label_matrix[:,-1]
            biclas[:,0] = torch.sum(label_matrix[:,:-1],-1)
            label_biclas = torch.where(
                torch.gt(label,1.5),                
                torch.ones(traindata.size()[0]).to(self.device),
                torch.zeros(traindata.size()[0]).to(self.device),
                )

            loss1 = F.cross_entropy(label_matrix, label) 
            # print(biclas.size(),label_biclas.size())
            loss2 = F.cross_entropy(biclas.to(self.device), label_biclas.to(torch.int64).to(self.device))

            # label_OOD = torch.where(
            #     torch.gt(label,self.config.K),                
            #     torch.ones(traindata.size()[0]).to(self.device),
            #     torch.zeros(traindata.size()[0]).to(self.device),
            #     )
            # output_OOD = torch.where(
            #     torch.gt(output,self.config.K),                
            #     torch.ones(traindata.size()[0]).to(self.device),
            #     torch.zeros(traindata.size()[0]).to(self.device),
            #     )
            
            # loss2 = torch.sum(torch.abs(label_OOD - output_OOD))
            loss = self.config.gamma * loss1 + (1 - self.config.gamma) * loss2

            loss.backward()
            self.optimizer.step()
            
            acc = accuracy(label_matrix, label)[0]  
            # acc_matrix = output - label
            # non_zero = acc_matrix.nonzero().size()
            # acc_count += traindata.size()[0] - non_zero

            train_loss_recorder.update(loss.item(), label_matrix.size(0))
            train_acc_recorder.update(acc.item(), label_matrix.size(0))
            
        self.scheduler.step()
        train_loss = train_loss_recorder.avg
        train_acc = train_acc_recorder.avg
        # train_acc = acc_count / (self.config.N * 0.8)
        return train_loss, train_acc
        
    def test_per_epoch(self, model,ep):
        test_loss_recorder = AverageMeter()
        test_acc_recorder = AverageMeter()
        with torch.no_grad():
            model.eval()
            for data, label in tqdm(self.test_loader):
            
                testdata = data.to(self.device)
                label = label.to(self.device)
                output = self.model(testdata)[0]
                # print(label)
                # print(output)
                # print(label-output)
                label_matrix = self.model(testdata)[1]#[LOGITS]

                loss1 = F.cross_entropy(label_matrix, label) 

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
                loss2 = torch.sum(torch.abs(label_OOD - output_OOD))
                loss = self.config.gamma * loss1 + (1 - self.config.gamma) * loss2

                acc = accuracy(label_matrix, label)[0]  
                test_loss_recorder.update(loss.item(), label_matrix.size(0))
                test_acc_recorder.update(acc.item(), label_matrix.size(0))

            test_loss = test_loss_recorder.avg
            test_acc = test_acc_recorder.avg

            return test_loss, test_acc



