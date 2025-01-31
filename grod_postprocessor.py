from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm

from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BasePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, attention):
        output = net(input_ids=data, attention_mask=attention)[0]
        score = torch.softmax(net.head(output), dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader, 
                  alpha, w, b, u, NS,
                  progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch['label'].cuda()
            pred, conf = self.postprocess(net.cuda(), input_ids, attention_mask, alpha, w, b, u, NS)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(labels.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list


class GRODPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config['postprocessor']['postprocessor_args']
        self.args_dict = self.config['postprocessor']['postprocessor_sweep']
        self.dim = self.args['dim']
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict):
        if not self.setup_flag:
            net.eval()
            net.cuda()
            with torch.no_grad():
                self.w, self.b = net.head.weight[:-1,:].cpu().numpy(), net.head.bias[:-1].cpu().numpy()
                # print(self.w.size())
                print('Extracting id training feature')
                feature_id_train = []
                logit_id_train = []
                for batch in tqdm(id_loader_dict,
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['input_ids'].cuda()
                    attention_mask = batch["attention_mask"].cuda()
                    labels = batch['label'].cuda()
                    
                    hidden_states = net.backbone.model(input_ids=data, attention_mask=attention_mask)[0]

                    feature = hidden_states[torch.arange(data.size(0), device=hidden_states.device), -1].squeeze()
                    logit = net.head(feature)
                    score = torch.softmax(logit, dim=1)
                    score0 = torch.softmax(logit[:,:-1], dim=1)
                    conf, pred = torch.max(score, dim=1)
                    conf0, pred0 = torch.max(score0, dim=1)
                    for i in range(pred.size(0)):
                        if pred[i] == logit.size(1) - 1:
                            conf[i] = 0.01
                            pred[i] = 1
                            score0[i, :] = 0.01 * torch.ones(score0.size(1)).cuda()
                        else:
                            conf[i] = conf0[i]     
                        
                    feature_id_train.append(feature.cpu().numpy())
                    logit_id_train.append(score0.cpu().numpy())
                feature_id_train = np.concatenate(feature_id_train, axis=0)
                logit_id_train = np.concatenate(logit_id_train, axis=0)

            self.u = -np.matmul(pinv(self.w), self.b)
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(feature_id_train - self.u)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            self.NS = np.ascontiguousarray(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim:]]).T)

            vlogit_id_train = norm(np.matmul(feature_id_train - self.u,
                                             self.NS),
                                   axis=-1)
            
            print(feature_id_train - self.u, self.NS)
            
            self.alpha = logit_id_train.max(
                axis=-1).mean() / vlogit_id_train.mean()
            print(f'{self.alpha=:.4f}')

            self.setup_flag = True
        else:
            pass
        return self.alpha, self.w, self.b, self.u, self.NS
    
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, attention, alpha, w, b, u, NS):
        hidden_states = net.backbone.model(input_ids=data, attention_mask=attention)[0]      
        feature_ood = hidden_states[torch.arange(data.size(0), device=hidden_states.device), -1].squeeze()
        logit = net.head(feature_ood)
        score = torch.softmax(logit, dim=1)
        score0 = torch.softmax(logit[:,:-1], dim=1)
        conf, pred = torch.max(score, dim=1)
        conf0, pred0 = torch.max(score0, dim=1)
        for i in range(pred.size(0)):
          if pred[i] == logit.size(1) - 1:
            conf[i] = 0.1
            pred[i] = 1
            score0[i, :] = 0.1 * torch.ones(score0.size(1)).cuda()
          else:
            conf[i] = conf0[i]
        logit_ood = score0.cpu()    
        
        feature_ood = feature_ood.cpu()
        
        # logit_ood = feature_ood @ w.T + b
        _, pred = torch.max(logit_ood, dim=1)
        energy_ood = logsumexp(logit_ood.numpy(), axis=-1)
        vlogit_ood = norm(np.matmul(feature_ood.numpy() - u, NS),
                          axis=-1) * alpha
        score_ood = -vlogit_ood + energy_ood
        return pred, torch.from_numpy(score_ood)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim
