from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
import torch
from .base_postprocessor import BasePostprocessor

### modified for GROD ###

class VIMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()

            with torch.no_grad():
                # self.w, self.b = net.backbone.get_fc()
                # print(self.b.size)
                self.w, self.b = net.head.weight[:-1,:].cpu().numpy(), net.head.bias[:-1].cpu().numpy()
                # print(self.w.size())
                print('Extracting id training feature')
                feature_id_train = []
                logit_id_train = []
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()
                    _, feature = net.backbone(data, return_feature=True)
                    logit = net.head(feature)
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
                        
                    feature_id_train.append(feature.cpu().numpy())
                    logit_id_train.append(score0.cpu().numpy())
                feature_id_train = np.concatenate(feature_id_train, axis=0)
                logit_id_train = np.concatenate(logit_id_train, axis=0)

                # logit_id_train = feature_id_train @ self.w.T + self.b

            self.u = -np.matmul(pinv(self.w), self.b)
            
            feature_id_train_tensor = torch.tensor(feature_id_train, dtype=torch.float32).cuda()
            u_tensor = torch.tensor(self.u, dtype=torch.float32).cuda()
            
            centered_data_tensor = feature_id_train_tensor - u_tensor
            empirical_covariance = torch.matmul(centered_data_tensor.t(), centered_data_tensor) / centered_data_tensor.size(0)
            
            eig_vals, eigen_vectors = torch.linalg.eigh(empirical_covariance)#, eigenvectors=True)
            eig_vals = eig_vals.cpu().detach().numpy()
            eigen_vectors = eigen_vectors.cpu().detach().numpy()

            
            self.NS = np.ascontiguousarray(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim:]]).T)

            vlogit_id_train = norm(np.matmul(feature_id_train - self.u,
                                             self.NS),
                                   axis=-1)
            self.alpha = logit_id_train.max(
                axis=-1).mean() / vlogit_id_train.mean()
            print(f'{self.alpha=:.4f}')

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature_ood = net.backbone.forward(data, return_feature=True)
        
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
        # logit_ood = feature_ood @ self.w.T + self.b
        _, pred = torch.max(logit_ood, dim=1)
        energy_ood = logsumexp(logit_ood.numpy(), axis=-1)
        vlogit_ood = norm(np.matmul(feature_ood.numpy() - self.u, self.NS),
                          axis=-1) * self.alpha
        score_ood = -vlogit_ood + energy_ood
        return pred, torch.from_numpy(score_ood)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim
