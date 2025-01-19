from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import openood.utils.comm as comm

### Modified by GROD ###
class VIMPostprocessor: #base version
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, target: Any):
        output = net(data, target)[0] #(b, 768)
        output = net.head(output) #(b, 11)

        score = torch.softmax(output, dim=1)
        score0 = torch.softmax(output[:,:-1], dim=1)
        conf, pred = torch.max(score, dim=1)
        conf0, pred0 = torch.max(score0, dim=1)
        for i in range(pred.size(0)):
            if pred[i] == output.size(1) - 1:
                conf[i] = 0.01
                pred[i] = 1
            else:
                conf[i] = conf0[i]                

        return pred, conf

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            pred, conf = self.postprocess(net, data, label)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list
