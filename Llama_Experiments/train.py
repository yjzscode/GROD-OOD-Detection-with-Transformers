import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data_process import IMDBProcessor
from data_process import ClincProcessor
from data_process import TextClassificationDataset
from grod_trainer import GRODTrainer_Soft_Label
from utils import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class GRODNet(nn.Module):
    def __init__(self, backbone, feat_dim, num_classes):
        super(GRODNet, self).__init__()
        
        self.backbone = backbone
        if hasattr(self.backbone, 'score'):
            # remove fc otherwise ddp will
            # report unused params
            self.backbone.score = nn.Identity()

        self.lda = LDA(n_components=feat_dim)
        self.pca = PCA(n_components=feat_dim)

        self.n_cls = num_classes
        self.head1 = nn.Linear(4096, 2 * num_classes)
        self.head = nn.Linear(4096, self.n_cls + 1)
        self.k = nn.Parameter(torch.tensor([0.1], dtype=torch.float32, requires_grad=True))

    def forward(self, x, y, attention): #x:data feature, y:label
        
        hidden_states = self.backbone.model(input_ids=x, attention_mask=attention)[0]
        
        feat = hidden_states[torch.arange(x.size(0), device=hidden_states.device), -1].squeeze()
        
        self.lda.fit(feat, y)
        X_lda = self.lda.transform(feat) #(b, feat_dim)
        
        self.pca.fit(feat)
        X_pca = self.pca.transform(feat)

        return feat, X_lda, X_pca
    
    def intermediate_forward(self, x, attention):
        hidden_states = self.backbone.transformer(input_ids=x, attention_mask=attention)[0]
        
        feat = hidden_states[torch.arange(x.size(0), device=hidden_states.device), -1].squeeze()
        
        output = self.head(feat)
        score = torch.softmax(output, dim=1)
        score0 = output[:,:-1]
        conf = torch.max(score, dim=1)
        pred = torch.argmax(score, dim=1)
        conf0 = torch.max(score0, dim=1)
        pred0 = torch.argmax(score0, dim=1)
        for i in range(pred.size(0)):
            if pred[i] == output.size(1) - 1:

                score0[i] = 0.5 * torch.ones(score0.size(1)).to(x.device)

        return torch.softmax(score0, dim=1)



class LDA(nn.Module):
    def __init__(self, n_components):
        super(LDA, self).__init__()
        self.n_components = n_components

    def fit(self, X, y):
        try:
            n_samples, n_features = X.shape
        except:
            n_features = X.shape[0]
        classes = torch.unique(y)
        n_classes = len(classes)
        
        means = torch.zeros(n_classes, n_features).to(X.device)
        for i, c in enumerate(classes):
            try:
                means[i] = torch.mean(X[y==c], dim=0)
            except:
                X = torch.unsqueeze(X, dim=0)
                means[i] = torch.mean(X[y==c], dim=0)
        
        overall_mean = torch.mean(X, dim=0)
        
        within_class_scatter = torch.zeros(n_features, n_features).to(X.device)
        for i, c in enumerate(classes):
            class_samples = X[y==c]
            deviation = class_samples - means[i]
            within_class_scatter += torch.mm(deviation.t(), deviation)
        
        between_class_scatter = torch.zeros(n_features, n_features).to(X.device)
        for i, c in enumerate(classes):
            n = len(X[y==c])
            mean_diff = (means[i] - overall_mean).unsqueeze(1)
            between_class_scatter += n * torch.mm(mean_diff, mean_diff.t())

        within_class_scatter_double = within_class_scatter.double()
        between_class_scatter_double = between_class_scatter.double()

        # Compute eigenvalues and eigenvectors in FP64
        eigenvalues, eigenvectors = torch.linalg.eigh(
            torch.inverse(within_class_scatter_double @ between_class_scatter_double + 
                        1e-2 * torch.eye((within_class_scatter_double @ between_class_scatter_double).size(0)).to(X.device).double())
        )
        eigenvalues = eigenvalues.float()  
        eigenvectors = eigenvectors.float()
        _, top_indices = torch.topk(eigenvalues, k=self.n_components, largest=True)
        self.components = eigenvectors[:, top_indices]

    def transform(self, X):
        return torch.mm(X, self.components)

class PCA(nn.Module):
    def __init__(self, n_components):
        super(PCA, self).__init__()
        self.n_components = n_components

    def fit(self, X):
        try:
            n_samples, n_features = X.shape
        except:
            n_samples = 1
        
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean
        
        covariance_matrix = torch.mm(X_centered.t(), X_centered) / max((n_samples - 1),1)
        
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
        _, top_indices = torch.topk(eigenvalues, k=self.n_components, largest=True)
        self.components = eigenvectors[:, top_indices]

    def transform(self, X):
        X_centered = X - self.mean
        return torch.mm(X_centered, self.components)
    

yaml_file_path = 'paras.yml'
loaded_parameters = load_parameters_from_yaml(yaml_file_path)

config = loaded_parameters

if config['dataset'] == 'IMDB':
    OOD_DataProcessor = IMDBProcessor
    datasets_dir = "/root/autodl-tmp/ood/nlp/imdb_yelp"
    max_seq_length = 256
    batch_size = config['batch_size']

elif config['dataset'] == 'clinc':
    OOD_DataProcessor = ClincProcessor 
    datasets_dir = "/root/autodl-tmp/ood/nlp/clinc150"
    max_seq_length = 128
    batch_size = config['batch_size']

dataset = {}
dataset['train'] = OOD_DataProcessor(True).get_examples(datasets_dir, "train")
# dataset['val'] = OOD_DataProcessor(True).get_examples(datasets_dir, "valid")

dataset['val'] = OOD_DataProcessor(True).get_examples(datasets_dir, "test")

dataset["val_ood"] = OOD_DataProcessor(False).get_examples(datasets_dir, "valid")
dataset['test'] = OOD_DataProcessor(False).get_examples(datasets_dir, "test")


model_name = "/root/autodl-tmp/Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config['K'])
model.config.pad_token_id = model.config.eos_token_id
model.score = nn.Identity()

train_data = [{"tgt_text": example.tgt_text, "label": example.label} for example in dataset['train']]
val_data = [{"tgt_text": example.tgt_text, "label": example.label} for example in dataset['val']]
test_data = [{"tgt_text": example.tgt_text, "label": example.label} for example in dataset['test']]

train_dataloader = create_data_loader(train_data, tokenizer, batch_size=config['batch_size'], max_length=max_seq_length)
val_dataloader = create_test_data_loader(val_data, tokenizer, batch_size=config['batch_size'], max_length=max_seq_length)
test_dataloader = create_test_data_loader(test_data, tokenizer, batch_size=config['batch_size'], max_length=max_seq_length)


yaml_file_path = 'grod copy.yml'
loaded_parameters = load_parameters_from_yaml(yaml_file_path)
config_grod = loaded_parameters
        
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_grod = GRODNet(model, 1, config['K']).to(device)

trainer = GRODTrainer_Soft_Label(model_grod, train_dataloader, config_grod)
trainer.train(config_grod['optimizer']['num_epochs']) 