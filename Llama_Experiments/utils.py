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
from data_process import TextClassificationDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_parameters_from_yaml(file_path):
    with open(file_path, 'r') as file:
        parameters = yaml.safe_load(file)
    return parameters

def create_data_loader(data: List[Dict[str, str]], tokenizer: AutoTokenizer, batch_size: int = 32, max_length: int = 128):
    # if use ddp, need to create sampler
    dataset = TextClassificationDataset(data, tokenizer, max_length)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    return data_loader

def create_test_data_loader(data: List[Dict[str, str]], tokenizer: AutoTokenizer, batch_size: int = 32, max_length: int = 128):
    dataset = TextClassificationDataset(data, tokenizer, max_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader