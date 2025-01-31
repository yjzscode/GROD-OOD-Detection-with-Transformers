# 评价指标
import numpy as np
import sklearn
from sklearn import metrics


from grod_postprocessor import GRODPostprocessor

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, LoraConfig, get_peft_model

from data_process import IMDBProcessor
from data_process import ClincProcessor
from utils import *
import os
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class GRODNet(nn.Module):
    def __init__(self, backbone, feat_dim, num_classes):
        super(GRODNet, self).__init__()
        
        self.backbone = backbone

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
        # score0 = torch.softmax(output[:,:-1], dim=1)
        conf = torch.max(score, dim=1)
        pred = torch.argmax(score, dim=1)
        conf0 = torch.max(score0, dim=1)
        pred0 = torch.argmax(score0, dim=1)
        for i in range(pred.size(0)):
            if pred[i] == output.size(1) - 1:
                # conf[i] = 0.1
                # pred[i] = 1
                score0[i] = 0.5 * torch.ones(score0.size(1)).to(x.device)
            # else:
                # conf[i] = conf0[i]   
        # return score0
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

        # torch.backends.cuda.preferred_linalg_library('magma')
        # print((torch.inverse(within_class_scatter) @ between_class_scatter).size()) #(4096,4096)
        eigenvalues, eigenvectors = torch.linalg.eigh(
        torch.inverse(within_class_scatter @ between_class_scatter  + 1e-2 * torch.eye((within_class_scatter @ between_class_scatter).size(0)).to(X.device)
        ))
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

def save_metrics_to_csv(results, model_name, filename="metrics.csv"):
    headers = ["Model", "FPR", "AUROC", "AUPR_IN", "AUPR_OUT", "Accuracy"]
    results_with_model = [model_name] + results
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(results_with_model)

    print(f"Metrics saved to {filename}")


class CIDERNet(nn.Module):
    def __init__(self, backbone, head, feat_dim):
        super(CIDERNet, self).__init__()

        self.backbone = backbone
        if hasattr(self.backbone, 'fc'):
            # remove fc otherwise ddp will
            # report unused params
            self.backbone.fc = nn.Identity()
        feature_size = 4096
        # try:
        #     feature_size = backbone.feature_size
        # except AttributeError:
        #     feature_size = backbone.module.feature_size

        if head == 'linear':
            self.head = nn.Linear(feature_size, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(nn.Linear(feature_size, feature_size),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(feature_size, feat_dim))

    def forward(self, x, attention):
        hidden_states = self.backbone.model(input_ids=x, attention_mask=attention)[0]
        
        feat = hidden_states[torch.arange(x.size(0), device=hidden_states.device), -1].squeeze()
        
        unnorm_features = self.head(feat)
        features = F.normalize(unnorm_features, dim=1)
        return features

    def intermediate_forward(self, x, attention):
        hidden_states = self.backbone.model(input_ids=x, attention_mask=attention)[0]
        
        feat = hidden_states[torch.arange(x.size(0), device=hidden_states.device), -1].squeeze()
        return F.normalize(feat, dim=-1)


def compute_all_metrics(conf, label, pred):
    np.set_printoptions(precision=3)
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall)

    accuracy = acc(pred, label)

    results = [fpr, auroc, aupr_in, aupr_out, accuracy]

    return results


# accuracy
def acc(pred, label):
    ind_pred = pred[label != -1] #id acc
    ind_label = label[label != -1]

    # ind_pred = pred #all acc
    # ind_label = label

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc


# fpr_recall
def fpr_recall(conf, label, tpr):
    gt = np.ones_like(label)
    gt[label == -1] = 0

    fpr_list, tpr_list, threshold_list = metrics.roc_curve(gt, conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr)]
    thresh = threshold_list[np.argmax(tpr_list >= tpr)]
    return fpr, thresh


# auc
def auc_and_fpr_recall(conf, label, tpr_th):
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(1 - ood_indicator, conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(ood_indicator, -conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr


# ccr_fpr
def ccr_fpr(conf, fpr, pred, label):
    ind_conf = conf[label != -1]
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    ood_conf = conf[label == -1]

    num_ind = len(ind_conf)
    num_ood = len(ood_conf)

    fp_num = int(np.ceil(fpr * num_ood))
    thresh = np.sort(ood_conf)[-fp_num]
    num_tp = np.sum((ind_conf > thresh) * (ind_pred == ind_label))
    ccr = num_tp / num_ind

    return ccr


def detection(ind_confidences,
              ood_confidences,
              n_iter=100000,
              return_data=False):
    # calculate the minimum detection error
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / n_iter

    best_error = 1.0
    best_delta = None
    all_thresholds = []
    all_errors = []
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        detection_error = (tpr + error2) / 2.0

        if return_data:
            all_thresholds.append(delta)
            all_errors.append(detection_error)

        if detection_error < best_error:
            best_error = np.minimum(best_error, detection_error)
            best_delta = delta

    if return_data:
        return best_error, best_delta, all_errors, all_thresholds
    else:
        return best_error, best_delta
    
# base
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


model_name = "./Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config['K'])
# model.config.pad_token_id = model.config.eos_token_id

base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=config['K']
)
base_model.config.pad_token_id = base_model.config.eos_token_id
base_model.score = nn.Identity()


model_grod = GRODNet(base_model, 1, config['K'])
lora_model_path = "./grod_best"
model_grod = PeftModel.from_pretrained(model_grod, lora_model_path)


custom_layers_path = f"{lora_model_path}/custom_layers.pth"
custom_layers_state_dict = torch.load(custom_layers_path)
model_grod.load_state_dict(custom_layers_state_dict, strict=False)


model_grod.eval()


test_data = [{"tgt_text": example.tgt_text, "label": example.label} for example in dataset['test']]

test_dataloader = create_test_data_loader(test_data, tokenizer, batch_size=config['batch_size'], max_length=max_seq_length)



yaml_file_path = 'grod.yml'
loaded_parameters = load_parameters_from_yaml(yaml_file_path)
config_grod = loaded_parameters
alpha, w, b, u, NS = GRODPostprocessor(config_grod).setup(model_grod, test_dataloader)

pred_list, conf_list, label_list = GRODPostprocessor(config_grod).inference(model_grod, test_dataloader, alpha, w, b, u, NS)
name = lora_model_path.split("/")[-1].split("_best")[0] 
results = compute_all_metrics(conf_list, label_list, pred_list)
save_metrics_to_csv(results, name, filename="metrics_grod.csv")

