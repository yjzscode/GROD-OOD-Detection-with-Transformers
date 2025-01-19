import torch
import torch.nn as nn
import torch.nn.functional as F
# from umap_pytorch import PUMAP
from openood.utils import Config

class GRODNet(nn.Module):
    def __init__(self, backbone, feat_dim, num_classes):
        super(GRODNet, self).__init__()

        self.backbone = backbone
        if hasattr(self.backbone, 'fc'):
            # remove fc otherwise ddp will
            # report unused params
            self.backbone.fc = nn.Identity()

        self.lda = LDA(n_components=feat_dim)
        self.pca = PCA(n_components=feat_dim)

        self.n_cls = num_classes
        self.head1 = nn.Linear(self.backbone.hidden_dim, 2 * num_classes)
        self.head = nn.Linear(self.backbone.hidden_dim, self.n_cls + 1)
        self.k = nn.Parameter(torch.tensor([0.1], dtype=torch.float32, requires_grad=True))

    def forward(self, x, y): #x:data feature, y:label
        feat = self.backbone(x)[1]#.squeeze() #(b,768)
        # output = self.backbone(x)[0].squeeze() #(b,10)
        self.lda.fit(feat, y)
        X_lda = self.lda.transform(feat) #(b, feat_dim)
        
        self.pca.fit(feat)
        X_pca = self.pca.transform(feat)

        return feat, X_lda, X_pca
    def intermediate_forward(self, x):
        feat = self.backbone(x)[1]
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
                score0[i] = 0.1 * torch.ones(score0.size(1)).cuda()
            # else:
                # conf[i] = conf0[i]   
        # return score0
        return F.normalize(score0, dim=1)



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
        # print((torch.inverse(within_class_scatter) @ between_class_scatter).size()) #(768,768)
        eigenvalues, eigenvectors = torch.linalg.eigh(
        torch.inverse(within_class_scatter @ between_class_scatter  + 1e-7 * torch.eye((within_class_scatter @ between_class_scatter).size(0)).to(X.device)
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
