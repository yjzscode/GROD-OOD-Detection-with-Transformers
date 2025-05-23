import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import math

class SelfAttention(nn.Module): #one head
    def __init__(self, m_V, m_h, d, n, batch_size,device=None,requires_grad=True):
        super(SelfAttention, self).__init__()
        self.m_V = m_V
        self.m_h = m_h
        self.d = d
        self.n = n
        self.batch_size = batch_size
        self.device = device
        self.W_O = nn.Parameter(
            torch.tensor(
                torch.randn(int(self.d), int(self.m_V)) * 0.01, dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.W_K = nn.Parameter(
            torch.tensor(
                torch.randn(int(self.m_h), int(self.d)) * 0.01, dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.W_Q = nn.Parameter(
            torch.tensor(
                torch.randn(int(self.m_h), int(self.d)) * 0.01, dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.W_V = nn.Parameter(
            torch.tensor(
                torch.randn(int(self.m_V), int(self.d)) * 0.01, dtype=torch.float32, requires_grad=requires_grad
            )
        )

    def forward(self, x): # shape of x: (b, d, n)
        # perform linear operation and split into h heads

        k = torch.matmul(self.W_K.to(self.device), x)
        q = torch.matmul(self.W_Q.to(self.device), x)
        v = torch.matmul(self.W_V.to(self.device), x)
        
        # calculate attention using function we will define next
        scores = self.attention(q, k, v)
        output = torch.matmul(self.W_O.to(self.device), scores)
        return output
    
    def attention(self, q, k, v):
        scores = torch.matmul(k.transpose(-2, -1), q) / math.sqrt(self.m_h) #/ torch.sqrt(torch.tensor(self.d_head).float())
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(v, scores)
        return output
    

class FFN(nn.Module):
    def __init__(self, d, r,device=None,requires_grad=True,):
        super(FFN, self).__init__()
        self.d = d
        self.r = r
        self.device = device
        self.W_1 = nn.Parameter(
            torch.tensor(
                torch.randn(int(self.r), int(self.d)) * 0.01, dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.W_2 = nn.Parameter(
            torch.tensor(
                torch.randn(int(self.d), int(self.r)) * 0.01, dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.b_1 = nn.Parameter(
            torch.zeros(
                int(self.r), dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.b_2 = nn.Parameter(
            torch.zeros(
                int(self.d), dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.relu = nn.ReLU()
    def forward(self, x): #(N,d,n)
        b_1 = repeat(self.b_1, "r -> N r n", N = x.size()[0], r = self.b_1.size(0), n = x.size()[2])
        act = self.relu(torch.matmul(self.W_1.to(self.device), x) + b_1)
        b_2 = repeat(self.b_2, "d -> N d n", N = x.size()[0], d = self.b_2.size(0), n = x.size()[2])
        output = x + torch.matmul(self.W_2, act) + b_2
        return output

class FCNN_layer(nn.Module): #x: (b, d0, n)->(b, d, n)
    def __init__(self, d0, d, device=None,):
        super(FCNN_layer, self).__init__()
        self.d = d
        self.linear = nn.Linear(d0, d)
        self.relu = nn.ReLU()
    def forward(self, x):
#         print(x.shape)
        x = torch.transpose(self.linear(torch.transpose(x, -1, -2)), -1, -2)
        x = self.relu(x)
        return x


class classifier(nn.Module): 
    def __init__(self, d, K, mode_c, mode_E, lambda0, T, batch_size, n, device=None, epsilon=1e-7, ):
        super(classifier, self).__init__()
        self.epsilon = epsilon
        self.d = d
        self.K = K
        self.T = T
        self.n = n
        self.batch_size = batch_size
        self.mode_c = mode_c
        self.mode_E = mode_E
        self.linear = nn.Linear(self.d, 1)
        self.linear1 = nn.Linear(int(self.n),int(K)+1)
        self.lambda0 = lambda0
        self.device = device
        
    def scoring_function(self, x, mode_E, T):#(b, K+1)
        
        if mode_E == 'e': 
            sum = torch.sum(torch.exp(x), dim = 1)
            sum = repeat(sum, "b -> b d", b = x.size()[0], d = x.size()[1])
            x = torch.exp(x) / sum
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
    
    def forward(self, x): #x: (b, d0, n)->b
        x = self.linear(torch.transpose(x, -1, -2)) #(b,n,1)
        x = torch.squeeze(x,2) #(b,n)
        x = self.linear1(x) #(b,K+1)
        if self.mode_c == 'max':
            output = torch.argmax(x, dim = 1)  #1, ..., K+1 b   
            return output, x
        elif self.mode_c == 'score': #(d, n)
            score0 = self.scoring_function(x, self.mode_E, self.T) #b
            score = repeat(score0, "b-> b d", b = x.size(0), d = x.size(1)) 
            OOD = torch.zeros(x.size())
            OOD[:,-1] = torch.ones(x.size(0))
            x[:,-1] = torch.zeros(x.size(0))
            biclas_matrix = torch.where(
                torch.lt(score, self.lambda0),         
                OOD.to(self.device), 
                x
                )
            max_value = torch.max(biclas_matrix[:,:-1], dim = 1)[0] # b
            # print(max_value)
            hat_c = torch.argmax(biclas_matrix[:,:-1], dim = 1) 
            # print(hat_c.size())
            # print(torch.ones(x.size(0)).size())
            output = torch.where(
                torch.lt(max_value, self.epsilon),
                (self.K ) * torch.ones(x.size(0)).to(self.device),
                torch.tensor(hat_c)
                ) #b 
            return output, biclas_matrix   

        else:
            print('no such kind of classifier')
        


###########################################################
# SelfAttention(self, m_V, m_h, d, n, batch_size):
# FFN(self, d, r):     
# FCNN_layer(self, d0, d):
# classifier(self, d, K, mode_c, mode_E, lambda0, T, batch_size):
   
class TransformerBlock(nn.Module):  
    def __init__(self, m_V, m_h, d, n, r, h, batch_size,device=None,):
        super(TransformerBlock, self).__init__()
        self.h = h
        self.layers = []
        for _ in range(h):
            self.layers.append(SelfAttention(m_V, m_h, d, n, batch_size,device))
        self.FFN = FFN(d, r,device)
        
    def forward(self, x):
        attn_output = x
        for i in range(self.h):
            attn_output = attn_output + self.layers[i](x)
        output = self.FFN(attn_output)
        return output

# (b, d0, n)->b
class Transformer(nn.Module):  
    def __init__(self, 
                 m_V, 
                 m_h, 
                 d, 
                 n, 
                 r, 
                 h, 
                 d0, 
                 K, 
                 mode_c, 
                 mode_E, 
                 lambda0, 
                 T, 
                 l,
                 batch_size,
                 device=None,
                 ):
        super(Transformer, self).__init__()
        layers = []
        for _ in range(l):
            layers.append(TransformerBlock(m_V, m_h, d, n, r, h, batch_size,device))
        self.layers = nn.Sequential(*layers)
        self.fcnn = FCNN_layer(d0, d,device)
        self.classifier = classifier(d, K, mode_c, mode_E, lambda0, T, batch_size,n,device)
    def forward(self, x):
        x = self.fcnn(x)
        x = self.layers(x)
        output, label_matrix = self.classifier(x)    
        return output, label_matrix 
