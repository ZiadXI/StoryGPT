import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Formula 

RMS(x) = sqrt( mean(x^2) )

x_norm = x / RMS(x)  * gamma

"""
class RMSNorm(nn.Module):
    def __init__ (self,cfg,eps=1e-8):
     super().__init__()
     self.eps = eps
     self.gamma = nn.Parameter(torch.ones(cfg["emb_dim"]))

    def forward(self,x):
     RMS = x.pow(2).mean(dim=-1,keepdim=True).sqrt()
     return (x / (RMS+self.eps)) * self.gamma


""" 
Explaining the idea, We use keepdim for:

x.shape    # (2, 4, 8)
rms.shape  # (2, 4) - dimension is gone

and we use dim=-1 so we select the last dim (8)

so final dim becomes (2,4,1) which is broadcastable and divisible by (2,4,8)

"""