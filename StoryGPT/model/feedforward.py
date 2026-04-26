import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Instead of one linear projection into GELU,
you do two parallel projections and multiply 
them together

FFN: Input -> expand -> non-linearity -> project back
SwiGLUFFN: It adds capacity + non-linearity to the transformer

"""
class SwiGLUFFN(nn.Module):
    def __init__(self,cfg,hidden_dim):
      super().__init__()
      self.w1 = nn.Linear(cfg["emb_dim"], hidden_dim)  # main projection
      self.w2 = nn.Linear(cfg["emb_dim"], hidden_dim)  # gate projection
      self.w3 = nn.Linear(hidden_dim, cfg["emb_dim"])  # output projection

      
    def forward(self,x):
     gate = F.silu(self.w1(x))   # activated
     x = self.w2(x)              # raw gate signal
     return self.w3(gate * x)