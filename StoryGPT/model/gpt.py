import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(__file__)) 
from transformer import TransformerBlock
from rmsnorm import RMSNorm



class GPT(nn.Module):
     def __init__(self,cfg):
        super().__init__() 
        self.token_embedding = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.dropout=nn.Dropout(cfg["dropout"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
            )
        self.final_norm = RMSNorm(cfg)
        self.output_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.output_head.weight = self.token_embedding.weight #weight tying



     def forward(self,x):
        batch,seq_len = x.shape
        embeds = self.token_embedding(x)
        x = self.dropout(embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x) # output layer that outputs scores that will be softmaxed same as vanilla neural networks

        return logits
    
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import MODEL_CONFIG

    gpt = GPT(MODEL_CONFIG)
    print(gpt)

    x = torch.randint(0, MODEL_CONFIG["vocab_size"], (2, 10))  #  ints, 2D
    out = gpt(x)
    print("\nInput shape: ", x.shape)
    print("Output shape:", out.shape)
    assert out.shape == (2, 10, MODEL_CONFIG["vocab_size"])
    print("\nAll checks passed ")