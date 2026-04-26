import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(__file__))   # so direct-run finds siblings
from attention import GroupedQueryAttention
from rmsnorm import RMSNorm
from feedforward import SwiGLUFFN


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.norm1 = RMSNorm(cfg)
        self.GQA = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            n_kv_heads=cfg["n_kv_heads"],
            context_length=cfg["context_length"],
            dropout=cfg["dropout"],
        )

        self.norm2 = RMSNorm(cfg)
        self.FF    = SwiGLUFFN(cfg, cfg["ffn_hidden"])

        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        # Attention sub-layer with residual
        x = x + self.drop(self.GQA(self.norm1(x)))

        # FFN sub-layer with residual
        x = x + self.drop(self.FF(self.norm2(x)))

        return x


# if __name__ == "__main__":
#     sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
#     from config import MODEL_CONFIG

#     block = TransformerBlock(MODEL_CONFIG)
#     print(block)

#     x = torch.randn(2, 10, MODEL_CONFIG["emb_dim"])
#     out = block(x)
#     print("\nInput shape: ", x.shape)
#     print("Output shape:", out.shape)
#     assert out.shape == x.shape, "Shape mismatch!"
#     print("\nAll checks passed ")