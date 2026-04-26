import torch
import torch.nn as nn

def precompute_rope_freqs(dim, max_seq_len, theta=10000.0):
    """Precompute the complex exponential frequencies for RoPE.
    
    Each pair of dimensions gets a different frequency.
    Lower dimensions = higher frequency = captures local patterns.
    Higher dimensions = lower frequency = captures global patterns.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # freqs shape: (dim // 2,)
    positions = torch.arange(max_seq_len).float()
    # positions shape: (max_seq_len,)
    # Outer product: each position × each frequency
    angles = torch.outer(positions, freqs)
    # angles shape: (max_seq_len, dim // 2)
    # Store as cos and sin (we'll apply rotation using these)
    cos = angles.cos()
    sin = angles.sin()
    return cos, sin  # both (max_seq_len, dim // 2)
def apply_rope(x, cos, sin):
    """Apply rotary embeddings to queries or keys.
    
    x shape: (batch, n_heads, seq_len, head_dim)
    cos, sin shape: (seq_len, head_dim // 2)
    
    The trick: split each head into pairs, rotate each pair.
    """
    batch, n_heads, seq_len, head_dim = x.shape
    # Split into even and odd indices
    x_even = x[..., 0::2]  # (batch, n_heads, seq_len, head_dim // 2)
    x_odd  = x[..., 1::2]
    # Reshape cos/sin for broadcasting: (1, 1, seq_len, head_dim // 2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    # Apply 2D rotation to each pair
    out_even = x_even * cos - x_odd * sin
    out_odd  = x_even * sin + x_odd * cos
    # Interleave back: stack along last dim, then flatten
    out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
    return out





class GroupedQueryAttention(nn.Module):
    def __init__(self,d_in,d_out,num_heads,n_kv_heads,context_length,dropout,):
        super().__init__() 

        assert (d_out % num_heads == 0)
        assert (num_heads % n_kv_heads == 0)

        self.d_out = d_out
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.n_kv_heads = n_kv_heads

        self.dim_head = d_out // num_heads
        self.n_rep = self.num_heads // n_kv_heads
        
        kv_dim = n_kv_heads * (d_out // num_heads)
        self.W_Query=nn.Linear(d_in,d_out,bias=False)   # Q: full size (all 8 heads)
        self.W_Key=nn.Linear(d_in,kv_dim,bias=False)
        self.W_Value=nn.Linear(d_in,kv_dim,bias=False)

        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))

        rope_cos, rope_sin = precompute_rope_freqs(self.dim_head, context_length)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)
 
    def forward(self,x):
        b,num_tokens,d_in = x.shape
        Q = self.W_Query(x)
        K = self.W_Key(x)
        V = self.W_Value(x)

        Q = Q.view(b,num_tokens,self.num_heads,self.dim_head).transpose(1,2)
        K = K.view(b,num_tokens,self.n_kv_heads,self.dim_head).transpose(1,2) # JUST changing the dimentions of data , we didnt change the data itself
        V = V.view(b,num_tokens,self.n_kv_heads,self.dim_head).transpose(1,2)
        
        Q = apply_rope(Q, self.rope_cos, self.rope_sin)
        K = apply_rope(K, self.rope_cos, self.rope_sin)

        K = K.repeat_interleave(self.n_rep, dim=1)  # (b, 4, t, hd) → (b, 8, t, hd)
        V = V.repeat_interleave(self.n_rep, dim=1)

        attention_scores = Q@K.transpose(-2,-1)

        mask = self.mask[:num_tokens, :num_tokens]# 2 be seen         # slice top-left
        mask = mask.unsqueeze(0).unsqueeze(0)#2 be seen                 # broadcast to (1,1,seq_len,seq_len)
        attention_scores = attention_scores.masked_fill(mask.bool(),-torch.inf)
        attention_weights = torch.softmax(attention_scores / self.dim_head**0.5,dim=-1)
        attention_dropped = self.dropout(attention_weights)
        context = attention_dropped @ V #(batch,num_heads,tokens,tokens) @ (batch,num_heads,tokens,head_dim)
        # = (batch,num_heads,tokens,dim_head)
        context = context.transpose(1,2)
        context = context.contiguous().view(b,num_tokens,self.d_out)
        return context



# if __name__ == "__main__":
#     gqa = GroupedQueryAttention(
#         d_in=512,
#         d_out=512,
#         num_heads=8,
#         n_kv_heads=4,
#         context_length=256,
#         dropout=0.0,
#     )
#     print(gqa)

#     x = torch.randn(2, 10, 512)
#     out = gqa(x)
#     print("\nInput shape: ", x.shape)
#     print("Output shape:", out.shape)
#     assert out.shape == x.shape, "Shape mismatch!"
#     print("\nAll checks passed ")
 