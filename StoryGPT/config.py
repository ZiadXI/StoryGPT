
MODEL_CONFIG = {
    "vocab_size": 16384,        # Custom BPE tokenizer vocab (small but enough for stories)
    "context_length": 512,      # Max sequence length
    "emb_dim": 512,             # Embedding dimension (d_model)
    "n_heads": 8,               # Number of attention heads
    "n_kv_heads": 4,            # KV heads for GQA (Grouped Query Attention)
    "n_layers": 8,              # Number of transformer blocks
    "ffn_hidden": 1376,         # FFN hidden dim (≈ 2.68× emb_dim, standard SwiGLU ratio)
    "dropout": 0.0,             # No dropout (modern LLMs don't use it during pretraining)
    "norm_eps": 1e-6,           # RMSNorm epsilon
}

TRAIN_CONFIG = {
    "batch_size": 64,           # Per-GPU batch size
    "learning_rate": 3e-4,      # Peak LR
    "min_lr": 3e-5,             # Min LR for cosine schedule
    "warmup_steps": 500,        # Linear warmup steps
    "max_steps": 20000,         # Total training steps (~1-2 epochs)
    "weight_decay": 0.1,        # AdamW weight decay
    "grad_clip": 1.0,           # Gradient clipping
    "eval_interval": 500,       # Evaluate every N steps
    "save_interval": 2000,      # Checkpoint every N steps
}
