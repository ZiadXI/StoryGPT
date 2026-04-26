# StoryGPT — LLaMA-Style Language Model Pre-Trained from Scratch

A 50M parameter decoder-only transformer pre-trained from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.  
Built to demonstrate a complete, production-grade LLM pipeline — from BPE tokenizer training to multi-GPU inference.

---

## Architecture

| Component | Choice | Why |
|---|---|---|
| Attention | Grouped Query Attention (GQA) | Same as LLaMA 2/3 — reduces KV cache memory |
| Position Encoding | Rotary (RoPE) | Relative position awareness without learned embeddings |
| Normalization | RMSNorm | Faster and more stable than LayerNorm |
| Activation | SwiGLU FFN | Higher capacity than ReLU/GELU, used in LLaMA |
| Weight Tying | Input embed = Output head | Reduces parameters, improves training |

**Model Config:**
```
vocab_size    : 16,384   (custom BPE)
context_length: 512
emb_dim       : 512
n_heads       : 8
n_kv_heads    : 4        (GQA ratio 2:1)
n_layers      : 8
ffn_hidden    : 1,376    (≈ 2.68× emb_dim, standard SwiGLU ratio)
Parameters    : ~50M
```

---

## Training

- **Dataset:** TinyStories (150k stories, ~40M tokens)
- **Steps:** 20,000
- **Optimizer:** AdamW (`β=(0.9, 0.95)`, `weight_decay=0.1`)
- **LR Schedule:** Cosine decay with linear warmup (500 steps), peak `3e-4` → min `3e-5`
- **Gradient Clipping:** `1.0`
- **Mixed Precision:** `torch.cuda.amp` (AMP float16)
- **Hardware:** 2× NVIDIA T4 via `nn.DataParallel` on Kaggle

**Final Metrics:**
```
Train Loss : 1.36
Val Loss   : 1.41
Perplexity : 4.09
```

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/StoryGPT
cd StoryGPT
pip install -r requirements.txt
```

Place `best_model.pt` in the project root, then run:

```bash
python generate.py
```

**Sample output:**
> *Once upon a time, there was a little boy named Timmy. Timmy loved to play with his toys and go on adventures...*

---

## Project Structure

```
StoryGPT/
├── model/
│   ├── attention.py      # Grouped Query Attention + RoPE
│   ├── feedforward.py    # SwiGLU FFN
│   ├── rmsnorm.py        # RMS Normalization
│   ├── transformer.py    # TransformerBlock
│   └── gpt.py            # Full GPT model
├── config.py             # Model & training hyperparameters
├── train.py              # Full training loop (AMP, cosine LR, checkpointing)
├── generate.py           # Inference with temperature + top-k sampling
├── notebooks/
│   └── StoryGPT_Kaggle.ipynb   # Self-contained Kaggle training notebook
└── tokenizer/
    └── storygpt_tokenizer/
        └── storygpt_tokenizer.json   # Trained BPE tokenizer (16,384 vocab)
```

---

## Requirements

```
torch>=2.0
datasets
tokenizers
```

---

## Author

Built from scratch demonstrating modern LLM pre-training.
