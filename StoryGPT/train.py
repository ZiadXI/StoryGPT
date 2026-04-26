import torch
import torch.nn.functional as F
import math
import os

from model.gpt import GPT
from config import MODEL_CONFIG, TRAIN_CONFIG

"""
Training the model
"""
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine decay with linear warmup.
    
    - First `warmup_steps`: linearly increase from 0 → max_lr
    - Then cosine decay from max_lr → min_lr
    
    This is the standard LLM training schedule (GPT-3, LLaMA, etc.)
    """
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    
    if step >= max_steps:
        return min_lr
    
    # Cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def train(model, train_loader, val_loader, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),        # Standard for LLM training
    )
    step = 0
    best_val_loss = float("inf")
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(999):  # We stop by step count, not epochs
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Update learning rate
            lr = get_lr(step, cfg["warmup_steps"], cfg["max_steps"],
                       cfg["learning_rate"], cfg["min_lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            # Forward
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    y.view(-1),
                    ignore_index=0,  # Ignore pad token (id=0) in loss
                )
            # Backward
            optimizer.zero_grad()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Gradient clipping (prevents training instabilities)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            
            scaler.step(optimizer)
            scaler.update()
            # Logging
            if step % 100 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f} | LR: {lr:.6f}")
            # Evaluation
            if step % cfg["eval_interval"] == 0 and step > 0:
                train_loss,val_loss = evaluate_model(model, train_loader,val_loader, device)
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "best_model.pt")
                    print("Saved best model!")
                
                model.train()
            # Save checkpoint
            if step % cfg["save_interval"] == 0 and step > 0:
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                }, f"checkpoint_step{step}.pt")
            step += 1
            if step >= cfg["max_steps"]:
                return

def calc_loss_batch(input_batch,target_batch,model,device):
    input_batch,target_batch = input_batch.to(device),target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0,1),target_batch.flatten(),ignore_index=0) #we flatten because cross entropy expect that
    return loss #Ignore index ignores padding for safety
"""
Cross Entropy expects shape of Preds:(N,num_classes)
targets(N,)
Logits shape is (b,seqlen,voc_size)
logits:  (2, 10, 16384)   ← 2 stories, 10 positions, 16384 vocab scores
targets: (2, 10)           ← 2 stories, 10 true token IDs

flatten(0,1) merges first two dims:
logits.flatten(0,1):   (20, 16384)  ← 20 positions total, each with vocab scores
targets.flatten():     (20,)         ← 20 true token IDs
# Now cross_entropy is happy 
so in flatten0,1 its like I concatenate ids logits of story1,2 together to form a 2d of 20,,16--
and same for targets
"""
def calc_av_loss(loader,model,device,num_batches=None):
    total_loss=0.0
    for i,(x,y) in enumerate(loader):
     if num_batches is not None and i >= num_batches:
       break
     total_loss += calc_loss_batch(x,y,model,device).item() # item enhanced ram
    return total_loss / (i+1) # as i is zero indexed

def evaluate_model(model,train_loader,val_loader,device,eval_iter=50):
    
    model.eval()

    with torch.no_grad():
     train_loss = calc_av_loss(train_loader,model,device,num_batches=eval_iter)
     val_loss = calc_av_loss(val_loader,model,device,num_batches=eval_iter)

    model.train() #WARNING
    return train_loss,val_loss

    