import torch
import torch.nn.functional as F
import math
import os


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

def train_model():





 pass

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

def evaluate_modell(model,train_loadeval_loader,device,eval_iter=50):
    
    model.eval()

    with torch.no_grad():
     train_loss = calc_av_loss(train_loader,model,device,num_batches=eval_iter)
     val_loss = calc_av_loss(val_loader,model,device,num_batches=eval_iter)

    model.train() #WARNING
    return train_loss,val_loss