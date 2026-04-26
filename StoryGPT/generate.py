import torch
import torch.nn.functional as F

def gen_next_token(model, idx, max_pred_token, context_size, tokenizer, temperature=1.0, top_k=None):
    model.eval() # Make sure model is not in training mode (turns off dropout)
    eos_id = tokenizer.token_to_id("<eos>")
    
    with torch.no_grad():
        for _ in range(max_pred_token):
            # Crop to context window
            idx_cond = idx[:, -context_size:]
            
            # forward pass (StoryGPT returns logits directly, no .logits needed!)
            logits = model(idx_cond)     
            
            # extract last position
            logits = logits[:, -1, :]
            
            # temperature scaling (flattens or sharpens the distribution)
            if temperature != 1.0:
                logits = logits / temperature
                
            # top-k filtering (zeros out the tail of garbage predictions)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # multinomial sampling (randomly picks based on probability weights)
            idx_next = torch.multinomial(probs, num_samples=1) 
            
            # append
            idx = torch.cat((idx, idx_next), dim=1)
            
            # stop automatically if it generated the <eos> token!
            if idx_next.item() == eos_id:
                break
                
    return idx

def text_to_token_ids(text, tokenizer):
    # The tokenizers library returns an Encoding. We grab .ids and add <bos>
    bos_id = tokenizer.token_to_id("<bos>")
    encoded_ids = [bos_id] + tokenizer.encode(text).ids
    encoded_tensor = torch.tensor(encoded_ids).unsqueeze(0) # add batch dim
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0).tolist() # remove batch dim
    return tokenizer.decode(flat)


def generate(model, tokenizer, text, max_pred_token=200, temperature=1.0, top_k=None):
    from config import MODEL_CONFIG
    device = next(model.parameters()).device
    input_ids = text_to_token_ids(text, tokenizer).to(device)
    
    output_ids = gen_next_token(
        model=model, 
        idx=input_ids, 
        max_pred_token=max_pred_token, 
        context_size=MODEL_CONFIG["context_length"], 
        tokenizer=tokenizer, 
        temperature=temperature, 
        top_k=top_k
    )
    return token_ids_to_text(output_ids, tokenizer)

if __name__ == "__main__":
    from model.gpt import GPT
    from config import MODEL_CONFIG
    from tokenizers import Tokenizer
    import os

    # Automatically find the correct tokenizer 
    tok_paths = [
        "StoryGPT/tokenizer/storygpt_tokenizer/storygpt_tokenizer.json", 
        "tokenizer/storygpt_tokenizer/storygpt_tokenizer.json",
        "storygpt_tokenizer.json"
    ]
    # We enforce os.path.getsize > 5000 so it completely boycotts the 3.5 KB broken tokenizer file!
    tok_path = next((p for p in tok_paths if os.path.exists(p) and os.path.getsize(p) > 5000), tok_paths[0])
    tokenizer = Tokenizer.from_file(tok_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(MODEL_CONFIG)
    
    # Load model weights from notebooks folder (local)
    weight_paths = ["best_model.pt", "StoryGPT/notebooks/best_model.pt", "notebooks/best_model.pt"]
    weight_path = next((p for p in weight_paths if os.path.exists(p)), "best_model.pt")
    weights = torch.load(weight_path, map_location=device)
    if list(weights.keys())[0].startswith("module."):
        weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    
    model.to(device)
    
    # Generate the story
    prompt = "Once upon a time,"
    story = generate(model, tokenizer, prompt, max_pred_token=200, temperature=0.8, top_k=40)
    
    print("OUTPUT STORY:\n")
    print(story)
