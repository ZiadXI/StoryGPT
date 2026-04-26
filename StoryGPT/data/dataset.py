import torch
from torch.utils.data import Dataset,DataLoader

class TinyStoriesDataset(Dataset):
    def __init__ (self,dataset,tokenizer,stride=512,max_length=512):
     
     bos_id = tokenizer.token_to_id("<bos>")   
     eos_id = tokenizer.token_to_id("<eos>") 

     self.encoded_text = []

     for item in dataset: #Flatten and add bos eos
            tokens = tokenizer.encode(item["text"]).ids
            self.encoded_text.extend([bos_id] + tokens + [eos_id])


     self.inputs = []
     self.targets = []
     
     for i in range(0,len(self.encoded_text)-max_length,stride):
        self.inputs.append(torch.tensor(self.encoded_text[i:i+max_length])) 
        self.targets.append(torch.tensor(self.encoded_text[i+1:i+max_length+1]))  # used append to insure Batch dim as += removes batch dim

    def __len__ (self):
        return len(self.inputs)  # this answers: How many samples do I have? and returning len(targets) would be the same
    def __getitem__ (self,idx):
        x = self.inputs[idx],self.targets[idx]
        return x


def StoryDataLoader(dataset,tokenizer,batch_size=64,max_length=512,stride=512,shuffle=True,drop_last=True,num_workers=2,pin_memo=True):
            
    data =  TinyStoriesDataset(dataset,tokenizer,stride,max_length)
            
    dataloader = DataLoader(data,    
                    batch_size = batch_size,
                    shuffle=shuffle,
                    drop_last = drop_last,
                    num_workers=num_workers,
                    pin_memory=pin_memo)
    return dataloader