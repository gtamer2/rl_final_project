from datasets import load_dataset
from transformers import AutoTokenizer
import copy

def load_tokenizer(model_name: str):
    # HF automatically pulls the correct tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize(batch):
    tokenizer = load_tokenizer("google-t5/t5-small")
    batch["input_ids"] = tokenizer(
                            batch["query"], 
                            padding="max_length", 
                            truncation=True, 
                            return_tensors="pt",
                            max_length=32
                        ).input_ids
    
    return batch

def getDataset(dataset_size=-1, batch_size=32):    
    if dataset_size == -1:
        split = "train"
    else:
        split = "train[:" + str(dataset_size) + "]"
        
    dataset = load_dataset("argilla/OpenHermesPreferences", split=split)
    
    cols = list(dataset[0].keys())
    
    dataset = dataset.rename_column("prompt", "query")
    dataset = dataset.remove_columns([k for k in cols if k!="query" and k!='prompt' and k!='chosen'])
    dataset = dataset.map(tokenize, batched=True, batch_size=batch_size)
    
    return dataset

# Usage
# dataset, response_set = getDataset(10)