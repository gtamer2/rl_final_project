from datasets import load_dataset
from transformers import AutoTokenizer
import copy

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    batch["input_ids"] = tokenizer(batch["query"], padding=True, truncation=True, return_tensors="pt").input_ids
    return batch

def getDataset(dataset_size=None, batch_size=32):
    
    if dataset_size is None:
        split = "train"
    else:
        split = "train[:" + str(dataset_size) + "]"
        
    dataset = load_dataset("argilla/OpenHermesPreferences", split=split)
    response_set = copy.deepcopy(dataset)
    
    cols = list(dataset[0].keys())
    
    dataset = dataset.rename_column("prompt", "query")
    dataset = dataset.remove_columns([k for k in cols if k!="query" and k!='prompt' and k!='chosen'])
    dataset = dataset.map(tokenize, batched=True, batch_size=batch_size)
    
    return dataset

# Usage
# dataset, response_set = getDataset(10)