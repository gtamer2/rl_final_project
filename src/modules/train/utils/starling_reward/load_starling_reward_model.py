from GPTRewardModel import GPTRewardModel
import os
import torch
from huggingface_hub import snapshot_download
import math

def load_starling_reward_model_for_inference() -> tuple:
    ## Load the model and tokenizer
    reward_model = GPTRewardModel("meta-llama/Llama-2-7b-chat-hf")
    reward_tokenizer = reward_model.tokenizer
    reward_tokenizer.truncation_side = "left"

    directory = snapshot_download("berkeley-nest/Starling-RM-7B-alpha")
    for fpath in os.listdir(directory):
        if fpath.endswith(".pt") or fpath.endswith("model.bin"):
            checkpoint = os.path.join(directory, fpath)
            break
    
    reward_model.load_state_dict(torch.load(checkpoint), strict=False)
    reward_model.eval().requires_grad_(False)
    return reward_model, reward_tokenizer



def get_reward(reward_model, reward_tokenizer, samples, reward_batch_size) -> torch.Tensor:
## Define the reward function
    """samples: List[str]"""
    reward_device = reward_model.get_device()
    input_ids = []
    attention_masks = []
    encodings_dict = reward_tokenizer(
        samples,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors="pt",
    ).to(reward_device)
    input_ids = encodings_dict["input_ids"]
    attention_masks = encodings_dict["attention_mask"]
    mbs = reward_batch_size
    out = []
    for i in range(math.ceil(len(samples) / mbs)):
        rewards = reward_model(input_ids=input_ids[i * mbs : (i + 1) * mbs], attention_mask=attention_masks[i * mbs : (i + 1) * mbs])
        out.extend(rewards)
    return torch.hstack(out)