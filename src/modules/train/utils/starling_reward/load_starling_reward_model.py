# from GPTRewardModel import GPTRewardModel
import os
import torch
from huggingface_hub import snapshot_download
import math

from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


## Define the reward model function class
class GPTRewardModel(nn.Module):
    """https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha"""
    def __init__(self, model_path):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.model = model
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        return scores


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)
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