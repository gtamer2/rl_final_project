from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead 
from tqdm import tqdm
import torch
from .utils.dataset import getDataset, load_tokenizer
from .utils.reward import getScores
import numpy as np


def load_ppo_config(model_name, batch_size=32, lr=1.41e-5, seed=0):
    # The PPOConfig dataclass controls all the hyperparameters and settings for the PPO algorithm and trainer.
    # All parameters and their default values can be found here: https://huggingface.co/docs/trl/v0.8.6/en/trainer#trl.PPOConfig
    return PPOConfig(
        task_name="ppo-for-rlhf-on-t5",
        model_name=model_name,
        learning_rate=lr,
        seed=seed,
        batch_size=batch_size,
        mini_batch_size=batch_size//2
    )


def load_model(model_name: str):
    return AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name)


def build_ppo_trainer(model, config, dataset, tokenizer):
    return PPOTrainer(
        model=model,
        config=config,
        dataset=dataset,
        tokenizer=tokenizer,
    )


def perform_rlhf_ppo_training(model_name="google-t5/t5-small",dataset_size=500,
                              epochs=200,batch_size=32,lr=1.41e-5,seed=0,
                              model_save_path="my_ppo_model",
                              rewards_save_path="reward.npy"):
    # Get PPO trainer
    config = load_ppo_config(model_name=model_name, batch_size=32, lr=1.41e-5, seed=0)
    language_model = load_model(model_name=config.model_name)
    tokenizer = load_tokenizer(model_name=config.model_name)
    dataset = getDataset(dataset_size=dataset_size, batch_size=batch_size)
    ppo_trainer = build_ppo_trainer(language_model, config, dataset, tokenizer) 
    
    responseDict = {}
    for i in range(len(dataset)):
        responseDict[str(dataset[i]["query"])] = dataset[i]["chosen"]

    # Train the model
    
    generation_kwargs = {
        "min_length": 3,
        "top_k": 0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128
    }

    final_rewards = []
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}")
        epoch_rewards = []
        for batch in tqdm(ppo_trainer.dataloader):            
            query_tensors = batch["input_ids"]
            query_tensors = torch.stack(query_tensors)            
            query_tensors = torch.transpose(query_tensors, 0, 1)
            query_tensors = torch.split(query_tensors, 1, dim=0)
            query_tensors = [t.squeeze(0) for t in query_tensors]
        
            #### Get response from SFTModel
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            
            batch["response"] = [[tokenizer.decode(r.squeeze())] for r in response_tensors]
            
            batch["chosen"] = [responseDict[k] for k in batch["query"]]
        
            #### Compute reward score
            rewards = getScores(
                inputs= batch["query"],
                candidates= batch["response"],
                ref_candidates=batch["chosen"],
                batch_size=batch_size
            )
            
            epoch_rewards += list([k[0] for k in rewards])
            
            rewards = [torch.tensor(k) for k in rewards]
        
            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            
        avg_epoch_reward = sum(epoch_rewards)/len(epoch_rewards)
        print("Avg Reward:",avg_epoch_reward)
        final_rewards.append(avg_epoch_reward)

        #### Save rewards
        np.save(rewards_save_path, np.array(final_rewards))

    #### Save model
    ppo_trainer.save_pretrained(model_save_path)
    print("Model saved")