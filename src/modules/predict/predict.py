from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead 
from tqdm import tqdm
import torch
from ..train.utils.dataset import getDataset, load_tokenizer
from ..train.utils.reward import getScores
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import re
from transformers import BertModel, BertTokenizer
import torch
from scipy.spatial.distance import cosine


nltk.download('punkt')

def compute_bleu(references, candidates):
    scores = []
    for i in range(len(references)):
        # Tokenizing the texts
        reference_tokens = word_tokenize(references[i])
        candidate_tokens = word_tokenize(candidates[i])
        
        # Calculating BLEU score, treating the reference as a list of lists for multiple references
        scores.append(sentence_bleu([reference_tokens], candidate_tokens))
    return sum(scores)/len(scores)


def bert_cosine_similarity(text1, text2, model_name='bert-base-uncased'):
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # Prepare tokens for input
    encoded_input_1 = tokenizer(text1, return_tensors='pt', padding="max_length", max_length=128, truncation=True)
    encoded_input_2 = tokenizer(text2, return_tensors='pt', padding="max_length", max_length=128, truncation=True)
    
    # Get embeddings (mean of last layer hidden states)
    with torch.no_grad():
        output_1 = model(**encoded_input_1)
        output_2 = model(**encoded_input_2)
    
    embeddings_1 = output_1.last_hidden_state.mean(dim=1)
    embeddings_2 = output_2.last_hidden_state.mean(dim=1)
    
    # Compute cosine similarity
    cos_sim = 0
    for i in range(len(embeddings_2)):
        cos_sim += 1 - cosine(embeddings_1[i].squeeze().numpy(), embeddings_2[i].squeeze().numpy())
    
    return cos_sim/len(embeddings_2)


def load_ppo_config(model_name, batch_size=32, lr=1.41e-5, seed=0):
    # The PPOConfig dataclass controls all the hyperparameters and settings for the PPO algorithm and trainer.
    # All parameters and their default values can be found here: https://huggingface.co/docs/trl/v0.8.6/en/trainer#trl.PPOConfig
    return PPOConfig(
        task_name="ppo-for-rlhf",
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


def perform_rlhf_ppo_prediction(model_name="google-t5/t5-small",dataset_size=500,
                              epochs=200,batch_size=32,lr=1.41e-5,seed=0,
                              model_save_path="my_ppo_model",
                              rewards_save_path="reward.npy",
                              predictions_path="predictions.txt",
                              references_path="references.txt"):
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
        
        with open(predictions_path, 'a') as f:
            for k in batch["response"]:
                f.write(k[0]+"\n")
        
        with open(references_path, 'a') as f:
            for k in batch["chosen"]:
                f.write(k[1]["content"].replace("\n"," ").strip()+"\n")
    
        #### Compute reward score
        rewards = getScores(
            inputs= batch["query"],
            candidates= batch["response"],
            ref_candidates=batch["chosen"],
            batch_size=batch_size
        )
        
        epoch_rewards += list([k[0] for k in rewards])
        
    avg_epoch_reward = sum(epoch_rewards)/len(epoch_rewards)
    print("Avg Reward:",avg_epoch_reward)
    
    pred = []
    with open(predictions_path, "r") as f:
        pred = f.readlines()
        pred = [re.sub(r"<[^>]*>", "", k).strip() for k in pred]

    ref = []
    with open(references_path, "r") as f:
        ref = f.readlines()
        
    print("Avg Bleu Score:",compute_bleu(ref, pred))
        
    print("Avg BERT Score:",bert_cosine_similarity(ref, pred))
    