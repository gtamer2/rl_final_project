import torch
import argparse
from src.modules.train.ppo_trainer import perform_rlhf_ppo_training
import numpy as np
import matplotlib.pyplot as plt

def main(model_name,dataset_size,epochs,batch_size,lr,seed,mode,model_save_path,rewards_save_path):

    # if not torch.cuda.is_available():
    #     raise Exception("Training is not supported without GPU")

    if mode == 'train':
        print("Training")
        perform_rlhf_ppo_training(model_name=model_name,dataset_size=dataset_size,epochs=epochs,
                                  batch_size=batch_size,lr=lr,seed=seed,model_save_path=model_save_path,
                                  rewards_save_path=rewards_save_path)
    
    elif mode == 'predict':
        pass
    
    elif mode == 'visualize':
        print("Visualizing data")
        data = np.load(rewards_save_path)
        
        plt.plot(data)
        
        plt.xlabel('Epochs')
        plt.ylabel('Average Reward')
        
        plt.show()

    else:
        raise Exception("Invalid mode, use --mode=\"train\", --mode=\"predict\" or --mode=\"visualize\"")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="LLM - RLAIF Training")
    parser.add_argument('--model_name', default="google-t5/t5-small", type=str)
    parser.add_argument('--dataset_size', default=500, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--lr',default=1.41e-5,type=float)
    parser.add_argument('--seed',default=0,type=int)
    parser.add_argument('--mode', default='train',type=str)
    parser.add_argument('--model_save_path', default='my_ppo_model',type=str)
    parser.add_argument('--rewards_save_path', default='reward.npy',type=str)

    args = parser.parse_args()

    main(model_name=args.model_name,dataset_size=args.dataset_size,epochs=args.epochs,
         batch_size=args.batch_size,lr=args.lr,seed=args.seed,mode=args.mode,
         model_save_path=args.model_save_path,rewards_save_path=args.rewards_save_path)