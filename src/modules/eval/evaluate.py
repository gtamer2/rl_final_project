from torch.utils.data import DataLoader
from torch import nn

class Evaluator:
    def __init__(self, trained_model: nn.Module, test_dataloader: DataLoader, eval_task: str):
        self.trained_model = trained_model
        self.test_dataloader = test_dataloader
        self.eval = eval_task



    def evaluate(self):
        for batch_idx, batch in enumerate(self.test_dataloader):
