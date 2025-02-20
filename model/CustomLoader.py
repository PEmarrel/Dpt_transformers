import os
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

class CustomLoader(Dataset):
    def __init__(self, actions:list, outcomes:list, context_lenght:int, tokenizer=None):
        """
        Creates a custom dataset 
        """
        assert context_lenght % 2 != 0, "context_lenght must be odd"
        assert len(actions) == len(outcomes), "actions and outcomes must have the same length"
        self.actions = actions
        self.outcomes = outcomes
        self.unique_outcomes = len(set(outcomes))
        self.context_lenght = context_lenght
        self.tokenizer = tokenizer

    # def _make_data_set_(self, actions:list, outcomes:list):
    #     x, y = [], []
    #     gap = (self.context_lenght + 1) / 2
    #     for i in range(gap, len(actions), 1):
    #         actions_outcomes = zip(actions[i - gap:i - 1], outcomes[i - gap:i - 1])
    #         if self.tokenizer is not None:
    #             x.append(self.tokenizer.encode(actions_outcomes + [actions[i]]))
    #             y.append(self.tokenizer.encode(outcomes[i]))
    #         else:
    #             x.append(actions_outcomes + [actions[i]])
    #             y.append(outcomes[i])
    #     return x, y

    def create_x(self, idx):
        gap = (self.context_lenght - 1) // 2
        x = []
        for i in range(idx, idx + gap):
            x.append(self.actions[i])
            x.append(self.outcomes[i])
        x.append(self.actions[idx + gap])
        y = self.outcomes[idx + gap]
        if self.tokenizer is not None:
            x = self.tokenizer.encode(x)
            y = self.tokenizer.encode(y)
        return x, y
        
                
    def __len__(self):
        gap = (self.context_lenght + 1) // 2
        return len(self.actions) + 1 - gap

    def __getitem__(self, idx):
        x = []
        x, label = self.create_x(idx)
        x = torch.tensor(x)
        label = torch.tensor(label)
        label = F.one_hot(label, num_classes=self.unique_outcomes).to(torch.float)
        return x, label
