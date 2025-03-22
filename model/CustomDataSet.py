import os
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

class CustomDataSet(Dataset):
    def __init__(self, actions:list, outcomes:list, dim_out:int, context_lenght:int, tokenizer=None):
        """
        Creates a custom dataset

        :param actions: list of actions
        :param outcomes: list of outcomes
        :param context_lenght: the length of the context
        :param tokenizer: tokenizer to encode the actions and outcomes
        """
        # Je ne suis pas sur d'y garder
        assert context_lenght % 2 != 0, "context_lenght must be odd"
        assert len(actions) == len(outcomes), "actions and outcomes must have the same length"
        assert context_lenght <= len(actions) * 2, "context_lenght must be less than or equal to the length of actions * 2"
        assert context_lenght > 0, "context_lenght can't be negative or zero"

        self.actions = actions
        self.outcomes = outcomes
        self.context_lenght = context_lenght
        self.tokenizer = tokenizer
        self.dim_out = dim_out

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
        """
        Get the item at the index idx

        :param idx: index
        :return: x, y

        Example
        --------
        actions = ["a", "b", "c", "d", "e"] \n
        outcomes = ["1", "2", "3", "4", "5"] \n
        context_lenght = 3 \n
        dataset = CustomDataSet(actions, outcomes, context_lenght) \n
        dataset[0] -> (["a", "1", "b"], "2") \n
        dataset[1] -> (["b", "2", "c"], "3") \n
        dataset[2] -> (["c", "3", "d"], "4") \n
        dataset[3] -> (["d", "4", "e"], "5") \n
        """
        x = []
        x, label = self.create_x(idx)
        x = torch.tensor(x, dtype=torch.float)
        label = torch.tensor(label)
        label = F.one_hot(label, num_classes=self.dim_out).to(torch.float)
        return x, label


class CustomDataSetRNN(Dataset):
    def __init__(self, actions:list, outcomes:list, dim_out:int, context_lenght:int, tokenizer=None):
        """
        Creates a custom dataset

        :param actions: list of actions
        :param outcomes: list of outcomes
        :param context_lenght: the length of the context
        :param tokenizer: tokenizer to encode the actions and outcomes
        """
        # Je ne suis pas sur d'y garder
        assert context_lenght % 2 != 0, "context_lenght must be odd"
        assert len(actions) == len(outcomes), "actions and outcomes must have the same length"
        assert context_lenght <= len(actions) * 2, "context_lenght must be less than or equal to the length of actions * 2"
        assert context_lenght > 0, "context_lenght can't be negative or zero"

        self.actions = actions
        self.outcomes = outcomes
        self.context_lenght = context_lenght
        self.tokenizer = tokenizer
        self.dim_out = dim_out

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
        """
        Get the item at the index idx

        :param idx: index
        :return: x, y

        Example
        --------
        actions = ["a", "b", "c", "d", "e"] \n
        outcomes = ["1", "2", "3", "4", "5"] \n
        context_lenght = 3 \n
        dataset = CustomDataSet(actions, outcomes, context_lenght) \n
        dataset[0] -> (["a", "1", "b"], "2") \n
        dataset[1] -> (["b", "2", "c"], "3") \n
        dataset[2] -> (["c", "3", "d"], "4") \n
        dataset[3] -> (["d", "4", "e"], "5") \n
        """
        x = []
        x, label = self.create_x(idx)
        x = torch.tensor(x, dtype=torch.int)
        label = torch.tensor(label)
        return x, label
    
class CustomDataSetTextGen(Dataset):
    def __init__(self, actions:list, outcomes:list, dim_out:int, context_lenght:int, tokenizer=None):
        """
        Creates a custom dataset

        :param actions: list of actions
        :param outcomes: list of outcomes
        :param context_lenght: the length of the context
        :param tokenizer: tokenizer to encode the actions and outcomes
        """
        # Je ne suis pas sur d'y garder
        assert context_lenght % 2 != 0, "context_lenght must be odd"
        assert len(actions) == len(outcomes), "actions and outcomes must have the same length"
        assert context_lenght <= len(actions) * 2, "context_lenght must be less than or equal to the length of actions * 2"
        assert context_lenght > 0, "context_lenght can't be negative or zero"

        self.actions = actions
        self.outcomes = outcomes
        self.context_lenght = context_lenght
        self.tokenizer = tokenizer
        self.dim_out = dim_out

    def create_x(self, idx):
        gap = (self.context_lenght - 1) // 2
        x = []
        for i in range(idx, idx + gap):
            x.append(self.actions[i])
            x.append(self.outcomes[i])
        x.append(self.actions[idx + gap])
        if self.tokenizer is not None:
            x = self.tokenizer.encode(x)
        return x
        
                
    def __len__(self):
        gap = (self.context_lenght + 1) // 2
        return len(self.actions) + 1 - gap

    def __getitem__(self, idx):
        """
        Get the item at the index idx

        :param idx: index
        :return: x
        """
        x = []
        x = self.create_x(idx)
        x = torch.tensor(x, dtype=torch.int)
        return x

