from typing import Tuple
from torch import nn
import torch


class MlpClassifier(nn.Module):
    """
    Multi Layer Perceptron Classifier avec des fonctions d'activation ReLU
    """

    def __init__(self, input_size:int, hidden_size:list[int], output_size:int):
        """
        """
        super(MlpClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size[0])

        self.hidden_layers = nn.ModuleList()
        for size in range(len(hidden_size) - 1):
            self.hidden_layers.append(nn.Linear(hidden_size[size], hidden_size[size + 1]))
        self.fc4 = nn.Linear(hidden_size[-1], output_size)
        self._init_weights()


    def _init_weights(self):
        """
        Initialisation des poids des couches
        """
        for layer in self.hidden_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)