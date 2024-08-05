# Training of a neural network which models the economy (use of LSTM)
import pandas as pd
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.optim as optim
import torch.utils.data as data

class EcoModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 78
        hidden_dim = 15
        layer_dim = 2
        output_dim = input_dim - 1 

        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=False)
        self.linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.linear(x)
        return x
    
class ModelLoader():
    
    def __init__(self):
        self.model = EcoModel()
        self.model.load_state_dict(torch.load("model_weights.pth"))
        self.model.eval()

if __name__ == "__main__":
    # train the neural network
    pass
