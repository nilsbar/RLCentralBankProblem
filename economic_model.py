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
        self.lstm1 = nn.LSTM(input_size=79, hidden_size=15, num_layers=2, batch_first=False)
        # reduce from 80 to 79 and adjust the target data of the train data with dropped interest rate
        self.linear = nn.Linear(15, 78)
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.linear(x)
        return x
    
class ModelLoader():
    
    def __init__(self):
        self.model = EcoModel()
        self.model.load_state_dict(torch.load("model_weights.pth"))
        self.model.eval()
