# MLP model for force predicted
# input: 6D pose
# output: 6D force


import numpy as np
import torch
import torch.nn as nn



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(6, 150),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(150, 200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(200, 150),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(150, 6),
        )

    def forward(self, x):
        y_pred = self.layers(x)
        return y_pred
    

class predict():
    def __init__(self, model_pth="99.pth"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Net()
        self.net.load_state_dict(torch.load(model_pth))
        self.net.eval()
        self.net.to(self.device)

    def run(self, input=np.ones(6)):
        predicted_y = self.net.forward(torch.from_numpy(input).float().to(self.device))
        output = predicted_y.detach().cpu().numpy()
        return output
