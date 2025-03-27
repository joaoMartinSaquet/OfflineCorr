import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import deque
from torch.utils.data import DataLoader
import os

import shutil

from dataset_handling import *
from model_handling import *


import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
import json
# to remove pandas error
pd.options.mode.chained_assignment = None

# TODO : log, other models ( cgp, LSTM, RNN, GRN ), find why is ther overfitting ! 

def trainging_loops(model, opt, criterion, train_dl, val_dl, epochs, device, log_mod = 10):

    train_loss = []
    val_loss = []
    for epoch in range(epochs):

        running_loss = 0.0
        
        for i, data in enumerate(train_dl, 0):
            x, y = data
            x = x.float().to(device)
            y = y.float().to(device)
            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()

        train_loss.append(running_loss / len(train_dl))

        with torch.no_grad():
            val_running_loss = 0.0
            for i, data in enumerate(val_dl, 0):
                x, y = data
                x = x.float().to(device)
                y = y.float().to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_running_loss += loss.item()
            val_loss.append(val_running_loss / len(val_dl))
        if epoch % log_mod == 0:
            print(f"Epoch {epoch + 1} train loss: {train_loss[-1]} val loss: {val_loss[-1]}")
    return train_loss, val_loss

def train_ann(hyperparameters, device):

    # Assign hyperparameters to variables
    hidden_size = hyperparameters['hidden_size']
    learning_rate = hyperparameters['learning_rate']
    num_epochs = hyperparameters['num_epochs']
    batch_size = hyperparameters['batch_size']
    num_layers = hyperparameters['num_layers']
    model_type = hyperparameters['model']
    sequence_length = hyperparameters['sequence_length']

    # NN take in input the current displacement, and cursor position
    # it output the predicted displacement 
    x, y, _ = read_dataset("/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/P0_C0.csv", "vec")
    
    y = y.to_numpy()
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=False)
    
    train_dataset = FittsDataset(train_x, train_y)
    val_dataset = FittsDataset(val_x, val_y)

    

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()

    model = ANN(input_size=x[0].shape[0], output_size=2, hidden_size=hidden_size, num_layers=num_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate)


    train_loss, val_loss = trainging_loops(model, opt, criterion, train_dl, val_dl, num_epochs, device)

    return model, train_loss, val_loss

def train_lstm(hyperparameters, device):
    pass # not implemented yet



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    experiment_name = "P0_C0"
    log_dir = "results/" + experiment_name + "/"
    os.makedirs(log_dir, exist_ok=True)
    # read hyperparameters
    config = load_config("config/ann_config.yaml")
    hyperparameters = config['hyperparameters']
    print("Hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")

    
    print("Hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")
    print("using device : ",device)
    batch_size = hyperparameters['batch_size']
    model_type = hyperparameters['model']
    log_dir = f"results/{experiment_name}/{model_type}/"

    if model_type == "ANN":
        model, train_loss, val_loss = train_ann(hyperparameters, device)
    else: 
        model, train_loss, val_loss = train_lstm(hyperparameters, device)

    torch.save(model.state_dict(), log_dir + "model.pt")
    loss = {"train_loss": train_loss, "val_loss": val_loss}
    pd.DataFrame(loss).to_csv(log_dir + "loss.csv")
    shutil.copy("config/ann_config.yaml", log_dir + "config.yaml")

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(["train loss", "val loss"])
    plt.title("ANN loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.show()