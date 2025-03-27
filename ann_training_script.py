import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
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
    hidden_size, learning_rate, num_epochs, batch_size, num_layers, _, _ = load_hyperparameters(hyperparameters)

    x, y, _ = read_dataset("/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/P0_C0.csv", "vec")
    x, y, scaler =     preprocess_dataset(x, y)

    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=False)
    train_dataset = FittsDataset(train_x, train_y)
    val_dataset = FittsDataset(val_x, val_y)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()

    # NN take in input the current displacement, and cursor position
    # it output the predicted displacement 
    model = ANN(input_size=x[0].shape[0], output_size=2, hidden_size=hidden_size, num_layers=num_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss, val_loss = trainging_loops(model, opt, criterion, train_dl, val_dl, num_epochs, device)

    return model, train_loss, val_loss

def train_lstm(exp_name, hyperparameters, device):


    # Assign hyperparameters to variables
    hidden_size, learning_rate, num_epochs, batch_size, num_layers, _, seq_l = load_hyperparameters(hyperparameters)

    x, y, _ = read_dataset(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{exp_name}.csv", "vec")
    x, y, scaler = preprocess_dataset(x, y, 'minmax')

    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=False)
    train_dataset = FittsDatasetSeq(train_x, train_y, sequence_length=seq_l)
    val_dataset = FittsDatasetSeq(val_x, val_y, sequence_length=seq_l)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()

    # NN take in input the current displacement, and cursor position
    # it output the predicted displacement 
    model = LSTMModel(input_size=x[0].shape[0], output_size=2, hidden_size=hidden_size, num_layers=num_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate)


    train_loss, val_loss = trainging_loops(model, opt, criterion, train_dl, val_dl, num_epochs, device)

    return model, train_loss, val_loss



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    experiment_name = "P0_C1"


    print("---------------- {} ----------------".format(experiment_name))
    # read hyperparameters
    config = load_config("config/ann_config.yaml")
    hyperparameters = config['hyperparameters']    
    print("Hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")
    print("using device : ",device)
    batch_size = hyperparameters['batch_size']
    model_type = hyperparameters['model']
    log_dir = f"results/{experiment_name}/{model_type}/"
    print("logging to ", log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    if model_type == "ANN":
        model, train_loss, val_loss = train_ann(experiment_name,hyperparameters, device)
    elif model_type == "LSTM": 
        model, train_loss, val_loss = train_lstm(experiment_name, hyperparameters, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    

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
    plt.savefig(log_dir + "loss.png")
