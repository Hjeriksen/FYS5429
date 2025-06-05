from tqdm import tqdm 
import wandb 

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Batch 

from custom_dataset import custom_graph_dataset
from mofnet_GNN import mofnet_GNN
from utils.load_graphs import lookup_graph

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde

from torchmetrics import R2Score


torch.manual_seed(0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

wandb.login()
run = wandb.init(project="MOFnet")


# Creat lookup table of graphs: read in and create all unqie graphs once 
print("Building node graph look up dict..")
graphs_node_dict,missing_n = lookup_graph("N")  # returns dict of node fragment graphs
print("Done!")
print("Building edge graph look up dict..")
graphs_edge_dict,missing_e = lookup_graph("E")  # returns dict of edge fragment graphs
print("Done!")

print("Loading and scaling data..")
file = pd.read_csv("./data/cycle_tot.txt",delimiter=" ",header=None)

# Exclude missing files
for m in missing_n:            
    file = file[~file[0].str.contains(m)]
for m in missing_e:
    file = file[~file[0].str.contains(m)]

# Split and create dataset 
train_df = file.sample(frac=0.8,random_state=42)
leftover = file.drop(train_df.index)
validation_df = leftover.sample(frac=0.5,random_state=42)
test_df = leftover.drop(validation_df.index)

train_dataset = custom_graph_dataset(train_df,graphs_node_dict,graphs_edge_dict) 
train_metrics = train_dataset.get_train_metrics()
validation_dataset = custom_graph_dataset(validation_df,graphs_node_dict,graphs_edge_dict,scaling=train_metrics) 
test_dataset = custom_graph_dataset(test_df,graphs_node_dict,graphs_edge_dict,scaling=train_metrics) 

# Create dataloaders
train_dataloader = DataLoader(train_dataset,batch_size=124,shuffle=True) 
validatation_dataloader = DataLoader(validation_dataset,batch_size=124) 
test_dataloader = DataLoader(test_dataset,batch_size=124) 

print("Done!")

# Set up training procedure 
model = mofnet_GNN()
model.to(device)
optimizer = optim.Adam(model.parameters(),lr=0.001)

loss_func_MSE = nn.MSELoss()
loss_func_MAE = nn.L1Loss()

epochs = 1000
min_validation_loss = np.inf

# Train and validate
for _ in tqdm(range(epochs)):

    #tranining
    model.train()    
    epoch_train_loss = 0  
    epoch_train_loss_MAE = 0  

    for topo,node_graphs,edge_graphs,label in train_dataloader:

        topo = topo.to(device)
        node_graphs = Batch.from_data_list(node_graphs).to(device)
        edge_graphs = Batch.from_data_list(edge_graphs).to(device)
        label = label.to(device)
 
        #clear gradients
        optimizer.zero_grad()

        #forward pass
        output = model([topo, node_graphs, edge_graphs])

        #calculate loss
        loss = loss_func_MSE(output,label.reshape([-1,1]))

        loss_MAE = loss_func_MAE(output,label.reshape([-1,1]))
        
        #calculate gradients and update weights
        loss.backward()
        optimizer.step()

        #collect running loss
        epoch_train_loss += loss.item()*output.shape[0]
        epoch_train_loss_MAE += loss_MAE.item()*output.shape[0]


    #validation
    epoch_validation_loss = 0
    epoch_validation_loss_MAE = 0
    model.eval()

    for topo,node_graphs,edge_graphs,label in validatation_dataloader:

        topo = topo.to(device)
        node_graphs = Batch.from_data_list(node_graphs).to(device)
        edge_graphs = Batch.from_data_list(edge_graphs).to(device)
        label = label.to(device)

        #forward pass
        output = model([topo, node_graphs, edge_graphs])
       
        #calculate loss
        loss = loss_func_MSE(output,label.reshape([-1,1]))
        loss_MAE = loss_func_MAE(output,label.reshape([-1,1]))

        #collect running loss
        epoch_validation_loss += loss.item()*output.shape[0]
        epoch_validation_loss_MAE += loss_MAE.item()*output.shape[0]


    #calculate and log average loss in epoch
    average_epoch_train_loss = epoch_train_loss/len(train_dataset)
    average_epoch_train_loss_MAE = epoch_train_loss_MAE/len(train_dataset)

    average_epoch_val_loss = epoch_validation_loss/len(validation_dataset)
    average_epoch_val_loss_MAE = epoch_validation_loss_MAE/len(validation_dataset)

    wandb.log({"average train loss (MSE)":average_epoch_train_loss,
               "average train loss (MAE)":average_epoch_train_loss_MAE , 
               "average validation loss (MSE)":average_epoch_val_loss,
               "average validation loss (MAE)":average_epoch_val_loss_MAE
               })

    #early stopping - save best model
    if min_validation_loss> epoch_validation_loss:
        min_validation_loss = epoch_validation_loss
        torch.save(model.state_dict(),"model.pt")


# Test
model.load_state_dict(torch.load("model.pt")) #load best model

y_label = []
y_pred = []
for topo,node_graphs,edge_graphs,label in test_dataloader:

    topo = topo.to(device)
    node_graphs = Batch.from_data_list(node_graphs).to(device)
    edge_graphs = Batch.from_data_list(edge_graphs).to(device)
    label = label.to(device)

    #forward pass
    pred = model([topo, node_graphs, edge_graphs])
    
    for i in pred: 
        y_pred.append(i.item())
    for i in label:
        y_label.append(i.item())

y_pred = np.array(y_pred)
y_label = np.array(y_label)

preds = y_label*100
label = y_pred*100


# Calculte R2 score
r2score = R2Score()
R2 = r2score(torch.tensor(preds), torch.tensor(label))
print(f"R^2 score: {R2}")

# Plot parity plot
plt.rcParams['font.size']=24
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 2.5
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 2.5
mpl.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.axisbelow'] = True

fig, ax = plt.subplots(1,1, figsize=(13, 10))

T = np.array([label,preds])
color = gaussian_kde(T)(T)

im = plt.scatter(label,preds, c=color, cmap='Reds',s=2)
ax.set_aspect('equal')

plt.plot([0, 42], [0, 42], c="black", linewidth=3, linestyle='--')

plt.colorbar(im)
plt.yticks([0,10,20,30,40],fontsize=24)
plt.xticks([0,10,20,30,40], fontsize=24)
plt.tick_params(axis='both', labelsize=24)
plt.tick_params(labelsize=24)
plt.xlabel('GCMC Calculated WC (g/L)', fontsize = 30)
plt.ylabel('Machine Learning Predicted WC (g/L)', fontsize = 30)
plt.grid(axis='both',linestyle='-.')
plt.xlim(xmin=0,xmax=42)
plt.ylim(ymin=0,ymax=42)
plt.tight_layout()
plt.savefig("parity_plot.png")
