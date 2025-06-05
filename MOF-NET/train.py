from tqdm import tqdm 
import wandb 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split

from custom_dataset import custom_dataset
from mofnet import mofnet

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



# Create dataset
dataset = custom_dataset("cycle_tot.txt") 

# Split dataset and create dataset
n_data = len(dataset)
n_train = int(n_data * 0.8)
n_validate = int(n_data * 0.1)
n_test = n_data-n_train-n_validate

train_dataset,validation_dataset,test_dataset, = random_split(dataset,[n_train,n_validate,n_test])  
train_dataloader = DataLoader(train_dataset,batch_size=124,shuffle=True) 
validatation_dataloader = DataLoader(validation_dataset,batch_size=124) 
test_dataloader = DataLoader(test_dataset,batch_size=124) 

# Set up training procedure 
model = mofnet()
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

    for input,label in train_dataloader:

        input = input.to(device) 
        label = label.to(device)
               
        #clear gradients
        optimizer.zero_grad()

        #forward pass
        output = model(input)

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


    for input,label in validatation_dataloader:
        
        input = input.to(device)
        label = label.to(device)
                
        #forward pass
        output = model(input)
       
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
for x, y in test_dataloader:

    x = x.to(device)
    y = y.to(device)

    #forward pass
    pred = model(x)

    for i in pred: 
        y_pred.append(i.item())
    for i in y:
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
