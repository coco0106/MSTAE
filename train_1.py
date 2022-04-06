import argparse
import logging
import math
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import tqdm
from sklearn.metrics import mean_squared_error
from torch.serialization import save
import util
from model_1 import EARLIEST
from model_dcrnn_backup import DCRNNModel

from sklearn.preprocessing import MinMaxScaler

# --- hyperparameters ---
day=15


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dimension', type=int, default=31*100)
parser.add_argument('--num_nodes', type=int, default=31)
parser.add_argument('--n_classes', type=int, default=31)
parser.add_argument('--seq_length', type=int, default=12)
parser.add_argument('--nhid', type=int, default=15)
parser.add_argument('--in_dim', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--n_features', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=1)

parser.add_argument('--train_length', type=int, default=int(day*24*0.8))
parser.add_argument('--val_length', type=int, default=int(day*24*0.1))
parser.add_argument('--batchsize', type=int, default=100)
parser.add_argument('--lam', type=float, default=0.00016)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--learning_rate_earliest', type=float, default=0.005)
parser.add_argument('--adj', type=str, default='data/adj/W.csv')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data', type=str, default='data/person_'+str(day))
parser.add_argument('--cell_type', type=str, default='DCRNN')

parser.add_argument('--gcn_bool', action='store_true')
parser.add_argument('--aptonly',action='store_true')
parser.add_argument('--addaptadj',action='store_true')
parser.add_argument('--randomadj',action='store_true')
parser.add_argument('--save',type=str,default='results/'+str(day)+'/')
args = parser.parse_args()
# --- initialize ---
exponentials = util.exponentialDecay(args.epochs)
#device
device = torch.device(args.device)
#dataset
dataloader = util.load_dataset(args.data, args.batchsize, args.batchsize, args.batchsize)
adj= pd.read_csv(args.adj,header=None)
# S=pd.read_csv(args.S,header=None).values
adj=adj.astype("float").values
# adj_parking=torch.mul(torch.tensor(adj),torch.tensor(S))
adj_parking=adj
# scaler = MinMaxScaler( )
# scaler.fit(adj_parking)
# scaler.data_max_
# b=scaler.transform(adj_parking)
# adj_parking=b.reshape(args.num_nodes,args.num_nodes)
dcrnn=DCRNNModel(adj_parking, device, args.batchsize)
dcrnn.to(device)
#model
model = EARLIEST(ninp=args.n_features, nclasses=args.n_classes, nhid=args.hidden_dimension, 
                rnn_type=args.cell_type, nlayers=args.n_layers, lam=args.lam, dcrnn=dcrnn).to('cuda:0')

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate_earliest)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

#log
logging.basicConfig(level=logging.DEBUG,
                    filename=args.save+"log/ear_lr"+str(args.learning_rate_earliest)+'_lam'+str(args.lam)+'.log',
                    filemode='w',
                    format=
                    '%(asctime)s: %(message)s'
                    )

print("start training......")
for epoch in range(args.epochs):
    # --- training ---
    time1=time.time()
    training_loss = []
    training_locations = []
    training_predictions = []
    model._REWARDS = 0
    model._r_sums = np.zeros(args.seq_length).reshape(1, -1)
    model._r_counts = np.zeros(args.seq_length).reshape(1, -1)
    model._epsilon = exponentials[epoch]
    loss_sum = 0
    rmse_sum=0
    mae_sum=0
    lossc_sum=0
    lossr_sum=0
    wait_penalty_sum=0
    dataloader['train_loader'].shuffle()
    for iter, (x, y) in enumerate(tqdm.tqdm(dataloader['train_loader'].get_iterator(),total=math.ceil(np.round(args.train_length/args.batchsize)))):
        # --- Forward pass ---
        
        
        pred, halting_points = model(x, y, epoch)
        halting_points=torch.tensor(halting_points.tolist())
        training_locations.append(halting_points.item())
        training_predictions.append(pred.float())

        # --- Compute gradients and update weights ---
        optimizer.zero_grad()
        loss, lossc,lossr, mae,wait_penalty= model.computeLoss(pred, y, halting_points)
        loss.backward()
        wait_penalty_sum+=wait_penalty.item()
        lossc_sum+=lossc.item()
        lossr_sum+=lossr.item()
        loss_sum += loss.item()
        mae_sum+=mae.item()
        optimizer.step()

    train_pred=(torch.cat(training_predictions,dim=0))[:args.train_length]
    time2=time.time()
    scheduler.step()
    #------Earlystopping------
   
    #--------Output information-----
    logging.debug('Epoch of train:{},Training time:{},MSE:{},lossr:{}, Loss:{},MAE:{},wait_penalty:{},Halting_points:{}'.format(epoch, time2-time1, lossc_sum/np.round(args.train_length/args.batchsize),lossr_sum/np.round(args.train_length/args.batchsize),
                    loss_sum/np.round(args.train_length/args.batchsize), mae_sum/np.round(args.train_length/args.batchsize),
                    wait_penalty_sum/(1+np.round(args.val_length/args.batchsize)),training_locations))
  
    # --- validation ---
    validation_loss = []
    validation_locations = []
    validation_predictions = []
    loss_sum = 0
    rmse_sum=0
    lossc_sum=0
    lossr_sum=0
    mae_sum=0
    wait_penalty_sum=0
    # dataloader['val_loader'].shuffle()
    for iter, (x, y) in enumerate(tqdm.tqdm(dataloader['val_loader'].get_iterator(),total=math.ceil(args.val_length/args.batchsize))):
       # --- Forward pass ---
        pred, halting_points = model(x, y, epoch,test=True)
        halting_points=torch.tensor(halting_points.tolist())
        validation_locations.append(halting_points.item())
        validation_predictions.append(pred.float())

        # --- Compute gradients and update weights ---
        # optimizer.zero_grad()
        loss,lossc,lossr, mae,wait_penalty = model.computeLoss(pred, y, halting_points)
        
        loss.backward()
        lossc_sum+=lossc.item()
        lossr_sum+=lossr.item()
        loss_sum += loss.item()
        wait_penalty_sum+=wait_penalty.item()
        
        mae_sum+=mae.item()
        # optimizer.step()
    # scheduler.step()
    val_pred=(torch.cat(validation_predictions,dim=0))[:args.val_length]
    time3=time.time()
    
    #--------Output information-----
    logging.debug('Epoch of val:{}, Validation Time:{},MSE:{},lossr:{}, Loss:{}, MAE:{}, wait_penalty:{}, Halting_points:{}'.format(epoch, time3-time2, lossc_sum/(1+np.round(args.val_length/args.batchsize)),lossr_sum/(1+np.round(args.val_length/args.batchsize)),
    loss_sum/(1+np.round(args.val_length/args.batchsize)), mae_sum/(1+np.round(args.val_length/args.batchsize)), wait_penalty_sum/(1+np.round(args.val_length/args.batchsize)),validation_locations))
    early_stopping = util.EarlyStopping(loss_sum)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    if epoch==100:
        torch.save(model.state_dict(), args.save+"model/lam"+str(args.lam)+"_lr"+str(args.learning_rate_earliest)+'_'+str(epoch)+'.pth')
        break
  
    
    