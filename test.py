import argparse
import logging
import pandas as pd
import numpy as np
import torch
import util
from model_1 import EARLIEST
from model_dcrnn_backup import DCRNNModel
from numpy import *
# --- hyperparameters ---
day=15
lr=0.005

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

parser.add_argument('--test_length', type=int, default=day*24*0.1)
parser.add_argument('--batchsize', type=int, default=100)
parser.add_argument('--lam', type=float, default=0.000125)

parser.add_argument('--adj', type=str, default='data/adj/W.csv')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--data', type=str, default='data/person_15')
parser.add_argument('--cell_type', type=str, default='DCRNN')

args = parser.parse_args()

# --- initialize ---
#device
device = torch.device(args.device)
#dataset
dataloader = util.load_dataset(args.data, args.batchsize, args.batchsize, args.batchsize)
#model
adj= pd.read_csv(args.adj,header=None)
adj=adj.astype("float").values
adj_parking=adj
dcrnn=DCRNNModel(adj_parking, device, args.batchsize)
dcrnn.to(device)
#model
model = EARLIEST(ninp=args.n_features, nclasses=args.n_classes, nhid=args.hidden_dimension, 
                rnn_type=args.cell_type, nlayers=args.n_layers, lam=args.lam, dcrnn=dcrnn).to('cuda:0')

model.load_state_dict(torch.load('results/'+str(day)+'/model/lam'+str(args.lam)+'_lr'+str(lr)+'_100.pth'))

# --- testing ---

testing_predictions = []
testing_locations = []
loss_sum = 0
rmse_sum=0
lossc_sum=0
wait_penalty_sum=0
lossr_sum=0
mae_sum=0
# dataloader['test_loader'].shuffle()
i=0
for i, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    i=i+1
    pred, halting_points = model(x, y, test=True)
    halting_points=torch.tensor(halting_points.tolist())
    testing_predictions.append(pred.float())
    
    testing_locations.append(float(halting_points))
    loss,lossc,lossr,mae,wait_penalty = model.computeLoss(pred, y, halting_points)
    loss_sum += loss.item()
    lossr_sum+=lossr.item()
    lossc_sum+=lossc.item()
    wait_penalty_sum+=wait_penalty.item()
    mae_sum+=mae.item()
    if i==0:
        break
# print("LOSS: {}".format(loss_sum/(np.round(args.test_length/args.batchsize))))
# print("MAE: {}".format(mae_sum/(np.round(args.test_length/args.batchsize))))
# print("MAPE: {}".format(MAPE/(np.round(args.test_length/args.batchsize))))
# print("proportion used: {}%".format(100.*mean(testing_locations)))
logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='results/'+str(day)+'/log/result.log',
                    filemode='a',
                    format=
                    '%(asctime)s: %(message)s'
                    )
logging.debug("LR:{},LAM:{},lossc:{}.lossr:{},wait_penalty:{},LOSS:{},MAE:{},proportion used: {}%".format(lr,args.lam,lossc_sum/(1+np.round(args.test_length/args.batchsize)),lossr_sum/(1+np.round(args.test_length/args.batchsize)),
                wait_penalty_sum/(1+np.round(args.test_length/args.batchsize)),loss_sum/(1+np.round(args.test_length/args.batchsize)),mae_sum/(1+np.round(args.test_length/args.batchsize)),100.*mean(testing_locations)))
