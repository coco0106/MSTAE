import argparse
import os

import numpy as np
from numpy.lib.function_base import select
import pandas as pd
import tensorflow as tf
import torch
from torch import nn

from model_dcrnn_backup import DCRNNModel
import util
from modules_1 import BaselineNetwork, Controller


from sklearn.preprocessing import MinMaxScaler

day=15
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dimension', type=int, default=1)
parser.add_argument('--num_nodes', type=int, default=31)
parser.add_argument('--n_classes', type=int, default=31)
parser.add_argument('--seq_length', type=int, default=12)
parser.add_argument('--in_dim', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--n_features', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=1)

parser.add_argument('--train_length', type=int, default=day*24*0.8)
parser.add_argument('--val_length', type=int, default=day*24*0.1)
parser.add_argument('--batchsize', type=int, default=100)
parser.add_argument('--lam', type=float, default=0.0)

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data', type=str, default='data/person_'+str(day))
parser.add_argument('--adj', type=str, default='data/adj/W.csv')
parser.add_argument('--cell_type', type=str, default='DCRNN')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')

parser.add_argument('--S',type=str, default='data/adj/mean_result_0.0001.csv')
args = parser.parse_args()
SEED=123
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
#-----initialization------
T, B, V = args.seq_length, args.batchsize, args.num_nodes
#device
device = torch.device(args.device)
adj= pd.read_csv(args.adj,header=None)
S=pd.read_csv(args.S,header=None).values
adj=adj.astype("float").values
# adj_parking=torch.mul(torch.tensor(adj),torch.tensor(S))
adj_parking=adj
scaler = MinMaxScaler( )
scaler.fit(adj_parking)
scaler.data_max_
b=scaler.transform(adj_parking)
adj_parking=b.reshape(args.num_nodes,args.num_nodes)
dcrnn=DCRNNModel(adj_parking, device, args.batchsize)
dcrnn.to(device)

class EARLIEST(nn.Module):
    def __init__(self, ninp=args.n_features, nclasses=args.num_nodes, nhid=args.hidden_dimension, rnn_type=args.cell_type,
                 nlayers=args.n_layers, lam=args.lam, dcrnn=dcrnn):
        super(EARLIEST, self).__init__()

        # --- Hyperparameters ---
        self.ninp = ninp
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.lam = lam
        self.nclasses = nclasses
        self.dcrnn=dcrnn

        # --- Sub-networks ---
        self.Controller = Controller(nhid+1, nclasses)
        self.BaselineNetwork = BaselineNetwork(nhid+1, nclasses)
        self.out=nn.Linear(self.nclasses*args.batchsize, self.nhid)

    def forward(self, x, y, epoch=0, test=False):
        #dataset
        dataloader = util.load_dataset(args.data, args.batchsize, args.batchsize, args.batchsize)
        scaler = dataloader['scaler']
      
        # supports = [torch.tensor(int(float(i))).to(device) for i in adj_parking]

        if test: 
            self.Controller._epsilon = 0.0
        else:
            self.Controller._epsilon = self._epsilon 
        # if args.randomadj:
        #     adjinit = None
        # else:
        #     adjinit = supports[0]
        # if args.aptonly:
        #     supports = None
        
        baselines = [] 
        actions = [] 
        log_pi = [] 
        halt_probs = []
        output_inner = []
        halt_points = -torch.ones((B, self.nclasses))
        predictions = torch.zeros((B, self.nclasses), requires_grad=True)
        GO_Symbol,encoder, decoder = self.dcrnn()
        init_hidden_state=encoder.init_hidden(args.batchsize).to(args.device)
        hidden_state_1=init_hidden_state[0]
        hidden_state_2=init_hidden_state[1]
          
        # --- for each timestep, select a set of actions ---
        for t in range(T):
            
            trainx = torch.Tensor(x[:,t:t+1,:,:]).to(device)
            trainy = torch.Tensor(y[:,t+1:13+t,:,:]).to(device)
            trainy=scaler.transform(trainy[..., 0])
            trainy = torch.unsqueeze(trainy, dim=3)
            source = torch.transpose(trainx, dim0=0, dim1=1)
            target = torch.transpose(trainy[..., :1], dim0=0, dim1=1)
            target = torch.cat([GO_Symbol, target], dim=0)
            
            encode_cell = encoder()      
            source = torch.reshape(source, (1, args.batchsize, -1))
            current_inputs_1 = source
            output_hidden = []  # the output hidden states, shape (num_layers, batch, outdim)
            
           
            _, hidden_state_1 = encode_cell[0](current_inputs_1[0, ...], hidden_state_1)  # (10, 31*64)
            
           
            output_inner.append(hidden_state_1)
            output_hidden.append(hidden_state_1)
            current_inputs_2 = torch.stack(output_inner, dim=0).to(device)
            
           
            _, hidden_state_2 = encode_cell[1](current_inputs_2[t, ...], hidden_state_2)
            
           
            output_hidden.append(hidden_state_2)
            
            teacher_forcing_ratio=0 
            pred, hidden= decoder(target, output_hidden,t, scaler, teacher_forcing_ratio=teacher_forcing_ratio)
            
            if test == False:
                self.dcrnn.train()
            else:
                self.dcrnn.eval()
            
            
            # compute halting probability and sample an action
            time = torch.tensor([t], dtype=torch.float, requires_grad=False).view(1, 1,1).repeat(1,B, 1).cuda(0)
            hidden=hidden.cuda(0).unsqueeze(0)
            
            c_in = torch.cat((hidden.cuda(0),time.cuda(0)), dim=2).detach().cuda(0)
          
            a_t, p_t, w_t = self.Controller(c_in)
            b_t = self.BaselineNetwork(torch.cat((hidden, time), dim=2).detach())
            a_t.cuda(0)
            p_t.cuda(0)
            w_t.cuda(0)
            time=time.repeat(1,1,V).squeeze()

            # If a_t == 1 and this class hasn't been halted, save its prediction
            predictions = torch.where((a_t == 1) & (predictions.cuda(0) == 0), pred.squeeze().cuda(0), predictions.cuda(0))
            # If a_t == 1 and this class hasn't been halted, save the time
            halt_points = torch.where((halt_points.cuda(0) == -1) & (a_t == 1), time.cuda(0), halt_points.cuda(0))
            # compute baseline
           
            
            actions.append(a_t.squeeze())
            baselines.append(b_t.squeeze())
            log_pi.append(p_t)
            halt_probs.append(w_t)
            if (halt_points == -1).sum() == 0:  # If no negative values, every class has been halted
                break
        
        # If one element in the batch has not been halting, use its final prediction
        predictions = torch.where(predictions == 0.0, pred.squeeze().cuda(0), predictions.cuda(0))
       
        halt_points = torch.where(halt_points == -1, time.cuda(0), halt_points.cuda(0))
        
        self.locations = np.array(halt_points + 1)
        
        self.baselines = torch.stack(baselines)
        self.log_pi = torch.stack(log_pi)
        self.halt_probs = torch.stack(halt_probs)
        self.actions = torch.stack(actions).transpose(0, 1)
        # --- Compute mask for where actions are updated ---
        self.grad_mask = torch.zeros_like(self.actions)
        
        for b in range(B):
            for n in range(V):
                self.grad_mask[b, :(1 + halt_points[b, n]).long(),n] = 1
        return predictions, (1+halt_points).mean()/T

    def computeLoss(self, y_pred, y_real, halting_points):
        y_pred=y_pred.float().cuda(0)
        y_real = torch.from_numpy(y_real[:,12,:,:])
        y_real=y_real.squeeze().float().cuda(0)
        
        self.r = (2*(y_pred.float().round() == y_real.round()).float()-1).detach()
        y_real=(y_real-y_real.mean())/y_real.std()
        y_pred=(y_pred-y_pred.mean())/y_pred.std()
      
        self.grad_mask=self.grad_mask.transpose(1,0)
        
        self.R = self.r.float() * self.grad_mask.float()

        # --- rescale reward with baseline ---
        b = self.grad_mask.float() * self.baselines.float()
        self.adjusted_reward = self.R.float() - b.float().detach()
        
        # --- compute losses ---
        MSE = torch.nn.MSELoss()
        MAE=torch.nn.L1Loss()
        self.loss_b = MSE(b, self.R) 
        self.loss_r = (-self.log_pi*self.adjusted_reward).sum(0).mean()
        self.loss_c = MSE(y_pred, y_real)
        self.loss_c=(self.loss_c).float()
    
        self.wait_penalty = self.halt_probs.sum(0).sum(1).mean() 
        self.lam = torch.tensor([self.lam], dtype=torch.float, requires_grad=False)
        loss = self.loss_r.cuda(0) + self.loss_b.cuda(0) + self.loss_c.cuda(0) + self.lam.cuda(0)*(self.wait_penalty).cuda(0)
        
        loss_a = MAE(y_pred,y_real)
        # MAPE=np.abs(y_pred-y_real)/np.clip((np.abs(y_pred)+np.abs(y_real)),0.5,None)
        p=y_pred
        r=y_real
        MAPE=np.abs(p-r).sum()/(np.abs(r)+1.5).sum()
        print(loss_a,self.loss_c,MAPE,halting_points)

        return loss.cuda(0),self.loss_c,self.loss_r,loss_a,self.wait_penalty
