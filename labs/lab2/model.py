from typing import Union
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv

from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T

import numpy as np
from torch_geometric.data import Data
import os

#设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

path='./datasets/632d74d4e2843a53167ee9a1-momodel/' #数据保存路径
dataset_name='DGraph'
dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor())

nlabels = dataset.num_classes
if dataset_name in ['DGraph']:
    nlabels = 2    #本实验中仅需预测类0和类1

data = dataset[0]
data.adj_t = data.adj_t.to_symmetric() #将有向图转化为无向图

if dataset_name in ['DGraph']:
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
if data.y.dim() == 2:
    data.y = data.y.squeeze(1)

split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}  #划分训练集，验证集

train_idx = split_idx['train']

class MLP(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x):    
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)

class GCN(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            if self.batchnorm: 
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor, SparseTensor]):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm: 
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)

class SAGE(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor, SparseTensor]):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm: 
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)

def train(model, data, train_idx, optimizer, no_conv = True):
     # data.y is labels of shape (N, ) 
    model.train()

    optimizer.zero_grad()
    
    if no_conv:
        out = model(data.x[train_idx])
    else:
        out = model(data.x, data.adj_t)[train_idx]

    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, data, split_idx, evaluator, no_conv = True):
    # data.y is labels of shape (N, )
    with torch.no_grad():
        model.eval()

        losses, eval_results = dict(), dict()
        for key in ['train', 'valid']:
            node_id = split_idx[key]
            
            if no_conv:
                out = model(data.x[node_id])
            else:
                out = model(data.x, data.adj_t)[node_id]
            
            y_pred = out.exp()  # (N,num_classes)
            
            losses[key] = F.nll_loss(out, data.y[node_id]).item()
            eval_results[key] = evaluator.eval(data.y[node_id], y_pred)[eval_metric]

    return eval_results, losses, y_pred


parameters = {
    'lr': 0.01
    , 'num_layers': 2
    , 'hidden_channels': 64
    , 'dropout': 0.0
    , 'batchnorm': False
    , 'weight_decay': 5e-7
                  }
epochs = 200
log_steps =10

para_dict = parameters
model_para = parameters.copy()
model_para.pop('lr')
model_para.pop('weight_decay')

# 训练MLP模型
no_conv = True
model = MLP(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)

eval_metric = 'auc'  #使用AUC衡量指标
evaluator = Evaluator(eval_metric)

model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['weight_decay'])
best_valid = 0
min_valid_loss = 1e8

for epoch in range(1,epochs + 1):
    loss = train(model, data, train_idx, optimizer, no_conv)
    eval_results, losses, out = test(model, data, split_idx, evaluator, no_conv)
    train_eval, valid_eval = eval_results['train'], eval_results['valid']
    train_loss, valid_loss = losses['train'], losses['valid']

    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), './results/mlp.pt') #将表现最好的模型保存

    if epoch % log_steps == 0:
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_eval:.3f}, ' # 我们将AUC值乘上100，使其在0-100的区间内
              f'Valid: {100 * valid_eval:.3f} ')


# 训练 GCN 模型
no_conv = False
model = GCN(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)

eval_metric = 'auc'  #使用AUC衡量指标
evaluator = Evaluator(eval_metric)

model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['weight_decay'])
best_valid = 0
min_valid_loss = 1e8

for epoch in range(1,epochs + 1):
    loss = train(model, data, train_idx, optimizer, no_conv)
    eval_results, losses, out = test(model, data, split_idx, evaluator, no_conv)
    train_eval, valid_eval = eval_results['train'], eval_results['valid']
    train_loss, valid_loss = losses['train'], losses['valid']

    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), './results/gcn.pt') #将表现最好的模型保存

    if epoch % log_steps == 0:
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_eval:.3f}, ' # 我们将AUC值乘上100，使其在0-100的区间内
              f'Valid: {100 * valid_eval:.3f} ')

# 训练 SAGE 模型
no_conv = False
model = SAGE(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)

eval_metric = 'auc'  #使用AUC衡量指标
evaluator = Evaluator(eval_metric)

model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['weight_decay'])
best_valid = 0
min_valid_loss = 1e8

for epoch in range(1,epochs + 1):
    loss = train(model, data, train_idx, optimizer, no_conv)
    eval_results, losses, out = test(model, data, split_idx, evaluator, no_conv)
    train_eval, valid_eval = eval_results['train'], eval_results['valid']
    train_loss, valid_loss = losses['train'], losses['valid']

    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), './results/sage.pt') #将表现最好的模型保存

    if epoch % log_steps == 0:
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_eval:.3f}, ' # 我们将AUC值乘上100，使其在0-100的区间内
              f'Valid: {100 * valid_eval:.3f} ')
