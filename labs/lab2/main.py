## 生成 main.py 时请勾选此 cell
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

out_all = {}
has_inferred = False

def predict(data,node_id):
    """
    加载模型和模型预测
    :param node_id: int, 需要进行预测节点的下标
    :return: tensor, 类0以及类1的概率, torch.size[1,2]
    """
    # 这里可以加载你的模型
    device = 0
    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    


    parameters = {
    'lr': 0.01
    , 'num_layers': 2
    , 'hidden_channels': 64
    , 'dropout': 0.0
    , 'batchnorm': False
    , 'weight_decay': 5e-7
                  }
    model_para = parameters.copy()
    model_para.pop('lr')
    model_para.pop('weight_decay')
    
    # model = MLP(in_channels=data.x.size(-1), out_channels=2, **model_para).to(device)
    # model.load_state_dict(torch.load('./results/mlp.pt'))#,map_location=torch.device('cpu'))

    model = GCN(in_channels=data.x.size(-1), out_channels=2, **model_para).to(device)
    model.load_state_dict(torch.load('./results/gcn4.pt'))#,map_location=torch.device('cpu'))

    # model = SAGE(in_channels=data.x.size(-1), out_channels=2, **model_para).to(device)
    # model.load_state_dict(torch.load('./results/sage5.pt'))

    # 模型预测时，测试数据已经进行了归一化处理
    # -------------------------- 实现模型预测部分的代码 ---------------------------

    with torch.no_grad():
        model.eval()
        # MLP
        # out = model(data.x[node_id])
        # SAGE or GCN
        global out_all
        global has_inferred
        if has_inferred == False:
            print("1")
            out_all = model(data.x, data.adj_t)
            has_inferred = True

        y_pred = out_all[node_id].exp()
    
    return y_pred