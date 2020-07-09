import numpy as np
import torch
import torch.nn as nn
from torch.nn import GRU, LSTM
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ELU, Tanh

class GraphAttentionLayer(torch.nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a_2 = nn.Parameter(torch.zeros(size=(3*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input_, adj, input_2=None):
        h = torch.mm(input_, self.W)
        N = h.size()[0]
        
        if input_2 is None:
            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        else:
            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1), input_2.repeat(N * N, 1)], dim=1).view(N, -1, 3 * self.out_features)
            e = self.leakyrelu(torch.matmul(a_input, self.a_2).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=True)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(torch.nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        #self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, water=None):
        x = F.dropout(x, self.dropout, training=True)
        #x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = torch.stack([att(x, adj, water) for att in self.attentions]).sum(dim=0) / self.nheads
        x = F.dropout(x, self.dropout, training=True)
        x = F.relu(x)
        return x   # last dimension is nheads*out_features


def softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    #num_nodes = maybe_num_nodes(index, num_nodes)
    #print('here1')
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    #print('here2')
    out = out.exp()
    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out

class set2set(torch.nn.Module):
    def __init__(self, input_dim, steps):
        super(set2set, self).__init__()
        self.input_dim = input_dim
        self.steps = steps
        self.lstm = torch.nn.LSTM(2*input_dim, input_dim)

    def layer(self, tensor, num_of_features, num_of_steps):
        ##### input format ########   timesteps X no_of_atoms X lengthof feature vector
        
        n = tensor.shape[0]
        tensor = tensor.transpose(0,1)
        q_star = torch.zeros(n, 2*num_of_features).to('cuda')
        hidden = (torch.zeros(1, n, num_of_features).to('cuda'),
              torch.zeros(1, n, num_of_features).to('cuda'))
        for i in range(num_of_steps):
            q,hidden = self.lstm(q_star.unsqueeze(0), hidden)
            e = torch.sum(tensor*q,2)
            a = F.softmax(e,dim=0)
            r = a.unsqueeze(2)*tensor
            r=  torch.sum(r,0)
            q_star = torch.cat([q.squeeze(0),r],1)
        return q_star

    def forward(self, x, batch, dim=0):
        batch_size = batch.max().item() + 1
        stacks = [x[batch == i].unsqueeze(dim) for i in range(batch_size)]

        return torch.stack([self.layer(i, self.input_dim, self.steps) for i in stacks]).squeeze(1)


class NNDropout(nn.Module):
    def __init__(self, level, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(NNDropout, self).__init__()

        self.level = level
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        
    def forward(self, data, layer):
        #p = torch.sigmoid(self.p_logit) # this is the drop out probablity, trainable. 
        p = torch.scalar_tensor(0.1)
        
        #----------------------------------------------------------------------------------
        if self.level == 'node':
            out = layer(self._concrete_dropout(data.x, p), data.edge_index, data.edge_attr)
            x = out
        elif self.level == 'subgraph':
            #drop_x = self._concrete_dropout(data.x, p)
            out = layer(self._concrete_dropout(data.x, p), data.edge_index_2)
            x = out
        elif self.level == 'graph':
            out = layer(self._concrete_dropout(data, p))
            x = out
        #----------------------------------------------------------------------------------
        
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)
        
        input_dimensionality = x[0].numel() # Number of elements of first item in batch
        #print(input_dimensionality)
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        
        regularization = weights_regularizer + dropout_regularizer
        return out, regularization
        
    def _concrete_dropout(self, x, p):
        # This is like reparameterization tricks. 
        eps = 1e-7
        temp = 0.1
        
        # Concrete distribution relaxation of the Bernoulli random variable
        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        
        return x
