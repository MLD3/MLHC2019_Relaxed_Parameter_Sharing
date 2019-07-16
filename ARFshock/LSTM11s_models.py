import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import metrics
from lib.model import ExampleMowLSTM

class mixLSTM(nn.Module):

    def __init__(self, args, setKT=True):
        nn.Module.__init__(self)
        self.args  = args
        self.model = ExampleMowLSTM(args['d'], args['hidden_size'],
                                  args['num_classes'], num_layers=1,
                                  num_directions=1, dropout=0,
                                  activation=nn.LogSoftmax(dim=-1))
        self.num_layers = 1
        self.num_directions = 1
        self.hidden_size = args['hidden_size']
        self.batch_size = args['batch_size']

        if setKT:
            self.model.setKT(args['share_num'], args['T'])
        
    def forward(self, x):
        # change x from (bs, seq_len, d) => (seq_len, bs, d)
        x = x.permute(1, 0, 2)
        input_lengths = [self.args['T']] * x.size(1)

        # set initial hidden and cell states
        h, c = (torch.zeros(self.num_layers * self.num_directions,
                            self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers * self.num_directions,
                            self.batch_size, self.hidden_size))
        cuda_check = next(self.parameters())
        if cuda_check.is_cuda:
        # if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()
        
        states = (h, c)
        outputs, states = self.model(x, states, input_lengths)
        
        return outputs.permute(1, 0, 2)

    def after_backward(self):
        return 
        
def orthogonal(shape):
    flat_shape = (int(shape[0]), int(np.prod(shape[1:])))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

def lstm_ortho_initializer(shape, scale=1.0):
    size_x = shape[0]
    size_h = int(shape[1]/4) # assumes lstm.
    t = np.zeros(shape)
    t[:, :size_h] = orthogonal([size_x, size_h])*scale
    t[:, size_h:size_h*2] = orthogonal([size_x, size_h])*scale
    t[:, size_h*2:size_h*3] = orthogonal([size_x, size_h])*scale
    t[:, size_h*3:] = orthogonal([size_x, size_h])*scale
    return t

def layer_norm_all(h, batch_size, base, num_units, epsilon = 1e-3, gamma = None, beta = None):
    # base = 4 for LSTM    
    h_reshape = h.view(batch_size, base, num_units)
    mean = torch.mean(h_reshape, 2, keepdim=True)
    var = torch.mean(torch.mul(h_reshape-mean, h_reshape-mean), 2, keepdim=True)
    rstd = torch.reciprocal(torch.sqrt(torch.add(var, epsilon)))
    h_reshape = (h_reshape - mean) * rstd
    h = h_reshape.view(batch_size, base*num_units)
    if (gamma is not None) and (beta is not None):
        return gamma*h + beta
    return gamma * h

def layer_norm(x, num_units, epsilon = 1e-3, gamma=None, beta=None):
    x = x.view(x.shape[1], x.shape[2])
    mean = torch.mean(x, 1, keepdim=True)
    x_shifted = x-mean
    var = torch.mean(torch.mul(x_shifted,x_shifted), 1, keepdim=True)
    inv_std = torch.reciprocal(torch.sqrt(torch.add(var, epsilon)))
    if (gamma is not None) and (beta is not None):
        return (gamma*(x_shifted)*inv_std + beta).view(1, x.shape[0], x.shape[1])
    return None

class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args=args
        self.h0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda()) 
        self.c0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda())
        self.fc = nn.Linear(args['hidden_size'], args['num_classes'])
        self.input_weights = nn.Linear(args['d'], 4 * args['hidden_size'])
        self.hidden_weights = nn.Linear(args['hidden_size'], 4 * args['hidden_size'])
        
        gamma_start = 1.0
        self.lna_gamma = Variable((torch.ones(4*args['hidden_size'])*gamma_start).cuda())
        self.lna_beta = Variable((torch.zeros(4*args['hidden_size'])).cuda())
        self.ln_gamma = Variable((torch.ones(args['hidden_size'])*gamma_start).cuda())
        self.ln_beta = Variable((torch.zeros(args['hidden_size'])).cuda())
        
        self.input_weights.weight.data.uniform_()
        self.hidden_weights.weight.data = torch.from_numpy(\
            lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])).transpose()\
            ).type(torch.FloatTensor)

    def forward(self, input):
        def recurrence(input, hidden):
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            gates = layer_norm_all(gates, self.args['batch_size'], 4, self.args['hidden_size'], gamma=self.lna_gamma, beta=self.lna_beta)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)  # o_t
            outgate = torch.sigmoid(outgate)
            
            cy = (forgetgate * cx) + (ingate * cellgate)            
            hy = outgate * torch.tanh(layer_norm(cy, self.args['hidden_size'], gamma=self.ln_gamma, beta=self.ln_beta))            

            return hy, cy
                
        input = input.transpose(0, 1)
        output = []
        steps = range(input.size(0))
        hidden = (self.h0,self.c0)
        for i in steps:
            hidden = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.view(self.args['T'],self.args['batch_size'],self.args['hidden_size']).transpose(0, 1)
        yhat = F.log_softmax(self.fc(output.contiguous().view(-1,self.args['hidden_size'])),dim=1)
        return yhat.view(self.args['batch_size'],self.args['T'],-1)
    
class HyperLSTM(nn.Module):

    def __init__(self, args):
        super(HyperLSTM, self).__init__()
        
        self.args=args
        self.h0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']+args['hyp_hidden_size']).cuda()) 
        self.c0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']+args['hyp_hidden_size']).cuda())
        gamma_start = 1.0
        
        # Main Model
        self.lna_gamma = Variable((torch.ones(4*args['hidden_size'])*gamma_start).cuda())
        self.lna_beta = Variable((torch.zeros(4*args['hidden_size'])).cuda())
        self.ln_gamma = Variable((torch.ones(args['hidden_size'])*gamma_start).cuda())
        self.ln_beta = Variable((torch.zeros(args['hidden_size'])).cuda())
        self.fc = nn.Linear(args['hidden_size'], args['num_classes'])
        
        self.input_weights = nn.Linear(args['d'], 4 * args['hidden_size'], bias=False)
        self.hidden_weights = nn.Linear(args['hidden_size'], 4 * args['hidden_size'], bias=False)
        self.input_weights.weight.data.uniform_()
        self.hidden_weights.weight.data = torch.from_numpy(\
            lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])).transpose()\
            ).type(torch.FloatTensor)
        self.bias = Variable(torch.zeros(4*args['hidden_size']).uniform_().cuda())
                
        # Hypernetwork Specific: Bias
        self.ib_zb = Variable((torch.normal(mean=torch.zeros(self.args['hyp_hidden_size'], self.args['hyp_hidden_size']), 
                                            std=torch.ones(self.args['hyp_hidden_size'], self.args['hyp_hidden_size'])*.01)).cuda())
        self.ib_beta = Variable((torch.zeros(self.args['hyp_hidden_size'], self.args['hidden_size'])).cuda())
        self.jb_zb = Variable((torch.normal(mean=torch.zeros(self.args['hyp_hidden_size'], self.args['hyp_hidden_size']),
                                            std=torch.ones(self.args['hyp_hidden_size'], self.args['hyp_hidden_size'])*.01)).cuda())
        self.jb_beta = Variable((torch.zeros(self.args['hyp_hidden_size'], self.args['hidden_size'])).cuda())
        self.fb_zb = Variable((torch.normal(mean=torch.zeros(self.args['hyp_hidden_size'], self.args['hyp_hidden_size']),
                                            std=torch.ones(self.args['hyp_hidden_size'], self.args['hyp_hidden_size'])*.01)).cuda())
        self.fb_beta = Variable((torch.zeros(self.args['hyp_hidden_size'], self.args['hidden_size'])).cuda())
        self.ob_zb = Variable((torch.normal(mean=torch.zeros(self.args['hyp_hidden_size'], self.args['hyp_hidden_size']),
                                            std=torch.ones(self.args['hyp_hidden_size'], self.args['hyp_hidden_size'])*.01)).cuda())
        self.ob_beta = Variable((torch.zeros(self.args['hyp_hidden_size'], self.args['hidden_size'])).cuda())
        
        # Hypernetwork Specific: input (x) & hidden
        init_gamma = 0.10 / self.args['hyp_hidden_size'] # recurrent batch norm init trick (https://arxiv.org/abs/1603.09025).
        
        self.ix_zw_w = Variable((torch.zeros(self.args['hyp_hidden_size'],self.args['hyp_hidden_size'])).cuda())
        self.ix_zw_b = Variable((torch.ones(self.args['hyp_hidden_size'])).cuda())
        self.ix_alpha = Variable((torch.ones(self.args['hyp_hidden_size'],self.args['hidden_size'])*init_gamma).cuda())
        self.jx_zw_w = Variable((torch.zeros(self.args['hyp_hidden_size'],self.args['hyp_hidden_size'])).cuda())
        self.jx_zw_b = Variable((torch.ones(self.args['hyp_hidden_size'])).cuda())
        self.jx_alpha = Variable((torch.ones(self.args['hyp_hidden_size'],self.args['hidden_size'])*init_gamma).cuda())
        self.fx_zw_w = Variable((torch.zeros(self.args['hyp_hidden_size'],self.args['hyp_hidden_size'])).cuda())
        self.fx_zw_b = Variable((torch.ones(self.args['hyp_hidden_size'])).cuda())
        self.fx_alpha = Variable((torch.ones(self.args['hyp_hidden_size'],self.args['hidden_size'])*init_gamma).cuda())
        self.ox_zw_w = Variable((torch.zeros(self.args['hyp_hidden_size'],self.args['hyp_hidden_size'])).cuda())
        self.ox_zw_b = Variable((torch.ones(self.args['hyp_hidden_size'])).cuda())
        self.ox_alpha = Variable((torch.ones(self.args['hyp_hidden_size'],self.args['hidden_size'])*init_gamma).cuda())
        
        self.ih_zw_w = Variable((torch.zeros(self.args['hyp_hidden_size'],self.args['hyp_hidden_size'])).cuda())
        self.ih_zw_b = Variable((torch.ones(self.args['hyp_hidden_size'])).cuda())
        self.ih_alpha = Variable((torch.ones(self.args['hyp_hidden_size'],self.args['hidden_size'])*init_gamma).cuda())
        self.jh_zw_w = Variable((torch.zeros(self.args['hyp_hidden_size'],self.args['hyp_hidden_size'])).cuda())
        self.jh_zw_b = Variable((torch.ones(self.args['hyp_hidden_size'])).cuda())
        self.jh_alpha = Variable((torch.ones(self.args['hyp_hidden_size'],self.args['hidden_size'])*init_gamma).cuda())
        self.fh_zw_w = Variable((torch.zeros(self.args['hyp_hidden_size'],self.args['hyp_hidden_size'])).cuda())
        self.fh_zw_b = Variable((torch.ones(self.args['hyp_hidden_size'])).cuda())
        self.fh_alpha = Variable((torch.ones(self.args['hyp_hidden_size'],self.args['hidden_size'])*init_gamma).cuda())
        self.oh_zw_w = Variable((torch.zeros(self.args['hyp_hidden_size'],self.args['hyp_hidden_size'])).cuda())
        self.oh_zw_b = Variable((torch.ones(self.args['hyp_hidden_size'])).cuda())
        self.oh_alpha = Variable((torch.ones(self.args['hyp_hidden_size'],self.args['hidden_size'])*init_gamma).cuda())
        
        # Auxiliary Model
        self.hyp_input_weights = nn.Linear(args['d']+args['hidden_size'], 4 * args['hyp_hidden_size'])
        self.hyp_hidden_weights = nn.Linear(args['hyp_hidden_size'], 4 * args['hyp_hidden_size'])
        self.hyp_input_weights.weight.data.uniform_()
        self.hyp_hidden_weights.weight.data = torch.from_numpy(\
            lstm_ortho_initializer((args['hyp_hidden_size'], 4 * args['hyp_hidden_size'])).transpose()\
            ).type(torch.FloatTensor)
        self.hyp_lna_gamma = Variable((torch.ones(4*args['hyp_hidden_size'])*gamma_start).cuda())
        self.hyp_lna_beta = Variable((torch.zeros(4*args['hyp_hidden_size'])).cuda())
        self.hyp_ln_gamma = Variable((torch.ones(args['hyp_hidden_size'])*gamma_start).cuda())
        self.hyp_ln_beta = Variable((torch.zeros(args['hyp_hidden_size'])).cuda())
        
    
    def hyp_recurrence(self, input, hidden):
        hx, cx = hidden  # n_b x hidden_dim
        gates = self.hyp_input_weights(input) + self.hyp_hidden_weights(hx)
        gates = layer_norm_all(gates, self.args['batch_size'], 4, self.args['hyp_hidden_size'], gamma=self.hyp_lna_gamma, beta=self.hyp_lna_beta)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate) 
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(layer_norm(cy, self.args['hyp_hidden_size'], gamma=self.hyp_ln_gamma, beta=self.hyp_ln_beta))            

        return hy, cy

    def recurrence(self, input, hidden, hyper_output):
        hx, cx = hidden  # n_b x hidden_dim

        xh = self.input_weights(input)
        hh = self.hidden_weights(hx)[0,:,:]
        
        ib, jb, fb, ob = self.bias.chunk(4,0) 
        ib = ib + torch.matmul(torch.matmul(hyper_output, self.ib_zb), self.ib_beta) 
        jb = jb + torch.matmul(torch.matmul(hyper_output, self.jb_zb), self.jb_beta)
        fb = fb + torch.matmul(torch.matmul(hyper_output, self.fb_zb), self.fb_beta)
        ob = ob + torch.matmul(torch.matmul(hyper_output, self.ob_zb), self.ob_beta)         
        
        ix, jx, fx, ox = xh.chunk(4, 1)
        ix = torch.mul(torch.matmul(torch.matmul(hyper_output,self.ix_zw_w)+self.ix_zw_b,self.ix_alpha), ix)
        jx = torch.mul(torch.matmul(torch.matmul(hyper_output,self.jx_zw_w)+self.jx_zw_b,self.jx_alpha), jx)
        fx = torch.mul(torch.matmul(torch.matmul(hyper_output,self.fx_zw_w)+self.fx_zw_b,self.fx_alpha), fx)
        ox = torch.mul(torch.matmul(torch.matmul(hyper_output,self.ox_zw_w)+self.ox_zw_b,self.ox_alpha), ox)
        
        ih, jh, fh, oh = hh.chunk(4, 1)
        ih =torch.mul(torch.matmul(torch.matmul(hyper_output,self.ih_zw_w)+self.ih_zw_b,self.ih_alpha), ih)
        jh =torch.mul(torch.matmul(torch.matmul(hyper_output,self.jh_zw_w)+self.jh_zw_b,self.jh_alpha), jh)
        fh =torch.mul(torch.matmul(torch.matmul(hyper_output,self.fh_zw_w)+self.fh_zw_b,self.fh_alpha), fh)
        oh =torch.mul(torch.matmul(torch.matmul(hyper_output,self.oh_zw_w)+self.oh_zw_b,self.oh_alpha), oh)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i = ix + ih + ib
        j = jx + jh + jb
        f = fx + fh + fb
        o = ox + oh + ob
            
        gates = torch.cat((i,j,f,o),1)
        gates = layer_norm_all(gates, self.args['batch_size'], 4, self.args['hidden_size'], gamma=self.lna_gamma, beta=self.lna_beta)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate) 
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(layer_norm(cy, self.args['hidden_size'], gamma=self.ln_gamma, beta=self.ln_beta))            

        return hy, cy
        
    def forward(self, input):
        input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        hidden = (self.h0[:,:,:self.args['hidden_size']],self.c0[:,:,:self.args['hidden_size']])
        hyp_hidden = (self.h0[:,:,self.args['hidden_size']:],self.c0[:,:,self.args['hidden_size']:])
        
        for i in steps:
            hx, cx = hidden
            hyp_hidden = self.hyp_recurrence(torch.cat((input[i],hx[0,:,:]),1), hyp_hidden)
            hyp_hx, hyp_cx = hyp_hidden
            
            hidden = self.recurrence(input[i], hidden, hyp_hx[0,:,:])
            
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.view(self.args['T'],self.args['batch_size'],self.args['hidden_size']).transpose(0, 1)

        yhat = F.relu(self.fc(output.contiguous().view(-1,self.args['hidden_size'])))
        return yhat.view(self.args['batch_size'],self.args['T'],-1)

        
class nidLSTM(nn.Module):

    def __init__(self, args):
        super(nidLSTM, self).__init__()
        self.args=args
        # if args['nidLSTM']==0: raise ValueError("Segmentation length of nidLSTM set to 0")
        # self.args['num_segments']=np.ceil(args['T']/args['nidLSTM']).astype('int')
        self.args['nidLSTM']=np.ceil(self.args['T']/self.args['shiftLSTMk']).astype('int')
            
        self.h0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda()) 
        self.c0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda())
        
        self.fc = nn.ModuleList([nn.Linear(args['hidden_size'], args['num_classes']) for i in range(self.args['shiftLSTMk'])])
        self.input_weights = nn.ModuleList([nn.Linear(args['d'], 4 * args['hidden_size']) for i in range(self.args['shiftLSTMk'])])
        self.hidden_weights = nn.ModuleList([nn.Linear(args['hidden_size'], 4 * args['hidden_size']) for i in range(self.args['shiftLSTMk'])])
        
        gamma_start = 1.0
        self.lna_gamma = Variable((torch.ones(4*args['hidden_size'])*gamma_start).cuda())
        self.lna_beta = Variable((torch.zeros(4*args['hidden_size'])).cuda())
        self.ln_gamma = Variable((torch.ones(args['hidden_size'])*gamma_start).cuda())
        self.ln_beta = Variable((torch.zeros(args['hidden_size'])).cuda())
        
        for i in range(self.args['shiftLSTMk']):
            self.input_weights[i].weight.data.uniform_()
            self.hidden_weights[i].weight.data = torch.from_numpy(\
                lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])).transpose()\
                ).type(torch.FloatTensor)

    def forward(self, input):
        
        def recurrence(input, hidden, timestep):
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights[timestep](input) + self.hidden_weights[timestep](hx)
            gates = layer_norm_all(gates, self.args['batch_size'], 4, self.args['hidden_size'], gamma=self.lna_gamma, beta=self.lna_beta)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)  # o_t
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(layer_norm(cy, self.args['hidden_size'], gamma=self.ln_gamma, beta=self.ln_beta))            
            return hy, cy

        input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        hidden = (self.h0,self.c0)
        for i in steps:
            timestep = np.trunc(i/self.args['nidLSTM']).astype('int')
            hidden = recurrence(input[i], hidden, timestep)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.view(self.args['T'],self.args['batch_size'],self.args['hidden_size'])
        yhat = []

        for i in steps:
            timestep = np.trunc(i/self.args['nidLSTM']).astype('int')
            yhat.append(F.log_softmax(self.fc[timestep](output[i,:,:].contiguous()),dim=1))

        yhat = torch.cat(yhat,0).view(input.size(0), *yhat[0].size())
        yhat = yhat.view(self.args['T'], self.args['batch_size'],self.args['num_classes']).transpose(0,1)
        return yhat 
    
    
class LSTMT(nn.Module):

    def __init__(self, args):
        super(LSTMT, self).__init__()
        self.args=args
        self.h0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda()) 
        self.c0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda())
        self.fc = nn.Linear(args['hidden_size'], args['num_classes'])
        self.input_weights = nn.Linear(args['d']+1, 4 * args['hidden_size'])
        self.hidden_weights = nn.Linear(args['hidden_size'], 4 * args['hidden_size'])
        
        gamma_start = 1.0
        self.lna_gamma = Variable((torch.ones(4*args['hidden_size'])*gamma_start).cuda())
        self.lna_beta = Variable((torch.zeros(4*args['hidden_size'])).cuda())
        self.ln_gamma = Variable((torch.ones(args['hidden_size'])*gamma_start).cuda())
        self.ln_beta = Variable((torch.zeros(args['hidden_size'])).cuda())
        
        self.input_weights.weight.data.uniform_()
        self.hidden_weights.weight.data = torch.from_numpy(\
            lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])).transpose()\
            ).type(torch.FloatTensor)

    def forward(self, input):
        def recurrence(input, hidden, t):
            hx, cx = hidden 
            t_array = torch.ones(input.shape[0],1)*(t/self.args['T'])
            input = torch.cat([input,Variable(t_array.cuda())], dim=1)
        
            gates = self.input_weights(input) + self.hidden_weights(hx)
            gates = layer_norm_all(gates, self.args['batch_size'], 4, self.args['hidden_size'], gamma=self.lna_gamma, beta=self.lna_beta)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)  # o_t
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(layer_norm(cy, self.args['hidden_size'], gamma=self.ln_gamma, beta=self.ln_beta))            

            return hy, cy
        
        input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        hidden = (self.h0,self.c0)
        for i in steps:
            hidden = recurrence(input[i], hidden, i)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.view(self.args['T'],self.args['batch_size'],self.args['hidden_size']).transpose(0, 1)

        yhat = F.relu(self.fc(output.contiguous().view(-1,self.args['hidden_size'])))
        return yhat.view(self.args['batch_size'],self.args['T'],-1)
    
    
