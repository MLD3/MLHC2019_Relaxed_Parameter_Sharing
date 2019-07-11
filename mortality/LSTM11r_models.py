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

class mowLSTM(nn.Module):

    def __init__(self, args, setKT=True):
        from MTL_RNN.lib.model import RNN_LSTM_MoW        
        #super(self.__class__, self).__init__()
        nn.Module.__init__(self)
        self.args  = args
        self.model = RNN_LSTM_MoW(args['d'], args['hidden_size'],
                                  args['num_classes'], num_layers=1,
                                  num_directions=1, dropout=0,
                                  activation=nn.LogSoftmax(dim=1)).cuda()
        if setKT: # if false can be set later
            self.model.setKT(args['share_num'], args['T'])
        
    def forward(self, x):
        # change x from (bs, seq_len, d) => (seq_len, bs, d)
        x = x.permute(1, 0, 2)
        input_lengths = [self.args['T']] * x.size(1)

        # set initial hidden and cell states
        states = self.model.initHidden(self.args['batch_size'])
        outputs, states = self.model(x, states, input_lengths)
        
        return outputs.permute(1, 0, 2)

    def after_backward(self):
        self.model.after_backward()

class mooLSTM(nn.Module):

    def __init__(self, args, setKT=True):
        from MTL_RNN.lib.model import RNN_LSTM_MoO_time
        # super(self.__class__, self).__init__()
        nn.Module.__init__(self)
        self.args  = args
        self.model = RNN_LSTM_MoO_time(args['d'], args['hidden_size'],
                                       args['num_classes'], num_layers=1,
                                       num_directions=1, dropout=0,
                                       activation=nn.LogSoftmax(dim=1)).cuda()
        if setKT: # if False, then can set KT
            self.model.setKT(args['share_num'], args['T'])
        
    def forward(self, x):
        # change x from (bs, seq_len, d) => (seq_len, bs, d)
        x = x.permute(1, 0, 2)
        input_lengths = [self.args['T']] * x.size(1)

        # set initial hidden and cell states
        states = self.model.initHidden(self.args['batch_size'])
        outputs, states = self.model(x, states, input_lengths)
        
        return outputs.permute(1, 0, 2)

    def after_backward(self):
        self.model.after_backward()

class moeLSTM(nn.Module):

    def __init__(self, args, setKT=True):
        from MTL_RNN.lib.model import RNN_LSTM_MoO
        # super(self.__class__, self).__init__()
        nn.Module.__init__(self)
        self.args  = args
        self.model = RNN_LSTM_MoO(args['d'], args['hidden_size'],
                                  args['num_classes'], num_layers=1,
                                  num_directions=1, dropout=0,
                                  activation=nn.LogSoftmax(dim=1)).cuda()
        if setKT: # if False, then can set KT
            self.model.setKT(args['share_num'], args['T'])
        
    def forward(self, x):
        # change x from (bs, seq_len, d) => (seq_len, bs, d)
        x = x.permute(1, 0, 2)
        input_lengths = [self.args['T']] * x.size(1)

        # set initial hidden and cell states
        states = self.model.initHidden(self.args['batch_size'])
        outputs, states = self.model(x, states, input_lengths)
        
        return outputs.permute(1, 0, 2)

    def after_backward(self):
        self.model.after_backward()
        
###### new start ########################################################
class LSTM_jiaxuan(nn.Module):

    def __init__(self, args, activation=None):        
        """Initialize params."""
        super(self.__class__, self).__init__()
        if activation:
            self.activation = activation
        else:
            self.activation = lambda x: x
        
        self.args=args
        self.fc = nn.Linear(args['hidden_size'], args['num_classes'])

        # weights
        self.input_weights_input = nn.ModuleList()
        self.hidden_weights_input = nn.ModuleList()
        
        self.input_weights_output = nn.ModuleList()
        self.hidden_weights_output = nn.ModuleList()

        self.input_weights_forget = nn.ModuleList()
        self.hidden_weights_forget = nn.ModuleList()

        self.input_weights_cell = nn.ModuleList()
        self.hidden_weights_cell = nn.ModuleList()
        
        gamma_start = 1.0
        self.lna_gamma = Variable((torch.ones(4*args['hidden_size'])*gamma_start).cuda())
        self.lna_beta = Variable((torch.zeros(4*args['hidden_size'])).cuda())
        self.ln_gamma = Variable((torch.ones(args['hidden_size'])*gamma_start).cuda())
        self.ln_beta = Variable((torch.zeros(args['hidden_size'])).cuda())

    def forward(self, input, hidden, i_, f_, c_, o_):
        """Propogate input through the network."""
        # i_, f_, c_, o_ are indices for input, forget, cell, and output gate respectively
        # tag = None  #
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            # get the gates:
            gates = torch.cat([
                self.input_weights_input[i_](input) + self.hidden_weights_input[i_](hx),
                self.input_weights_forget[f_](input) + self.hidden_weights_forget[f_](hx),
                self.input_weights_cell[c_](input) + self.hidden_weights_cell[c_](hx),
                self.input_weights_output[o_](input) + self.hidden_weights_output[o_](hx),               
            ], 2)
            gates = layer_norm_all(gates, self.args['batch_size'], 4, self.args['hidden_size'], gamma=self.lna_gamma, beta=self.lna_beta)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)  # o_t
            outgate = torch.sigmoid(outgate)
            
            cy = (forgetgate * cx) + (ingate * cellgate)
            
            hy = outgate * torch.tanh(layer_norm(cy, self.args['hidden_size'], gamma=self.ln_gamma, beta=self.ln_beta))            
            return hy, cy
        
        # input is seq_len first 
        output = []
        steps = range(input.size(0))

        for i in steps:
            hidden = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0) # (seq_len, bs, d)
        seq_len, bs, d  = output.shape

        yhat = self.activation(self.fc(output.view(-1, d)))
        return yhat.view(seq_len, bs, self.args['num_classes']), hidden
        
class BaseModelGateSpecific(nn.Module):

    def __init__(self, base_model, gate_names, args):
        super(self.__class__, self).__init__()        
        gate_names = set(gate_names)
        self.i, self.f, self.c, self.o = 0, 0, 0, 0
        for gate_name in gate_names:
            assert gate_name in ['input', 'forget', 'cell', 'output'], "not supported"

        self.model = base_model
            
        for gate_name in ['input', 'forget', 'cell', 'output']:
            # append to the list of gate weights
            if gate_name == 'input' and (len(self.model.input_weights_input) == 0 or gate_name in gate_names):
                self.model.input_weights_input.append(nn.Linear(args['d'], args['hidden_size']))
                self.model.hidden_weights_input.append(nn.Linear(args['hidden_size'], args['hidden_size']))
                self.model.input_weights_input[-1].weight.data.uniform_()
                self.model.hidden_weights_input[-1].weight.data = nn.init.orthogonal_(torch.empty(self.model.hidden_weights_input[-1].weight.data.shape)).cuda()
                self.i = len(self.model.input_weights_input) - 1
            elif gate_name == 'output' and (len(self.model.input_weights_output) == 0 or gate_name in gate_names):
                self.model.input_weights_output.append(nn.Linear(args['d'], args['hidden_size']))
                self.model.hidden_weights_output.append(nn.Linear(args['hidden_size'], args['hidden_size']))
                self.model.input_weights_output[-1].weight.data.uniform_()
                self.model.hidden_weights_output[-1].weight.data = nn.init.orthogonal_(torch.empty(self.model.hidden_weights_output[-1].weight.data.shape)).cuda()
                self.o = len(self.model.input_weights_output) - 1             
            elif gate_name == 'forget' and (len(self.model.input_weights_forget) == 0 or gate_name in gate_names):
                self.model.input_weights_forget.append(nn.Linear(args['d'], args['hidden_size']))
                self.model.hidden_weights_forget.append(nn.Linear(args['hidden_size'], args['hidden_size']))
                self.model.input_weights_forget[-1].weight.data.uniform_()
                self.model.hidden_weights_forget[-1].weight.data = nn.init.orthogonal_(torch.empty(self.model.hidden_weights_forget[-1].weight.data.shape)).cuda()
                self.f = len(self.model.input_weights_forget) - 1            
            elif gate_name == 'cell' and (len(self.model.input_weights_cell) == 0 or gate_name in gate_names):
                self.model.input_weights_cell.append(nn.Linear(args['d'], args['hidden_size']))
                self.model.hidden_weights_cell.append(nn.Linear(args['hidden_size'], args['hidden_size']))
                self.model.input_weights_cell[-1].weight.data.uniform_()
                self.model.hidden_weights_cell[-1].weight.data = nn.init.orthogonal_(torch.empty(self.model.hidden_weights_cell[-1].weight.data.shape)).cuda()
                self.c = len(self.model.input_weights_cell) - 1
                
    def forward(self, x, hidden):
        # x.shape: (seq_len, bs, _)
        o, hidden = self.model(x, hidden, self.i, self.f, self.c, self.o)
        return o, hidden
    
class ChangeGate_moo(mooLSTM):

    def __init__(self, args):
        mooLSTM.__init__(self, args, setKT=False)                
        # create a shared base model here: mimic LSTM but is seqlen first and returns
        # both o and (h, c)
        self.shared_model = LSTM_jiaxuan(args, activation=nn.LogSoftmax(dim=1))
        # and then change the gate specific to the output
        self.model.base_model = lambda: BaseModelGateSpecific(self.shared_model, args["gatenames"], args)
        self.model.setKT(args['share_num'], args['T'])
        # print("{} input, {} output".format(len(self.shared_model.input_weights_input),
        #                                    len(self.shared_model.input_weights_output)))

class ChangeGate_mow(mowLSTM):

    def __init__(self, args):
        mowLSTM.__init__(self, args, setKT=False)                
        # create a shared base model here: mimic LSTM but is seqlen first and returns
        # both o and (h, c)
        self.shared_model = LSTM_jiaxuan(args, activation=nn.LogSoftmax(dim=1))
        # and then change the gate specific to the output
        self.model.base_model = lambda: BaseModelGateSpecific(self.shared_model, args["gatenames"], args)
        self.model.setKT(args['share_num'], args['T'])
        
        
################################## new ends here ############################

class pytorchLSTM(nn.Module):

    def __init__(self, args):
        """Initialize params."""
        super(pytorchLSTM, self).__init__()
        self.args=args
        self.lstm=nn.LSTM(input_size=args['d'], hidden_size=args['hidden_size'], batch_first=True)
        self.fc = nn.Linear(args['hidden_size'], args['num_classes'])
    
    def init_hidden(self):
        self.h0 = torch.zeros(1, self.args['batch_size'], self.args['hidden_size']).cuda()
        self.c0 = torch.zeros(1, self.args['batch_size'], self.args['hidden_size']).cuda()
        
    def forward(self, input):
        output, (hn,cn) = self.lstm(input, (self.h0,self.c0))
        output = F.relu(self.fc(output))
        return output


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
    #reuse=False, scope="layer_norm", gamma_start=1.0, use_bias=True):
                  
    # Layer Norm (faster version, but not using defun)
    # Performas layer norm on multiple base at once (ie, i, g, j, o for lstm)
    # Reshapes h in to perform layer norm in parallel
    # base = 4 for LSTM
    
    # print(h.shape) #1 100 600
    # h_reshape = tf.reshape(h, [tf.shape(h)[0], base, num_units]) 
        #joh: this step is trying to get from (batch_size x 4*hidden_state_size) -> (batch_size, 4, hidden_state_size)
    h_reshape = h.view(batch_size, base, num_units)
    # mean = tf.reduce_mean(h_reshape, [2], keepdims=True)
        #joh: this calulates the mean along hidden_state size -> returns matrix of shape (batch_size, 4, 1)
    mean = torch.mean(h_reshape, 2, keepdim=True)
    # var = tf.reduce_mean(tf.square(h_reshape - mean), [2], keepdims=True)
        #joh: this calculates average variance along hidden state size -> returns matrix of shape (batch_size, 4, 1)
    var = torch.mean(torch.mul(h_reshape-mean, h_reshape-mean), 2, keepdim=True)
    # epsilon = tf.constant(epsilon)
    # rstd = tf.rsqrt(var + epsilon) 
        #joh: this returns elementwise 1/sqrt(x)
    rstd = torch.reciprocal(torch.sqrt(torch.add(var, epsilon)))
    # h_reshape = (h_reshape - mean) * rstd
    h_reshape = (h_reshape - mean) * rstd
    # h = tf.reshape(h_reshape, [tf.shape(h_reshape)[0], base * num_units])
    h = h_reshape.view(batch_size, base*num_units)
#     with tf.variable_scope(scope):
#         if reuse == True:
#             tf.get_variable_scope().reuse_variables()
        # gamma = tf.get_variable('ln_gamma', [4*num_units], initializer=tf.constant_initializer(gamma_start))
        # if use_bias:
            # beta = tf.get_variable('ln_beta', [4*num_units], initializer=tf.constant_initializer(0.0))
    # if use_bias:
        # return gamma*h + beta
    if (gamma is not None) and (beta is not None):
        return gamma*h + beta
    # return gamma * h
        #joh: gamma and beta are learnable parameters 
    return gamma * h

def layer_norm(x, num_units, epsilon = 1e-3, gamma=None, beta=None):
    # the way I've written this - we can't have gamme or beta = None
    # scope="layer_norm", gamma_start=1.0, reuse=False, use_bias=True
    
    # print(x.shape) # (1, 100, 100) 
    # joh: x is the cell state -> batch_size x hidden_state_size
    # axes = [1]
    x = x.view(x.shape[1], x.shape[2])
    # mean = tf.reduce_mean(x, axes, keepdims=True)
    # joh: tf implementation also expects cell state to be batch_size x hidden_state_size
    mean = torch.mean(x, 1, keepdim=True)
    # x_shifted = x-mean
    x_shifted = x-mean
    # var = tf.reduce_mean(tf.square(x_shifted), axes, keepdims=True)
    var = torch.mean(torch.mul(x_shifted,x_shifted), 1, keepdim=True)
    # inv_std = tf.rsqrt(var + epsilon)
    inv_std = torch.reciprocal(torch.sqrt(torch.add(var, epsilon)))
    # with tf.variable_scope(scope):
    #     if reuse == True:
    #         tf.get_variable_scope().reuse_variables()
    #     gamma = tf.get_variable('ln_gamma', [num_units], initializer=tf.constant_initializer(gamma_start))
    # joh: get_variable gets an existing variable with these parameters or makes a new one. 
    # joh: however, the scope variable passed into this function differs between layer_norm and layer_norm_all
    #     if use_bias:
    #         beta = tf.get_variable('ln_beta', [num_units], initializer=tf.constant_initializer(0.0))
    if (gamma is not None) and (beta is not None):
        return (gamma*(x_shifted)*inv_std + beta).view(1, x.shape[0], x.shape[1])
    # output = gamma*(x_shifted)*inv_std
    # if use_bias:
    #     output = output + beta
    # return output
    return None

class LSTM(nn.Module):

    def __init__(self, args):
        """Initialize params."""
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
        
        # Added to mimic hypernetwork's LSTM implementation
        self.input_weights.weight.data.uniform_()
        self.hidden_weights.weight.data = torch.from_numpy(\
            lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])).transpose()\
            ).type(torch.FloatTensor)

    def forward(self, input):
        """Propogate input through the network."""
        # tag = None  #
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
#             print(gates.shape) #1,8,400
            gates = layer_norm_all(gates, self.args['batch_size'], 4, self.args['hidden_size'], gamma=self.lna_gamma, beta=self.lna_beta)
            # ingate, forgetgate, cellgate, outgate = gates.view(self.args['batch_size'],-1).chunk(4, 1) 
            # joh: no longer need view because layer_norm reshapes
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)  # o_t
            outgate = torch.sigmoid(outgate)
            
#             print(forgetgate.shape) #1,2,400 
#             print(cx.shape) #1,8,100
#             print(ingate.shape) #1,2,400
#             print(cellgate.shape) #1,2,400
            cy = (forgetgate * cx) + (ingate * cellgate)
            
            hy = outgate * torch.tanh(layer_norm(cy, self.args['hidden_size'], gamma=self.ln_gamma, beta=self.ln_beta))            
            # hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            return hy, cy
        
        # print(input.shape) #3,48,76 <- batch is already first => batch_size, T, d
        
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

            # output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
            # output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
#         print(output.shape) # 48, 1, 8, 100
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
        
        # TRY 2
        self.input_weights = nn.Linear(args['d'], 4 * args['hidden_size'], bias=False)
        self.hidden_weights = nn.Linear(args['hidden_size'], 4 * args['hidden_size'], bias=False)
        self.input_weights.weight.data.uniform_().cuda()
        self.hidden_weights.weight.data = nn.init.orthogonal_(torch.empty(self.hidden_weights.weight.data.shape)).cuda()
        self.bias = Variable(torch.zeros(4*args['hidden_size']).uniform_().cuda())
        
        # TRY 1
        # self.input_weights = Variable((torch.ones(args['d'], 4 * args['hidden_size'])).uniform_().cuda())
        # k = np.sqrt(1/args['d'])
        # self.input_bias = Variable(torch.zeros(4*args['hidden_size']).uniform_(-k,k).cuda())        
        # self.hidden_weights = torch.from_numpy(lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size']))).type(torch.FloatTensor)
        # self.hidden_weights = Variable(self.hidden_weights.cuda())
        # k = np.sqrt(1/args['hidden_size'])
        # self.hidden_bias = Variable(torch.zeros(4*args['hidden_size']).uniform_(-k,k).cuda())
        
        # ORIG
        # self.input_weights = nn.Linear(args['d'], 4 * args['hidden_size'])
        # self.hidden_weights = nn.Linear(args['hidden_size'], 4 * args['hidden_size'])
        # self.input_weights.weight.data.uniform_()
        # self.hidden_weights.weight.data = torch.from_numpy(\
        #     lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])).transpose()\
        #     ).type(torch.FloatTensor)
        
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
    
        # TRY 2
        # gates = self.input_weights(input) + self.hidden_weights(hx) + self.bias

        # TRY 1
        # hx_reshape = hx.contiguous().view(self.args['batch_size'],-1)
        # gates = torch.add(torch.matmul(input, self.input_weights),self.input_bias) + torch.add(torch.matmul(hx_reshape, self.hidden_weights),self.hidden_bias)

        # ORIG
        # gates = self.input_weights(input) + self.hidden_weights(hx)
        
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
            # print(hx.shape) #1 hidden_size, hidden_size
            # print(input[i].shape) # hidden_size, d
            hyp_hidden = self.hyp_recurrence(torch.cat((input[i],hx[0,:,:]),1), hyp_hidden)
            hyp_hx, hyp_cx = hyp_hidden
            
            hidden = self.recurrence(input[i], hidden, hyp_hx[0,:,:])
            
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

            # output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
            # output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
#         print(output.shape) # 48, 1, 8, 100
        output = output.view(self.args['T'],self.args['batch_size'],self.args['hidden_size']).transpose(0, 1)

        yhat = F.log_softmax(self.fc(output.contiguous().view(-1,self.args['hidden_size'])),dim=1)
        return yhat.view(self.args['batch_size'],self.args['T'],-1)

    
class nidLSTM(nn.Module):

    def __init__(self, args):
        """Initialize params."""
        super(nidLSTM, self).__init__()
        self.args=args
        if args['nidLSTM']==0: raise ValueError("Segmentation length of nidLSTM set to 0")
        self.args['num_segments']=np.ceil(args['T']/args['nidLSTM']).astype('int')
            
        self.h0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda()) 
        self.c0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda())
        
        self.fc = nn.ModuleList([nn.Linear(args['hidden_size'], args['num_classes']) for i in range(self.args['num_segments'])])
        self.input_weights = nn.ModuleList([nn.Linear(args['d'], 4 * args['hidden_size']) for i in range(self.args['num_segments'])])
        self.hidden_weights = nn.ModuleList([nn.Linear(args['hidden_size'], 4 * args['hidden_size']) for i in range(self.args['num_segments'])])
        
        gamma_start = 1.0
        self.lna_gamma = Variable((torch.ones(4*args['hidden_size'])*gamma_start).cuda())
        self.lna_beta = Variable((torch.zeros(4*args['hidden_size'])).cuda())
        self.ln_gamma = Variable((torch.ones(args['hidden_size'])*gamma_start).cuda())
        self.ln_beta = Variable((torch.zeros(args['hidden_size'])).cuda())
        
        # Added to mimic hypernetwork's LSTM implementation
        for i in range(self.args['num_segments']):
            self.input_weights[i].weight.data.uniform_()
            self.hidden_weights[i].weight.data = torch.from_numpy(\
                lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])).transpose()\
                ).type(torch.FloatTensor)

    def forward(self, input):
        """Propogate input through the network."""
        
        def recurrence(input, hidden, timestep):
            """Recurrence helper."""
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
    
# nidLSTM + Time varying output parameters only
class nidLSTMout(nn.Module):

    def __init__(self, args):
        """Initialize params."""
        super(nidLSTMout, self).__init__()
        self.args=args
        if args['nidLSTM']==0: raise ValueError("Segmentation length of nidLSTM set to 0")
        self.args['num_segments']=np.ceil(args['T']/args['nidLSTM']).astype('int')
        
        self.h0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda()) 
        self.c0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda())
        
        self.fc = nn.Linear(args['hidden_size'], args['num_classes'])
        self.input_weights_out = nn.ModuleList([nn.Linear(args['d'], args['hidden_size']) for i in range(self.args['num_segments'])])
        self.hidden_weights_out = nn.ModuleList([nn.Linear(args['hidden_size'], args['hidden_size']) for i in range(self.args['num_segments'])])
        self.input_weights = nn.Linear(args['d'], 3 * args['hidden_size']) 
        self.hidden_weights = nn.Linear(args['hidden_size'], 3 * args['hidden_size']) 
        
        gamma_start = 1.0
        self.lna_gamma = Variable((torch.ones(4*args['hidden_size'])*gamma_start).cuda())
        self.lna_beta = Variable((torch.zeros(4*args['hidden_size'])).cuda())
        self.ln_gamma = Variable((torch.ones(args['hidden_size'])*gamma_start).cuda())
        self.ln_beta = Variable((torch.zeros(args['hidden_size'])).cuda())
        
        # # Added to mimic hypernetwork's LSTM implementation
        self.input_weights.weight.data.uniform_().cuda()
        # self.hidden_weights.weight.data = (torch.transpose(lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])),0,1)\
        #                                                   [:3 * args['hidden_size'],:]).type(torch.FloatTensor)
        self.hidden_weights.weight.data = torch.from_numpy(lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])).transpose()\
                                                          [:3 * args['hidden_size'],:]).type(torch.FloatTensor)
        
        # # Added to mimic hypernetwork's LSTM implementation
        for i in range(self.args['num_segments']):
            self.input_weights_out[i].weight.data.uniform_().cuda()
            # self.hidden_weights_out[i].weight.data = (torch.transpose(\
            #     lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])),0,1)\
            #     [:args['hidden_size'],:]).type(torch.FloatTensor)
            self.hidden_weights_out[i].weight.data = torch.from_numpy(lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])).transpose()\
                [:args['hidden_size'],:]).type(torch.FloatTensor)

    def forward(self, input):
        """Propogate input through the network."""
        
        def recurrence(input, hidden, timestep):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + self.hidden_weights(hx)
            gates_out = self.input_weights_out[timestep](input) + self.hidden_weights_out[timestep](hx)
            gates = torch.cat((gates,gates_out),2)
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
#         print(output.shape) # 48, 1, 8, 100
        output = output.view(self.args['T'],self.args['batch_size'],self.args['hidden_size']).transpose(0, 1)

        yhat = F.relu(self.fc(output.contiguous().view(-1,self.args['hidden_size'])))
        return yhat.view(self.args['batch_size'],self.args['T'],-1)

    
class LSTMT(nn.Module):

    def __init__(self, args):
        """Initialize params."""
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
        
        # Added to mimic hypernetwork's LSTM implementation
        self.input_weights.weight.data.uniform_().cuda()
        self.hidden_weights.weight.data = nn.init.orthogonal_(torch.empty(self.hidden_weights.weight.data.shape)).cuda()

    def forward(self, input):
        """Propogate input through the network."""
        def recurrence(input, hidden, t):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            #joh: input originally comes in as batch_size x d
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

        yhat = F.log_softmax(self.fc(output.contiguous().view(-1,self.args['hidden_size'])),dim=1)
        return yhat.view(self.args['batch_size'],self.args['T'],-1)
    
    
# Note: this version only affects i&o
class rLSTM(nn.Module):

    def __init__(self, args):
        """Initialize params."""
        super(rLSTM, self).__init__()
        self.args=args
        self.h0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda()) 
        self.c0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda())
        self.fc = nn.Linear(args['hidden_size'], args['num_classes'])
        self.input_weights = nn.Linear(args['d'], 4 * args['hidden_size'])
        self.hidden_weights = nn.Linear(args['hidden_size'], 4 * args['hidden_size'])
#         self.i_w = Variable(torch.FloatTensor([6.0]).cuda(),requires_grad=True)
#         self.o_w = Variable(torch.FloatTensor([6.0]).cuda(),requires_grad=True)
        self.i_w = nn.Parameter(torch.FloatTensor([args['rLSTM_init']]))
        self.o_w = nn.Parameter(torch.FloatTensor([args['rLSTM_init']]))
        
        gamma_start = 1.0
        self.lna_gamma = Variable((torch.ones(4*args['hidden_size'])*gamma_start).cuda())
        self.lna_beta = Variable((torch.zeros(4*args['hidden_size'])).cuda())
        self.ln_gamma = Variable((torch.ones(args['hidden_size'])*gamma_start).cuda())
        self.ln_beta = Variable((torch.zeros(args['hidden_size'])).cuda())
        
        # Added to mimic hypernetwork's LSTM implementation
        self.input_weights.weight.data.uniform_()
        self.hidden_weights.weight.data = torch.from_numpy(\
            lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])).transpose()\
            ).type(torch.FloatTensor)
        

    def forward(self, input):
        
        """Recurrence helper"""
        def recurrence(input, hidden, disti):
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + self.hidden_weights(hx)
            gates = layer_norm_all(gates, self.args['batch_size'], 4, self.args['hidden_size'], gamma=self.lna_gamma, beta=self.lna_beta)
            ingate, forgetgate, cellgate, outgate = gates.view(self.args['batch_size'],-1).chunk(4, 1)
            disti = Variable(torch.FloatTensor(disti).cuda(), requires_grad=False)
            
            ingate = torch.sigmoid(torch.add(torch.mul(disti,self.i_w),ingate))
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(torch.add(torch.mul(disti,self.o_w),outgate))

            cy = (forgetgate * cx) + (ingate * cellgate)
            # hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            hy = outgate * torch.tanh(layer_norm(cy, self.args['hidden_size'], gamma=self.ln_gamma, beta=self.ln_beta)) 
            return hy, cy

        """Calculate d"""
        def calcMean(t,sigma):
            newt=(t-1)*sigma+(sigma/2)
            return newt

        def calcNormal(x,mu,sigma):
            if x.shape[0]==0:
                newx=np.arange(0)
            else:
                newx=(-1*np.square(x-mu))/(2*sigma*sigma)
                newx=((2*np.pi*sigma*sigma)**(-1/2))*np.exp(newx)
                newx=newx-(max(newx)/2)
            return newx  
    
        def calcDist(t):            
            h_max=int(self.args['hidden_size']*self.args['ratio'])
            ind=np.arange(h_max)+1
            sigma=h_max/self.args['T']
            disti=calcNormal(ind,calcMean(t+1,sigma),sigma)
            distconst=np.zeros(self.args['hidden_size']-h_max)
            disti=np.hstack((distconst, disti))
            # distf=np.hstack((distconst, calcNormal(ind,calcMean(t+1,sigma),sigma) ))
            return disti

        """Main"""
        input = input.transpose(0, 1)
        output = []
        steps = range(input.size(0))
        hidden = (self.h0,self.c0)
        for i in steps:
            disti=calcDist(i)
            hidden = recurrence(input[i], hidden, disti)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)
                        
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.view(self.args['T'],self.args['batch_size'],self.args['hidden_size']).transpose(0, 1)

        yhat = F.relu(self.fc(output.contiguous().view(-1,self.args['hidden_size'])))
        return yhat.view(self.args['batch_size'],self.args['T'],-1)

class rLSTM_learned(nn.Module):

    def __init__(self, args):
        """Initialize params."""
        super(rLSTM_learned, self).__init__()
        self.args=args
        self.h0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda()) 
        self.c0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda())
        self.fc = nn.Linear(args['hidden_size'], args['num_classes'])
        self.input_weights = nn.Linear(args['d'], 4 * args['hidden_size'])
        self.hidden_weights = nn.Linear(args['hidden_size'], 4 * args['hidden_size'])
        self.t_weights1 = nn.Linear(1, args['T'])
        self.t_weights2 = nn.Linear(args['T'], 3 * args['hidden_size'])
        
        gamma_start = 1.0
        self.lna_gamma = Variable((torch.ones(4*args['hidden_size'])*gamma_start).cuda())
        self.lna_beta = Variable((torch.zeros(4*args['hidden_size'])).cuda())
        self.ln_gamma = Variable((torch.ones(args['hidden_size'])*gamma_start).cuda())
        self.ln_beta = Variable((torch.zeros(args['hidden_size'])).cuda())
        
        # Added to mimic hypernetwork's LSTM implementation
        self.input_weights.weight.data.uniform_()
        self.hidden_weights.weight.data = torch.from_numpy(\
            lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])).transpose()\
            ).type(torch.FloatTensor)
        

    def forward(self, input):
        
        """Recurrence helper"""
        def recurrence(input, hidden, t):
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + self.hidden_weights(hx)
            t_array = torch.ones(input.shape[0],1)*(t/self.args['T'])
            bias = F.relu(self.t_weights1(Variable(t_array.cuda())))
            bias = self.t_weights2(bias)
            gates = gates + torch.cat((bias, torch.zeros(self.args['batch_size'],self.args['hidden_size']).cuda()),1)
            gates = layer_norm_all(gates, self.args['batch_size'], 4, self.args['hidden_size'], gamma=self.lna_gamma, beta=self.lna_beta)
            ingate, forgetgate, outgate, cellgate = gates.view(self.args['batch_size'],-1).chunk(4, 1)
            
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            # hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            hy = outgate * torch.tanh(layer_norm(cy, self.args['hidden_size'], gamma=self.ln_gamma, beta=self.ln_beta)) 
            return hy, cy

        """Main"""
        input = input.transpose(0, 1)
        output = []
        steps = range(input.size(0))
        hidden = (self.h0,self.c0)
        for i in steps:
            hidden = recurrence(input[i], hidden, i+1)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)
                        
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.view(self.args['T'],self.args['batch_size'],self.args['hidden_size']).transpose(0, 1)

        yhat = F.relu(self.fc(output.contiguous().view(-1,self.args['hidden_size'])))
        return yhat.view(self.args['batch_size'],self.args['T'],-1)

class rLSTM_learned2(nn.Module):

    def __init__(self, args):
        """Initialize params."""
        super(rLSTM_learned2, self).__init__()
        self.args=args
        self.h0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda()) 
        self.c0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda())
        self.fc = nn.Linear(args['hidden_size'], args['num_classes'])
        self.input_weights = nn.Linear(args['d'], 4 * args['hidden_size'])
        self.hidden_weights = nn.Linear(args['hidden_size'], 4 * args['hidden_size'])
        self.t_weights1 = nn.Linear(args['te_size'], args['T'])
        self.t_weights2 = nn.Linear(args['T'], 3 * args['hidden_size'])
        
        # Temporal Embedding
        self.te = torch.zeros(args['T'],args['te_size'])
        time = torch.arange(0, args['T']).unsqueeze(1).type(torch.FloatTensor)
        div_term = torch.exp((torch.arange(0, args['te_size'], 2) * -(np.log(args['te_base']) / args['te_size'])).type(torch.FloatTensor)).unsqueeze(0)
        self.te[:, 0::2] = torch.sin(time * div_term)
        self.te[:, 1::2] = torch.cos(time * div_term)
        
        gamma_start = 1.0
        self.lna_gamma = Variable((torch.ones(4*args['hidden_size'])*gamma_start).cuda())
        self.lna_beta = Variable((torch.zeros(4*args['hidden_size'])).cuda())
        self.ln_gamma = Variable((torch.ones(args['hidden_size'])*gamma_start).cuda())
        self.ln_beta = Variable((torch.zeros(args['hidden_size'])).cuda())
        
        # Added to mimic hypernetwork's LSTM implementation
        self.input_weights.weight.data.uniform_().cuda()
        self.hidden_weights.weight.data = nn.init.orthogonal_(torch.empty(self.hidden_weights.weight.data.shape)).cuda()

    def forward(self, input):
        
        """Recurrence helper"""
        def recurrence(input, hidden, t):
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + self.hidden_weights(hx)
            t_array = self.te[t,:].repeat(input.shape[0],1)
            bias = F.relu(self.t_weights1(Variable(t_array.cuda())))
            bias = self.t_weights2(bias)
            gates = gates + torch.cat((bias, torch.zeros(self.args['batch_size'],self.args['hidden_size']).cuda()),1)
            gates = layer_norm_all(gates, self.args['batch_size'], 4, self.args['hidden_size'], gamma=self.lna_gamma, beta=self.lna_beta)
            ingate, forgetgate, outgate, cellgate = gates.view(self.args['batch_size'],-1).chunk(4, 1)
            
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            # hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            hy = outgate * torch.tanh(layer_norm(cy, self.args['hidden_size'], gamma=self.ln_gamma, beta=self.ln_beta)) 
            return hy, cy

        """Main"""
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

        yhat = F.log_softmax(self.fc(output.contiguous().view(-1,self.args['hidden_size'])),dim=1)
        return yhat.view(self.args['batch_size'],self.args['T'],-1)
    
# Note: this version only affects i&o
class rLSTM_sigma(nn.Module):

    def __init__(self, args):
        """Initialize params."""
        super(rLSTM_sigma, self).__init__()
        self.args=args
        self.h0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda()) 
        self.c0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda())
        self.fc = nn.Linear(args['hidden_size'], args['num_classes'])
        self.input_weights = nn.Linear(args['d'], 4 * args['hidden_size'])
        self.hidden_weights = nn.Linear(args['hidden_size'], 4 * args['hidden_size'])
#         self.i_w = Variable(torch.FloatTensor([6.0]).cuda(),requires_grad=True)
#         self.o_w = Variable(torch.FloatTensor([6.0]).cuda(),requires_grad=True)
        self.i_w = nn.Parameter(torch.FloatTensor([args['rLSTM_init']]))
        self.o_w = nn.Parameter(torch.FloatTensor([args['rLSTM_init']]))
        
        gamma_start = 1.0
        self.lna_gamma = Variable((torch.ones(4*args['hidden_size'])*gamma_start).cuda())
        self.lna_beta = Variable((torch.zeros(4*args['hidden_size'])).cuda())
        self.ln_gamma = Variable((torch.ones(args['hidden_size'])*gamma_start).cuda())
        self.ln_beta = Variable((torch.zeros(args['hidden_size'])).cuda())
        
        # Added to mimic hypernetwork's LSTM implementation
        self.input_weights.weight.data.uniform_()
        self.hidden_weights.weight.data = torch.from_numpy(\
            lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])).transpose()\
            ).type(torch.FloatTensor)
        

    def forward(self, input):
        
        """Recurrence helper"""
        def recurrence(input, hidden, disti):
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + self.hidden_weights(hx)
            gates = layer_norm_all(gates, self.args['batch_size'], 4, self.args['hidden_size'], gamma=self.lna_gamma, beta=self.lna_beta)
            ingate, forgetgate, cellgate, outgate = gates.view(self.args['batch_size'],-1).chunk(4, 1)
            disti = Variable(torch.FloatTensor(disti).cuda(), requires_grad=False)
            
            ingate = torch.sigmoid(torch.add(torch.mul(disti,self.i_w),ingate))
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(torch.add(torch.mul(disti,self.o_w),outgate))

            cy = (forgetgate * cx) + (ingate * cellgate)
            # hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            hy = outgate * torch.tanh(layer_norm(cy, self.args['hidden_size'], gamma=self.ln_gamma, beta=self.ln_beta)) 
            return hy, cy

        """Calculate d"""
        def calcMean(t,sigma):
            newt=(t-1)*sigma+(sigma/2)
            return newt

        def calcNormal(x,mu,sigma):
            if x.shape[0]==0:
                newx=np.arange(0)
            else:
                newx=(-1*np.square(x-mu))/(2*sigma*sigma)
                newx=((2*np.pi*sigma*sigma)**(-1/2))*np.exp(newx)
                newx=newx-(max(newx)/2)
            return newx  
    
        def calcDist(t):            
            h_max=int(self.args['hidden_size']*self.args['ratio'])
            ind=np.arange(h_max)+1
            sigma=(h_max/self.args['T'])*self.args['sigma']
            disti=calcNormal(ind,calcMean(t+1,h_max/self.args['T']),sigma)
            distconst=np.zeros(self.args['hidden_size']-h_max)
            disti=np.hstack((distconst, disti))
            # distf=np.hstack((distconst, calcNormal(ind,calcMean(t+1,sigma),sigma) ))
            return disti

        """Main"""
        input = input.transpose(0, 1)
        output = []
        steps = range(input.size(0))
        hidden = (self.h0,self.c0)
        for i in steps:
            disti=calcDist(i)
            hidden = recurrence(input[i], hidden, disti)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)
                        
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.view(self.args['T'],self.args['batch_size'],self.args['hidden_size']).transpose(0, 1)

        yhat = F.relu(self.fc(output.contiguous().view(-1,self.args['hidden_size'])))
        return yhat.view(self.args['batch_size'],self.args['T'],-1)
    
# Note: this version only affects i&o
class rLSTM_sigma2(nn.Module):

    def __init__(self, args):
        """Initialize params."""
        super(rLSTM_sigma2, self).__init__()
        self.args=args
        self.h0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda()) 
        self.c0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda())
        self.fc = nn.Linear(args['hidden_size'], args['num_classes'])
        self.input_weights = nn.Linear(args['d'], 4 * args['hidden_size'])
        self.hidden_weights = nn.Linear(args['hidden_size'], 4 * args['hidden_size'])
#         self.i_w = Variable(torch.FloatTensor([6.0]).cuda(),requires_grad=True)
#         self.o_w = Variable(torch.FloatTensor([6.0]).cuda(),requires_grad=True)
        self.i_w = nn.Parameter(torch.FloatTensor([args['rLSTM_init']]))
        self.o_w = nn.Parameter(torch.FloatTensor([args['rLSTM_init']]))
        
        gamma_start = 1.0
        self.lna_gamma = Variable((torch.ones(4*args['hidden_size'])*gamma_start).cuda())
        self.lna_beta = Variable((torch.zeros(4*args['hidden_size'])).cuda())
        self.ln_gamma = Variable((torch.ones(args['hidden_size'])*gamma_start).cuda())
        self.ln_beta = Variable((torch.zeros(args['hidden_size'])).cuda())
        
        # Added to mimic hypernetwork's LSTM implementation
        self.input_weights.weight.data.uniform_()
        self.hidden_weights.weight.data = torch.from_numpy(\
            lstm_ortho_initializer((args['hidden_size'], 4 * args['hidden_size'])).transpose()\
            ).type(torch.FloatTensor)
        

    def forward(self, input):
        
        """Recurrence helper"""
        def recurrence(input, hidden, disti, disto):
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + self.hidden_weights(hx)
            gates = layer_norm_all(gates, self.args['batch_size'], 4, self.args['hidden_size'], gamma=self.lna_gamma, beta=self.lna_beta)
            ingate, forgetgate, cellgate, outgate = gates.view(self.args['batch_size'],-1).chunk(4, 1)
            disti = Variable(torch.FloatTensor(disti).cuda(), requires_grad=False)
            disto = Variable(torch.FloatTensor(disto).cuda(), requires_grad=False)
            
            ingate = torch.sigmoid(torch.add(torch.mul(disti,self.i_w),ingate))
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(torch.add(torch.mul(disto,self.o_w),outgate))

            cy = (forgetgate * cx) + (ingate * cellgate)
            # hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            hy = outgate * torch.tanh(layer_norm(cy, self.args['hidden_size'], gamma=self.ln_gamma, beta=self.ln_beta)) 
            return hy, cy

        """Calculate d"""
        def calcMean(t,tv_hid_size):
            # time variant hidden state size.
            increment = tv_hid_size/self.args['T']
            newt=(t-1)*increment + (increment/2)
            return newt

        def calcNormal(x,mu,sigma):
            if x.shape[0]==0:
                newx=np.arange(0)
            else:
                newx=(-1*np.square(x-mu))/(2*sigma*sigma)
                newx=((2*np.pi*sigma*sigma)**(-1/2))*np.exp(newx)
                newx=newx-(max(newx)/2)
            return newx  
            #notice here that we shift the distribution downwards. 
    
        def calcDist(t):            
            h_max=int(self.args['hidden_size']*self.args['ratio'])
            ind=np.arange(h_max)+1
            sigma=(h_max/self.args['T'])*self.args['sigma']
            disti=calcNormal(ind,calcMean(t+1,h_max),sigma)
            distmin=np.amin(disti)
            disti[:np.maximum(int(calcMean(t,h_max)),0)]=distmin
            disto=calcNormal(ind,calcMean(t+1,h_max),sigma)
            disto[np.minimum(int(np.ceil(calcMean(t+2,h_max))),h_max-1):] = distmin
            distconst=np.zeros(self.args['hidden_size']-h_max)
            disti=np.hstack((distconst, disti))
            disto=np.hstack((distconst, disto))
            # distf=np.hstack((distconst, calcNormal(ind,calcMean(t+1,sigma),sigma) ))
            return disti, disto

        """Main"""
        input = input.transpose(0, 1)
        output = []
        steps = range(input.size(0))
        hidden = (self.h0,self.c0)
        for i in steps:
            disti, disto=calcDist(i)
            hidden = recurrence(input[i], hidden, disti, disto)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)
                        
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.view(self.args['T'],self.args['batch_size'],self.args['hidden_size']).transpose(0, 1)

        yhat = F.relu(self.fc(output.contiguous().view(-1,self.args['hidden_size'])))
        return yhat.view(self.args['batch_size'],self.args['T'],-1)

class LSTMTE(nn.Module):

    def __init__(self, args):
        """Initialize params."""
        super(LSTMTE, self).__init__()
        self.args=args
        self.h0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda()) 
        self.c0 = Variable(torch.zeros(1, args['batch_size'], args['hidden_size']).cuda())
        self.fc = nn.Linear(args['hidden_size'], args['num_classes'])
        self.input_weights = nn.Linear(args['d']+args['te_size'], 4 * args['hidden_size'])
        self.hidden_weights = nn.Linear(args['hidden_size'], 4 * args['hidden_size'])
        
        gamma_start = 1.0
        self.lna_gamma = Variable((torch.ones(4*args['hidden_size'])*gamma_start).cuda())
        self.lna_beta = Variable((torch.zeros(4*args['hidden_size'])).cuda())
        self.ln_gamma = Variable((torch.ones(args['hidden_size'])*gamma_start).cuda())
        self.ln_beta = Variable((torch.zeros(args['hidden_size'])).cuda())
        
        # Added to mimic hypernetwork's LSTM implementation
        self.input_weights.weight.data.uniform_().cuda()
        self.hidden_weights.weight.data = nn.init.orthogonal_(torch.empty(self.hidden_weights.weight.data.shape)).cuda()
        
        # Temporal Embedding
        self.te = torch.zeros(args['T'],args['te_size'])
        time = torch.arange(0, args['T']).unsqueeze(1).type(torch.FloatTensor)
        div_term = torch.exp((torch.arange(0, args['te_size'], 2) * -(np.log(args['te_base']) / args['te_size'])).type(torch.FloatTensor)).unsqueeze(0)
        self.te[:, 0::2] = torch.sin(time * div_term)
        self.te[:, 1::2] = torch.cos(time * div_term)
        
    def forward(self, input):
        """Propogate input through the network."""
        def recurrence(input, hidden, t):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            #joh: input originally comes in as batch_size x d
            t_array = self.te[t,:].repeat(input.shape[0],1)
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

        yhat = F.log_softmax(self.fc(output.contiguous().view(-1,self.args['hidden_size'])),dim=1)
        return yhat.view(self.args['batch_size'],self.args['T'],-1)
    
"""A PyTorch implementation of the SNAIL building blocks.

This module implements the three blocks in the _A Simple Neural Attentive
Meta-Learner_ paper Mishra et al.

    URL: https://openreview.net/forum?id=B1DmUzWAW&noteId=B1DmUzWAW

The three building blocks are the following:
    - A dense block, built with causal convolutions.
    - A TC Block, built with a stack of dense blocks.
    - An attention block, similar to the attention mechanism described by
      Vaswani et al (2017).
"""
import math

class CausalConv1d(nn.Module):
    """A 1D causal convolution layer.

    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions per step, and T is the number of steps.
    Output: (B, D_out, T), where B is the minibatch size, D_out is the number
        of dimensions in the output, and T is the number of steps.

    Arguments:
        in_channels (int): number of input channels = D_in
        out_channels (int): number of output channels = D_out
        
    """

    def __init__(self, in_channels, out_channels, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = dilation
        self.causal_conv = nn.Conv1d(
            in_channels,
            out_channels,
            2, #kernel_size, stride=1 default
            padding = self.padding, 
            dilation = dilation
        )

    def forward(self, minibatch):
        return self.causal_conv(minibatch)[:, :, :-self.padding]
        #joh: this last step is necessary to keep last dimension size T otherwise it equals T+dilation
        


class DenseBlock(nn.Module):
    """Two parallel 1D causal convolution layers w/tanh and sigmoid activations

    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions of the input, and T is the number of steps.
    Output: (B, D_in+F, T), where where `B` is the minibatch size, `D_in` is the
        number of dimensions of the input, `F` is the number of filters, and `T`
        is the length of the input sequence.

    Arguments:
        in_channels (int): number of input channels
        filters (int): number of filters per channel
    """
    def __init__(self, in_channels, filters, dilation=1):
        super(DenseBlock, self).__init__()
        self.causal_conv1 = CausalConv1d(
            in_channels,
            filters,
            dilation=dilation
        )
        self.causal_conv2 = CausalConv1d(
            in_channels,
            filters,
            dilation=dilation
        )
        self.causal_conv1_bn = nn.BatchNorm1d(filters)
        self.causal_conv2_bn = nn.BatchNorm1d(filters)
        #BatchNorm1d takes in C. Expects input of size (N,C,L).  Normalization done over the C dimension

    def forward(self, minibatch):
        tanh = torch.tanh(self.causal_conv1_bn(self.causal_conv1(minibatch)))
        sig = torch.sigmoid(self.causal_conv2_bn(self.causal_conv2(minibatch)))
        out = torch.cat([minibatch, tanh*sig], dim=1)
        return out

class TCBlock(nn.Module):
    """A stack of DenseBlocks which dilates to desired sequence length

    The TCBlock adds `ceil(log_2(seq_len))*filters` channels to the output.

    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions of the input, and T is the number of steps.
    Output: (B, D_in+F, T), where where `B` is the minibatch size, `D_in` is the
        number of dimensions of the input, `F` is the number of filters, and `T`
        is the length of the input sequence.

    Arguments:
        in_channels (int): channels for the input
        seq_len (int): length of the sequence. The number of denseblock layers
            is log base 2 of `seq_len`.
        filters (int): number of filters per channel
    """
    def __init__(self, in_channels, seq_len, filters):
        super(TCBlock, self).__init__()
        layer_count = math.ceil(math.log(seq_len)/math.log(2))
        blocks = []
        channel_count = in_channels
        for layer in range(layer_count):
            block = DenseBlock(channel_count, filters, dilation=2**layer)
            blocks.append(block)
            channel_count += filters
        self.blocks = nn.Sequential(*blocks)

    def forward(self, minibatch):
        return self.blocks(minibatch)


class AttentionBlock(nn.Module):
    """An attention mechanism similar to Vaswani et al (2017)

    The input of the AttentionBlock is `BxDxT` where `B` is the input
    minibatch size, `D` is the dimensions of each feature, `T` is the length of
    the sequence.

    The output of the AttentionBlock is `Bx(D+V)xT` where `V` is the size of the
    attention values.

    Arguments:
        input_dims (int): the number of dimensions (or channels) of each element
            in the input sequence
        k_size (int): the size of the attention keys
        v_size (int): the size of the attention values
    """
    def __init__(self, input_dims, k_size, v_size):
        super(AttentionBlock, self).__init__()
        self.key_layer = nn.Linear(input_dims, k_size)
        self.query_layer = nn.Linear(input_dims, k_size)
        self.value_layer = nn.Linear(input_dims, v_size)
            #joh: nn.Linear(in_features,out_features)
        self.sqrt_k = math.sqrt(k_size)

    def forward(self, minibatch):
        minibatch = minibatch.permute(0,2,1)
            #joh: permute changes the ordering of the dimensions. (BxDxT) -> (BxTxD)
            #joh: nn.Linear Input=(N,*,in_features), Output=(N,*,out_features)
            #joh: for each timestep T, returns a linear combination of the channel values.
        keys = self.key_layer(minibatch) #joh: (N,T,k_size)
        queries = self.query_layer(minibatch) #joh: (N,T,k_size)
        values = self.value_layer(minibatch) #joh: (N,T,v_size)
        logits = torch.bmm(queries, keys.transpose(2,1)) #joh: (N,T,T)
            #joh: batch matrix multiplication
        mask = logits.data.new(logits.size(1), logits.size(2)).fill_(1).byte()
            #joh: (T,T) matrix of ones
        mask = torch.triu(mask, 1) 
            #joh: upper right traingle - does not include the diagonal
        mask = mask.unsqueeze(0).expand_as(logits) #joh: (N,T,T)
            #joh: unsqueeze adds a dim 1 at position 0
            #joh: expand_as: expand tensor to same size as logits
        logits.data.masked_fill_(mask, float('-inf'))
            #joh: fills elements of logit with -inf where mask is 1.
            #joh: lower left triangle including diagonal of orig logit values, -inf elsewhere
        probs = F.softmax(logits / self.sqrt_k, dim=2) #joh: (N,T,T)
            #joh: first row only considers t<=1, second row only considers t<=2 etc.
        read = torch.bmm(probs, values) #joh: (N,T,v_size)
        return torch.cat([minibatch, read], dim=2).permute(0,2,1)
            #permute (B,T,D+v_size) -> (B,D+v_size,T)
        
class AttentionBlockT(nn.Module):
    """An attention mechanism similar to Vaswani et al (2017)

    The input of the AttentionBlock is `BxDxT` where `B` is the input
    minibatch size, `D` is the dimensions of each feature, `T` is the length of
    the sequence.

    The output of the AttentionBlock is `Bx(D+V)xT` where `V` is the size of the
    attention values.

    Arguments:
        input_dims (int): the number of dimensions (or channels) of each element
            in the input sequence
        k_size (int): the size of the attention keys
        v_size (int): the size of the attention values
        
    JOH: This version adds a dimension for t in the inpust of key_layer and query_layer
    """
    def __init__(self, input_dims, k_size, v_size):
        super(AttentionBlockT, self).__init__()
        self.key_layer = nn.Linear(input_dims+1, k_size)
        self.query_layer = nn.Linear(input_dims+1, k_size)
        self.value_layer = nn.Linear(input_dims, v_size)
            #joh: nn.Linear(in_features,out_features)
        self.sqrt_k = math.sqrt(k_size)

    def forward(self, minibatch):
        minibatch = minibatch.permute(0,2,1)
            #joh: permute changes the ordering of the dimensions. (BxDxT) -> (BxTxD)
            #joh: nn.Linear Input=(N,*,in_features), Output=(N,*,out_features)
            #joh: for each timestep T, returns a linear combination of the channel values.
        t_array = torch.arange(1,minibatch.shape[1]+1).repeat(minibatch.shape[0],1).unsqueeze(2).float()
        minibatchT = torch.cat([minibatch,Variable(t_array.cuda())], dim=2) 
            #joh: version of minibatch with extra dimension for t
        keys = self.key_layer(minibatchT) #joh: (N,T,k_size)
        queries = self.query_layer(minibatchT) #joh: (N,T,k_size)
        values = self.value_layer(minibatch) #joh: (N,T,v_size)
        logits = torch.bmm(queries, keys.transpose(2,1)) #joh: (N,T,T)
            #joh: batch matrix multiplication
        mask = logits.data.new(logits.size(1), logits.size(2)).fill_(1).byte()
            #joh: (T,T) matrix of ones
        mask = torch.triu(mask, 1) 
            #joh: upper right traingle - does not include the diagonal
        mask = mask.unsqueeze(0).expand_as(logits) #joh: (N,T,T)
            #joh: unsqueeze adds a dim 1 at position 0
            #joh: expand_as: expand tensor to same size as logits
        logits.data.masked_fill_(mask, float('-inf'))
            #joh: fills elements of logit with -inf where mask is 1.
            #joh: lower left triangle including diagonal of orig logit values, -inf elsewhere
        probs = F.softmax(logits / self.sqrt_k, dim=2) #joh: (N,T,T)
            #joh: first row only considers t<=1, second row only considers t<=2 etc.
        read = torch.bmm(probs, values) #joh: (N,T,v_size)
        return torch.cat([minibatch, read], dim=2).permute(0,2,1)
        #permute (B,T,D+v_size) -> (B,D+v_size,T)

class zSNAIL(nn.Module):

    def __init__(self, args):
        super(zSNAIL, self).__init__()
        layer_count = math.ceil(math.log(args['T'])/math.log(2))
        
        self.mod0 = AttentionBlock(args['d'], args['kvdims'], args['kvdims']) #input size , kdims, vdims
#         self.mod1 = TCBlock(args['d']+args['kvdims'], args['T'], args['num_filters']) # in_channels, seq_len, filters
        
#         self.mod2 = AttentionBlock(args['d']+args['kvdims']+args['num_filters']*layer_count, args['kvdims']*2, args['kvdims']*2)
#         self.mod3 = TCBlock(args['d']+args['kvdims']*3+args['num_filters']*layer_count, args['T'], args['num_filters'])
#         self.mod4 = AttentionBlock(args['d']+args['kvdims']*3+2*args['num_filters']*layer_count, args['kvdims']*3, args['kvdims']*3)
        
        # self.out_layer = nn.Conv1d(args['d']+args['kvdims']*6+2*args['num_filters']*layer_count, args['num_classes'], 1) 
        self.out_layer = nn.Conv1d(args['d']+args['kvdims'], args['num_classes'], 1) #<-- this might be an issue
        #in_channels, out_channels, kernel_size    
    
    def forward(self, minibatch):
        minibatch = minibatch.permute(0,2,1)
        out = self.mod0(minibatch)
        # out = self.mod1(out)
        # out = self.mod2(out)
        # out = self.mod3(out)
        # out = self.mod4(out)
        return self.out_layer(out).permute(0,2,1)
    
class SNAIL(nn.Module):

    def __init__(self, args):
        super(SNAIL, self).__init__()
        layer_count = math.ceil(math.log(args['T'])/math.log(2))
        
        self.mod0 = AttentionBlock(args['d'], args['kvdims'], args['kvdims']) #input size , kdims, vdims
        self.mod1 = TCBlock(args['d']+args['kvdims'], args['T'], args['num_filters']) # in_channels, seq_len, filters
        
        self.mod2 = AttentionBlock(args['d']+args['kvdims']+args['num_filters']*layer_count, args['kvdims']*2, args['kvdims']*2)
        self.mod3 = TCBlock(args['d']+args['kvdims']*3+args['num_filters']*layer_count, args['T'], args['num_filters'])
        self.mod4 = AttentionBlock(args['d']+args['kvdims']*3+2*args['num_filters']*layer_count, args['kvdims']*3, args['kvdims']*3)
        
        self.out_layer = nn.Conv1d(args['d']+args['kvdims']*6+2*args['num_filters']*layer_count, args['num_classes'], 1) 
        #in_channels, out_channels, kernel_size    
    
    def forward(self, minibatch):
        minibatch = minibatch.permute(0,2,1)
        out = self.mod0(minibatch)
        out = self.mod1(out)
        out = self.mod2(out)
        out = self.mod3(out)
        out = self.mod4(out)
        out = self.out_layer(out)
        out = F.log_softmax(out, dim=1)
        return out.permute(0,2,1)
        
class SNAILT(nn.Module):

    def __init__(self, args):
        super(SNAILT, self).__init__()
        layer_count = math.ceil(math.log(args['T'])/math.log(2))
        
        self.mod0 = AttentionBlockT(args['d'], args['kvdims'], args['kvdims']) #input size , kdims, vdims
        self.mod1 = TCBlock(args['d']+args['kvdims'], args['T'], args['num_filters']) # in_channels, seq_len, filters
        
        self.mod2 = AttentionBlockT(args['d']+args['kvdims']+args['num_filters']*layer_count, 2*args['kvdims'], 2*args['kvdims'])
        self.mod3 = TCBlock(args['d']+args['kvdims']*3+args['num_filters']*layer_count, args['T'], args['num_filters'])
        self.mod4 = AttentionBlockT(args['d']+args['kvdims']*3+2*args['num_filters']*layer_count, 3*args['kvdims'], 3*args['kvdims'])
        
        self.out_layer = nn.Conv1d(args['d']+args['kvdims']*6+2*args['num_filters']*layer_count, args['num_classes'], 1) 
        #in_channels, out_channels, kernel_size
                
    def forward(self, minibatch):
        minibatch = minibatch.permute(0,2,1)
        out = self.mod0(minibatch)
        out = self.mod1(out)
        out = self.mod2(out)
        out = self.mod3(out)
        out = self.mod4(out)
        return self.out_layer(out).permute(0,2,1)
    

    
class TCN(nn.Module):

    def __init__(self, args):
        super(TCN, self).__init__()
        layer_count = math.ceil(math.log(args['T'])/math.log(2))

        self.mod0 = TCBlock(args['d'], args['T'], args['num_filters']) # in_channels, seq_len, filters        
        self.out_layer = nn.Conv1d(args['d']+args['num_filters']*layer_count, args['num_classes'], 1) #in_channels, out_channels, kernel_size
        
    def forward(self, minibatch):
        minibatch = minibatch.permute(0,2,1)
        out = self.mod0(minibatch)
        return F.log_softmax(self.out_layer(out),dim=1).permute(0,2,1)


    
