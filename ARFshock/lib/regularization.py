from collections import defaultdict
import torch

def bdot(a, b): # not used
    if len(a.shape) == 1:
        return a.dot(b)
    
    B = a.shape[0]
    S = a.shape[1]
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

def LSM(net): # layer similarity matrix for feed forward network
    '''
    apply to layers with experts with the same id
    return a list of LSM 
    1. test if the module has last_coef 
    2. collect last_coef for experts with the same id
    3. calculate LSM as desired
    '''

    id2coef = defaultdict(list)
    for name, module in net.named_modules():
        if hasattr(module, 'last_coef'):
            id2coef[id(module.experts)].append(module.last_coef)

    Ss = []
    for eid, coef_list in id2coef.items():
        coef_list = torch.stack(coef_list)
        # flatten for adaptive gate
        coef_list = coef_list.view(len(coef_list), -1) # (T, bs x num_experts)
    
        S = torch.mm(coef_list, coef_list.t())
        s_d = torch.diag(S)
        denom = torch.sqrt(s_d.ger(s_d)) # outer product
        S /= denom
        Ss.append(S)
        
    return Ss

def LSM_RNN(net): # layer similarity matrix for recurrent networks
    '''
    apply to layers with experts with the same id
    return a list of LSM
    '''
    assert hasattr(net, 'cells')

    coef_list = torch.stack([m.last_coef for m in net.cells])
    # flatten for adaptive gate
    coef_list = coef_list.view(len(coef_list), -1) # (T, bs x num_experts)
    
    S = torch.mm(coef_list, coef_list.t())
    s_d = torch.diag(S)
    denom = torch.sqrt(s_d.ger(s_d)) # outer product
    return [S / denom]

def LSM_RNN_old(net): # layer similarity matrix for recurrent networks
    '''
    apply to layers with experts with the same id, 5 times slower? what?
    return a list of LSM (only upper half filled as its symmetric)
    '''
    coef_list = torch.nn.functional.softmax(net.coef, dim=1)
    # flatten for adaptive gate
    coef_list = coef_list.view(len(coef_list), -1) # (T, bs x num_experts)
    
    S = torch.mm(coef_list, coef_list.t())
    s_d = torch.diag(S)
    denom = torch.sqrt(s_d.ger(s_d)) # outer product
    return [S / denom]

class Total_variation_LSM:

    def __init__(self, criterion, alpha, net, LSM):
        self.criterion = criterion
        self.alpha = alpha
        self.net = net
        self.LSM = LSM
        
    def reg(self):
        # regularize on the diagonal + 1 locations for each LSM matrix        
        return -sum([sum(S.diag(1)) for S in self.LSM(self.net)])

    def __call__(self, yhat, y):
        self.loss = self.criterion(yhat, y)
        self.reg_loss = self.reg() * self.alpha
        return self.loss + self.reg_loss
    
class All_share_LSM:

    def __init__(self, criterion, alpha, net, LSM):
        self.criterion = criterion
        self.alpha = alpha
        self.net = net
        self.LSM = LSM

    def reg(self):
        # regularize upper triangular part of LSM
        return -sum([sum(S[~torch.ones(len(S), len(S)).tril().byte()])
                     for S in self.LSM(self.net)])

    def __call__(self, yhat, y):
        self.loss = self.criterion(yhat, y)
        self.reg_loss = self.reg() * self.alpha
        return self.loss + self.reg_loss
    

