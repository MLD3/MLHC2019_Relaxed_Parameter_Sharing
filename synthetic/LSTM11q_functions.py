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
import time
import os

def train(train_loader, model, args, optimizer, verbose=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # if args['use_cuda']:
        #     data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data) #includes all time steps
        
        output = output[:,args['k']:,:].contiguous()
        target = target[:,args['k']:,:].contiguous()
        
        loss = F.mse_loss(output.view(-1,args['output_size']), target.view(-1,args['output_size']))
        loss.backward()
        if args['mode'] in ['moo', 'mow', 'changegate_mow', 'changegate_moo']:
            model.after_backward()
        optimizer.step()
        if verbose & (batch_idx % 500 == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return
            

def test(test_loader, model, args, save_scores=False):
    
    model.eval()
    test_loss = 0
    test_loss_t = np.zeros(args['T']-args['k'])
    pred = []
    y_true = []
    
    for data, target in test_loader:
        # if args['use_cuda']:
        #     data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        
        output = output[:,args['k']:,:].contiguous()
        target = target[:,args['k']:,:].contiguous()

        # sum up batch loss
        test_loss += F.mse_loss(output.view(-1,args['output_size']), target.view(-1,args['output_size']), reduction='sum').item()
        for t in range(args['T']-args['k']):
            test_loss_t[t] += F.mse_loss(output[:,t,:].contiguous().view(-1,args['output_size']), 
                                         target[:,t,:].contiguous().view(-1,args['output_size']), reduction='sum').item()   
        if save_scores:
            pred=pred+list(output.data.cpu().numpy().flatten())
            y_true=y_true+list(target.data.cpu().numpy().flatten())
        
    if save_scores: 
        args['len dataset'] = len(test_loader.dataset)
        filename = args['savedir']+args['runname']+'_'+str(args['synth_num'])+'_'+args['current_modeltype']
        np.savez(filename + '_testscores', pred=pred, y_true=y_true,args=args)
        torch.save(model.state_dict(),  filename + ".ckpt")
    
    test_loss /= len(test_loader.dataset)*(args['T']-args['k'])
    test_loss_t /= len(test_loader.dataset)
    return test_loss, test_loss_t

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def HP_search(args, Net, train_loader, val_loader,test_loader):
    df=pd.DataFrame({'hidden_size':[]})
    val_loss_all=np.infty

    for run in range(args['budget']):
        
        #Draw HP
        args['hidden_size']=int(np.random.choice([100,150,300,500,700,900,1100]))
        args['hyp_hidden_size']=int(np.random.choice([25,50,75,100,125,150]))

        #Init Model
        model = Net(args)
        
        if args['use_cuda']:
            model.cuda()
        optimizer = optim.Adam(model.parameters()) #optim.SGD(model.parameters(), lr=0.01)

        #Train Model
        val_loss_run, epoch_run=np.infty,0        
        early_stop=[]
        for epoch in range(1, args['epochs'] + 1):
            train(train_loader, model, args, optimizer)            
            zval_loss, zval_loss_t =test(val_loader,model,args)
            early_stop.append(zval_loss)
            
            if zval_loss<val_loss_all:
                val_loss_all = zval_loss
            if zval_loss<val_loss_run: 
                val_loss_run, epoch_run, val_loss_t_run = zval_loss, epoch, zval_loss_t
                if df.shape[0]==0: save_scores=True
                elif zval_loss<df.val_loss.min(): save_scores=True
                else: save_scores=False
                args['val_loss'] = zval_loss
                args['current epoch'] = epoch
                args['current run'] = run
                ztest_loss, ztest_loss_t = test(test_loader,model,args,save_scores=save_scores)

            if len(early_stop)>5:
                if abs(early_stop.pop(0)-zval_loss)<.00005:
                    break
                
        #Save Run Information  
        print('Best test loss run {}: {}'.format(run,val_loss_run))
        df=df.append(pd.DataFrame({'val_loss':[val_loss_run],'test_loss':[ztest_loss],\
                                   'val_loss_t':[val_loss_t_run],'test_loss_t':[ztest_loss_t],\
                                   'hidden_size':[args['hidden_size']],'epoch':[epoch_run],\
                                   'hyp_hidden_size':[args['hyp_hidden_size']],\
                                   'num_params':[count_parameters(model)]}),sort=True)
        df.reset_index(inplace=True,drop=True)

    return df

def convert_distb(a):
    # First normalize in case value is negative
    a_min = min(a)
    a_max = max(a)
    a = (a-a_min)/(a_max-a_min)
    # Create distribution
    a_sum = sum(a)
    a = a/a_sum
    return a

def synth_data_search(args, synth_data, model_list):
    df=pd.DataFrame({'l2':[],'hidden_size':[],'epoch':[],'model':[]})
        
    for synth_num in np.arange(args['synthnum_batch'][0],args['synthnum_batch'][1]):
        
        simulation = np.load(args['savedir']+args['genrunname']+'_model'+str(synth_num)+'.npz')
        args['delta'] = simulation['delta']
        args['k'] = simulation['k']
        
        train_data = synth_data(args, k_dist=simulation['k_dist'], d_dist=simulation['d_dist'])
        val_data = synth_data(args, k_dist=simulation['k_dist'], d_dist=simulation['d_dist'])
        test_data = synth_data(args, k_dist=simulation['k_dist'], d_dist=simulation['d_dist'])

        print('Synthnum: {}'.format(synth_num))
        train_loader = data.DataLoader(train_data, batch_size=args['batch_size'],shuffle=True, drop_last=True)
        val_loader = data.DataLoader(val_data, batch_size=args['batch_size'],shuffle=True, drop_last=True)
        test_loader = data.DataLoader(test_data, batch_size=args['batch_size'],shuffle=True, drop_last=True)
        
        zdf = pd.DataFrame({'test_loss':[truerandom2(test_loader,args)]}) 
        print('True Random Baseline {}'.format(zdf.test_loss[0]))
        zdf['model'] = ['truerandom']
        zdf['synthnum'] = [synth_num]
        zdf['genrunname'] = [args['genrunname']]
        zdf['delta'] = [args['delta']]
        zdf['N'] = [args['N']]
        df = df.append(zdf,sort=True)
        
        for modelname, model in model_list:
            args['synth_num'] = synth_num
            print('{}, synth {}'.format(modelname,synth_num))
            
            args['current_modeltype']=modelname
            if 'nidLSTM' in modelname: args['nidLSTM']=float(modelname.split()[1])
                            
            zdf = HP_search(args, model, train_loader, val_loader, test_loader)
            
            zdf['model'] = [modelname]*zdf.shape[0]
            zdf['genrunname'] = [args['genrunname']]*zdf.shape[0]
            zdf['runname'] = [args['runname']]*zdf.shape[0]
            zdf['N'] = [args['N']]*zdf.shape[0]
            zdf['delta'] = [args['delta']]*zdf.shape[0]
            zdf['synthnum'] = [synth_num]*zdf.shape[0]
            df = df.append(zdf, sort=True)
            df.to_pickle(args'savedir']+args['runname']+'_data_search.pickle')
        
        df.to_pickle(args['savedir']+args['runname']+'_data_search.pickle')
    return df


def do_bootstrap(pred, target, args):
    #code from MLD3/ICHE therefore says AUROC but it's using mse loss
    nrep=1000
    lower=np.around(.025*nrep,decimals=0).astype(int)
    upper=np.around(.975*nrep,decimals=0).astype(int)

    zpred = Variable(torch.from_numpy(pred).type(torch.FloatTensor).cuda())
    ztarget = Variable(torch.from_numpy(target).type(torch.FloatTensor).cuda())
    auroc = F.mse_loss(zpred.view(-1,args['output_size']), ztarget.view(-1,args['output_size']), reduction='sum').item()
    auroc = auroc / (args['len dataset']*(args['T']-args['k']))
    
    #Bootstrap 95% CI
    np.random.seed(124)
    pred = np.reshape(pred, (args['len dataset'], args['T']-args['k']))
    target = np.reshape(target, (args['len dataset'], args['T']-args['k']))
    bootstrap=np.random.choice(args['len dataset'],(nrep,args['len dataset']),replace=True)

    aurocbs=np.empty(nrep)
    aurocbs[:]=np.nan
    for i in range(nrep):
        zpred = Variable(torch.from_numpy(pred[bootstrap[i,:],:]).type(torch.FloatTensor).cuda())
        ztarget = Variable(torch.from_numpy(target[bootstrap[i,:],:]).type(torch.FloatTensor).cuda())
        aurocbs[i] = F.mse_loss(zpred.view(-1,args['output_size']), ztarget.view(-1,args['output_size']), reduction='sum').item()
        aurocbs[i] = aurocbs[i] / (args['len dataset']*(args['T']-args['k']))
    zauroc=np.argsort(aurocbs) #sorts smallest to largest
        
    auroc_lower,auroc_upper=empBSCI(aurocbs,auroc,95)
    return auroc_lower, auroc, auroc_upper

def empBSCI(stat_bs,stat,ci):
    #calculates the empirical bootstrap confidence interval
    #stat_bs is the bootstrapped estimate: mean, auroc, aupr etc.
    #stat is the non bootstrapped estimate

    upper=100-(100-ci)/2.0 #upper percentile
    lower=(100-ci)/2.0 #lower percentile
    upperci=np.percentile(stat_bs,upper)
    lowerci=np.percentile(stat_bs,lower)
    # return stat-upperci, stat-lowerci #lower bound, upper bound
    return lowerci, upperci
