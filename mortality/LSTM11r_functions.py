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
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def real_train(train_loader, model, args, optimizer):
    # assumes data is already on cuda
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if len(target)<args['batch_size']:
            continue
        optimizer.zero_grad()
        output = model(data)
        
        loss = F.nll_loss(output.contiguous().view(-1,args['num_classes']), target.flatten())
        loss.backward()
        if args['model'] in ['moo', 'mow','changegate_mow','changegate_moo']:
            model.after_backward()
        optimizer.step()
    return loss.item()
            
def real_test(test_loader, model, args, save_scores=False):  
    model.eval()
    test_loss = 0
    predicted=[]
    y_true=[]
    predicted_last=[]
    y_true_last=[]
    predicted_max=[] #y_true_max = y_true_last
    predicted_last5=[]
    y_true_last5=[]
    
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if len(target)<args['batch_size']:
            continue
        output = model(data)
        
        test_loss += F.nll_loss(output.contiguous().view(-1,args['num_classes']), target.flatten(), size_average=False).item()
        
        # AUC
        if args['use_cuda']:
            output=output.cpu()
            target=target.cpu()
        predicted=predicted+list(output[:,:,1].data.numpy().flatten())
        y_true=y_true+list(target.data.numpy().flatten())
        predicted_last=predicted_last+list(output[:,-1,1].data.numpy().flatten())
        y_true_last=y_true_last+list(target[:,-1].data.numpy().flatten())        
        predicted_max=predicted_max+list(torch.max(output[:,:,1],dim=1)[0].data.numpy().flatten())
        predicted_last5=predicted_last5+list(output[:,-5:,1].data.numpy().flatten())
        y_true_last5=y_true_last5+list(target[:,-5:].data.numpy().flatten())
        
    test_loss /= len(test_loader.dataset)
    auc=roc_auc_score(y_true,predicted)
    auc_last=roc_auc_score(y_true_last,predicted_last)
    auc_max=roc_auc_score(y_true_last,predicted_max)
    auc_last5=roc_auc_score(y_true_last5,predicted_last5)
    
    if save_scores:
        np.savez(os.path.join(args['savedir'],args['runname']+'_testscores'), pred=predicted_max,y_true=y_true_last,args=args)
        torch.save(model.state_dict(),  os.path.join(args['savedir'],args['runname'] + ".ckpt"))       
        
    return auc,test_loss,auc_last,auc_max,auc_last5


def real_data_search3(args, real_data, Net):
    
    # Import Validation & Test Set
    val_data = real_data('val',args)
    val_loader = data.DataLoader(val_data, batch_size=args['batch_size'],shuffle=True)
    test_data = real_data('test',args)
    test_loader = data.DataLoader(test_data, batch_size=args['batch_size'],shuffle=True)

    # Setup
    if 'shiftLSTM' in args['model']: args['shiftLSTMk']=int(modelname.split()[1])
    print('{}'.format(args['model'])) 
    df=pd.DataFrame({'hidden_size':[]})                  
    val_auc_max_all=0
     
    # Training Data
    if args['realstart']:
        train_data = real_data('train',args)
        train_loader = data.DataLoader(train_data, batch_size=args['batch_size'],shuffle=True)
        np.save(os.path.join(args['savedir'],args['runname']+'sample_ind'), train_data.sample_ind)
    else:
        train_data = real_data('train',args,sample_ind=np.load(os.path.join(args['savedir'],args['genrunname']+'sample_ind.npy'))) 
        train_loader = data.DataLoader(train_data, batch_size=args['batch_size'],
                                       shuffle=True)
        
    # Runs
    for run in range(args['budget']):
        
        #Draw HP 
        args['hidden_size']=int(np.random.choice([100,150,300,500,700]))
        args['hyp_hidden_size']=int(np.random.choice([25,50,75,100,125,150]))

        #Init Model
        model = Net(args)
        if args['use_cuda']:
            model.cuda()
        optimizer = optim.Adam(model.parameters()) #optim.SGD(model.parameters(), lr=0.01)

        #Train Model
        val_auc_max_run, epoch_run=0,0        
        early_stop=[]
        for epoch in range(1, args['epochs'] + 1):
            training_loss = real_train(train_loader, model, args, optimizer)
            zval_auc,_,zval_auc_last,zval_auc_max,zval_auc_last5 = real_test(val_loader,model,args)
            print('-- Run {}, epoch {}, training loss {:.4f}, val auc {:.4f}, val auc last {:.4f}, val auc max {:.4f}'.\
                  format(run,epoch,training_loss,zval_auc,zval_auc_last,zval_auc_max))
            early_stop.append(zval_auc_max)
            if zval_auc_max>val_auc_max_all:
                val_auc_max_all = zval_auc_max
            if zval_auc_max>val_auc_max_run: 
                val_auc_run, epoch_run, val_auc_run_last, val_auc_max_run= zval_auc, epoch, zval_auc_last, zval_auc_max
                if df.shape[0]==0: save_scores=True
                elif zval_auc_max>df.val_auc_max.max(): save_scores=True
                else: save_scores=False
                args['current epoch'] = epoch
                args['current run'] = run
                args['val_auc_max'] = zval_auc_max
                ztest_auc,_,ztest_auc_last,ztest_auc_max,ztest_auc_last5= real_test(test_loader,model,args,save_scores=save_scores)

            if len(early_stop)>5:
                if abs(early_stop.pop(0)-zval_auc_max)<.001:
                    break
                
        #Save Run Information                
        print('Best val auc run {}: {}'.format(run,val_auc_run))
        df=df.append(pd.DataFrame({'val_auc':[val_auc_run],'test_auc':[ztest_auc],'val_auc_last':[val_auc_run_last],'val_auc_max':[val_auc_max_run],\
                                   'test_auc_last':[ztest_auc_last],'test_auc_max':[ztest_auc_max],'test_auc_last5':[ztest_auc_last5],\
                                   'hidden_size':[args['hidden_size']],'run':[run],\
                                  'hyp_hidden_size':[args['hyp_hidden_size']],'num_params':[count_parameters(model)]}),sort=False)
        df.reset_index(inplace=True,drop=True)
        df.to_hdf(os.path.join(args['savedir'],args['runname']+'_data_search.h5'),key='data',mode='w')
    
    df['N'] = args['N']
    df['model'] = [args['model']]*df.shape[0]
    df['genrunname'] = args['genrunname']
    df['batch_size']=args['batch_size']
    df['budget']=args['budget']
    df['epochs']=args['epochs']
    df.to_hdf(os.path.join(args['savedir'],args['runname']+'_data_search.h5'),key='data',mode='w')
    return df

def empBSCI_old(stat_bs,stat,ci):
    #calculates the empirical bootstrap confidence interval
    #stat_bs is the bootstrapped estimate: mean, auroc, aupr etc.
    #stat is the non bootstrapped estimate
    stat2=stat_bs-stat #this is elementwise
    upper=100-(100-ci)/2.0 #upper percentile
    lower=(100-ci)/2.0 #lower percentile
    upperci=np.percentile(stat2,upper)
    lowerci=np.percentile(stat2,lower)
    return stat-upperci, stat-lowerci #lower bound, upper bound

def empBSCI(stat_bs, ci):
    #calculates the empirical bootstrap confidence interval
    #stat_bs is the bootstrapped estimate: mean, auroc, aupr etc.
    #stat is the non bootstrapped estimate
    upper=100-(100-ci)/2.0 #upper percentile
    lower=(100-ci)/2.0 #lower percentile
    upperci=np.percentile(stat_bs,upper)
    lowerci=np.percentile(stat_bs,lower)
    return lowerci, upperci

def calc_auprc(y_true, y_predict):
    from sklearn import metrics
    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, y_predict)
    return metrics.auc(recalls, precisions) 

def do_bootstrap(pred, target, args, mode="auroc", ci=95):
    assert mode in ["auroc", "auprc"], "mode must be auroc or auprc"
    if mode == 'auroc':
        eval_function = roc_auc_score
    elif mode == 'auprc':
        eval_function = calc_auprc
    
    nrep=1000
    
    #Bootstrap 95% CI
    np.random.seed(124)
    bootstrap=np.random.choice(len(pred),(nrep,len(pred)),replace=True)

    aurocbs=np.empty(nrep)
    aurocbs[:]=np.nan
    for i in range(nrep):
        aurocbs[i]=eval_function(target[bootstrap[i,:]],pred[bootstrap[i,:]])
    zauroc=np.argsort(aurocbs) #sorts smallest to largest
    
    auroc=eval_function(target,pred)
    auroc_lower,auroc_upper=empBSCI(aurocbs,ci)
    return auroc_lower, auroc, auroc_upper


