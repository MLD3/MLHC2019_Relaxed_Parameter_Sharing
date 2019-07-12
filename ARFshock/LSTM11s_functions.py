import numpy as np
import pandas as pd
import pickle

import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import metrics

def real_train(train_loader, model, args, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # if args['use_cuda']:
        #     data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        if len(target)<args['batch_size']:
            continue
        optimizer.zero_grad()
        output = model(data)
        
        # loss = F.nll_loss(output[:,-1,:], target)
        # print(output.contiguous().view(-1,args['num_classes']).shape, target.shape)
        loss = F.nll_loss(output.contiguous().view(-1,args['num_classes']), target.flatten())
        loss.backward()
        if args['mode'] in ['Shock_moo', 'Shock_mow',
                            'Shock_changegate_moo', 'Shock_changegate_mow',
                            'ARF_moo', 'ARF_mow',
                            'ARF_changegate_moo', 'ARF_changegate_mow']:
            model.after_backward()
        
        optimizer.step()
        # if batch_idx % 500 == 0:
        #     print('Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data[0]))
    return loss.item()
            
def real_test(test_loader, model, args, save_scores=False):  
    model.eval()
    test_loss = 0
    # correct = 0
    predicted=[]
    y_true=[]
    predicted_last=[]
    y_true_last=[]
    predicted_max=[] #y_true_max = y_true_last
    predicted_last5=[]
    y_true_last5=[]
    
    for data, target in test_loader:
        # if args['use_cuda']:
        #     data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        if len(target)<args['batch_size']:
            continue
        output = model(data)
        # output = output[:,-1,:]
        
        # SUM UP BATCH LOSS
        # test_loss += F.nll_loss(output, target, size_average=False).item()
        test_loss += F.nll_loss(output.contiguous().view(-1,args['num_classes']), target.flatten(), size_average=False).item()
        # GET THE INDEX OF THE MAX LOG-PROBABILITY
        # pred = output.data.max(1, keepdim=True)[1]
        # pred = output.contiguous().view(-1,args['num_classes']).data.max(1, keepdim=True)[1]
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
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
    
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), AUC: {:.4f}\n'
    #   .format(test_loss, correct, len(test_loader.dataset),
    #           100. * correct / len(test_loader.dataset),auc))
    if save_scores: 
        np.savez(args['savedir']+args['modelname']+'_testscores', pred=predicted_max, y_true=y_true_last,args=args)
        torch.save(model.state_dict(),  args['savedir']+args['modelname'] + ".ckpt")
    return auc,test_loss,auc_last,auc_max,auc_last5



def real_data_search3(args, real_data, modelname, Net):
    
    # Import Validation & Test Set
    val_data = real_data('val',args)
    val_loader = data.DataLoader(val_data, batch_size=args['batch_size'],shuffle=True)
    test_data = real_data('test',args)
    test_loader = data.DataLoader(test_data, batch_size=args['batch_size'],shuffle=True)

    # Setup
    args['current_modeltype']=modelname
    if 'nidLSTM' in modelname: args['nidLSTM']=float(modelname.split()[1])
    print('{}'.format(modelname)) 
    df=pd.DataFrame({'hidden_size':[]})                  
    if modelname=='idLSTM': args['hidden_size_list'] = [25, 50, 75, 100, 125, 150] #[100,150,300,500]
    else: args['hidden_size_list'] = [25, 50, 75, 100, 125, 150] #[100,150,300,500,700]
    val_auc_max_all=0 
     
    # Training Data
    if args['realstart']:
        train_data = real_data('train',args)
        train_loader = data.DataLoader(train_data, batch_size=args['batch_size'],shuffle=True)
        np.save(args['savedir']+args['modelname']+'sample_ind', train_data.sample_ind)
    else: 
        train_data = real_data('train',args,sample_ind=np.load(args['savedir']+args['genmodelname']+'sample_ind.npy'))
        train_loader = data.DataLoader(train_data, batch_size=args['batch_size'],shuffle=True)
        
    # Runs
    for run in range(args['budget']):
        
        #Draw HP 
        args['hidden_size']=int(np.random.choice(args['hidden_size_list']))
        args['hyp_hidden_size']=int(np.random.choice([25,50,75,100,125,150]))
        args['ratio']=np.random.choice([.1,.25,.5,.75,.9])
        args['rLSTM_init']=np.random.choice([.5,1,3,6,9])
        # args['sigma']=np.random.choice([0.5,1,2,3,4,5])
        args['sigma']=np.random.choice([12, 6, 24, 18])
        args['kvdims']=np.random.choice([15,20,25,30,35,40])
        # if 'SNAIL' in args['current_modeltype']:
        #     args['num_filters']=np.random.choice([10,15,20,25,30,35])
        # else:
        #     args['num_filters']=np.random.choice([10,20,40,80,160,320])
        args['num_filters']=np.random.choice([160,320,640,1280,2560])

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
            _,_,_,ztrain_auc_max,_ = real_test(train_loader,model,args)
            print('-- Run {}, epoch {}, train loss {:.4f}, train auc max {:.4f}, val auc {:.4f}, val auc last {:.4f}, val auc max {:.4f}'.\
                  format(run,epoch,training_loss,ztrain_auc_max,zval_auc,zval_auc_last,zval_auc_max))
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
                ztest_auc,_,ztest_auc_last,ztest_auc_max,ztest_auc_last5= real_test(test_loader,model,args, save_scores=save_scores)

            if len(early_stop)>5:
                if abs(early_stop.pop(0)-zval_auc_max)<.001:
                    break
                
        #Save Run Information                
        print('Best val auc run {}: {}'.format(run,val_auc_run))
        df=df.append(pd.DataFrame({'val_auc':[val_auc_run],'test_auc':[ztest_auc],'val_auc_last':[val_auc_run_last],'val_auc_max':[val_auc_max_run],\
                                   'test_auc_last':[ztest_auc_last],'test_auc_max':[ztest_auc_max],'test_auc_last5':[ztest_auc_last5],\
                                   'hidden_size':[args['hidden_size']],'run':[run],\
                                  'ratio':[args['ratio']],'hyp_hidden_size':[args['hyp_hidden_size']],\
                                  'rLSTM_init':[args['rLSTM_init']],'rLSTM_sigma':[args['sigma']],\
                                  'kvdims':[args['kvdims']],'num_filters':[args['num_filters']]}),sort=False)
        df.reset_index(inplace=True,drop=True)
        df.to_hdf(args['savedir']+args['modelname']+'_data_search.h5',key='data',mode='w')
    
    df['N'] = args['N']
    df['model'] = [modelname]*df.shape[0]
    df['genmodel'] = args['genmodelname']
    df['batch_size']=args['batch_size']
    df['budget']=args['budget']
    df['epochs']=args['epochs']
    df.to_hdf(args['savedir']+args['modelname']+'_data_search.h5',key='data',mode='w')
    return df
