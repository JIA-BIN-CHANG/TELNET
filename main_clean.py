# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 10:54:39 2022

@author: Yuuki Misaki
"""
import torch.nn.functional as F
import os 
import math
import json
from torch.autograd import Variable
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import coverage_overflow as co
import torch
from torch import nn
from torch import optim

# import tools
import tools_clean as tools
from model.TELNet_Model import MyTransformer
from model.TELNet_Model import TELNet_model

def save_score(path, score_dict):
    allscore = 0
    with open(path, 'w') as fw:
        for video_name in list(score_dict.keys()):
            fscore = score_dict[video_name]
            fw.writelines(f'{video_name}:\t{fscore}\n')
            allscore = fscore + allscore
        avg = allscore/len(score_dict.keys())
        fw.writelines(f'Average:\t{avg}\n')
    return 


def save_bound(path, bound_dict):
    allscore = 0
    with open(path, 'w') as fw:
        for video_name in list(bound_dict.keys()):
            bound_list = bound_dict[video_name]
            fw.writelines(f'{video_name}: {bound_list}\n')
    return 

def train(config):
    # Read config
    device = torch.device(config['device'])
    
    # Declare model
    feature_dim = config['feature_dim']
    windowSize = config['window_size']
    

    
    n_head = 4                              # 固定是4個，反正之後會拿掉
    n_stack = config['n_stack']
    windowSize = config['window_size']
    
    dataset_dir = config['dataset_dir']
    video_list = os.listdir(os.path.join(dataset_dir,'parse_data'))
    label_dir = config['label_dir']
    eval_rate = config['eval_rate']
    print(f'Window Size: {windowSize}, n_stack: {n_stack}')
    if(config['Training_mode'] == 'cross'):
        model = TELNet_model(input_size=feature_dim, windowSize=windowSize).to(device)             ## 這個是簡化過的，現在要以這個為準
        lossfun = nn.CrossEntropyLoss()
        lossL1 = nn.L1Loss()
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10)                             ## 每10個epoch就降低學習率一次
        f_score = 0
        train_losses = []
        for epoch in range(config['epoches']):
            video_count = 0
            epoch_loss = 0
            for video_name in video_list:
                batch_loss = 0
                video_count = video_count +1
                model.train()
                visual_feature_dir = os.path.join(dataset_dir,'parse_data',video_name)   ## 聲明shot feature的路徑
                feature = tools.load_feature(visual_feature_dir).to(device)                 ## 讀取shot feature
                label = tools.load_keyShot(label_dir,video_name).to(device)             ## 讀取BBC的 keyshot
                '''
                ##########################################################################org
                nbatch = int(feature.shape[0]/windowSize)+1                             ## 一個batch15個shot，所以共有nshot/15+1個batch
                for i in range(nbatch):
                    start = i*windowSize                                                ## 每個batch開始的shot，比如說batch0 從shot0開始， batch1從16開始。這邊可以改成有重疊
                    end = min((i+1)*windowSize, feature.shape[0])                       ## 每個batch結束的shot
                    src = feature[start:end]                                            ## 取這一個batch所有的shot feature
                    gt_window = label[start:end]                                  ## 以及所對應的keyShot
                    tools.clean_gt(gt_window, start, windowSize)                              ## 這邊是因為有些的keyShot不在這15中，所以有另外的處理
                    att_out = model(src)                                                ## 將這15shot feature丟到model裡，att_out是任一shot對每個shot的分數
                    
                    lossout = lossfun(att_out.view(-1,windowSize),gt_window)            ## 計算loss值
                    batch_loss += lossout.item()
                    del src
                ##########################################################################
                '''

                ##########################################################################yang merge & loss
                nbatch = int((feature.shape[0]-windowSize)/(windowSize-5))+2              #yang   windowsize = 15
                all_link_np = np.zeros((feature.shape[0],feature.shape[0])) # 383*383 all zero matrix
                value_tmp = [] # store temp attention value
                ii = 1
                for i in range(nbatch):
                    start = i*windowSize                                                ## 每個batch開始的shot，比如說batch0 從shot0開始， batch1從16開始。這邊可以改成有重疊
                    end = (i+1)*windowSize   ## 每個batch結束的shot
                    if start > 0: # windowsize = 15
                        start = start - 5*i
                        end = end - 5*i
                    end = min(end, feature.shape[0])
                    
                    src = feature[start:end]                                            ## 取這一個batch所有的shot feature
                    gt_window = label[start:end]                                  ## 以及所對應的keyShot
                    att_out = model(src)                                                ## 將這15shot feature丟到model裡，att_out是任一shot對每個shot的分數
                ####################################################yang loss 
                    
                    # value, tmp = torch.topk(att_out.view(-1,windowSize),15)                   ## 前面與training一樣，topk是指選高的前k個，因為att_out是對每個shot的分數，所以就是把對一個shot而言連到另一個分數高的shot          
                    # tools.fix_pred(tmp, start)                                                 ## tmp給出的是相對的位置，因此實際的連線要加上開始的索引值

                    new_value = [] # attention output percentage
                    for i in range(5):
                        new_value.append([])
                    if end != feature.shape[0]:
                        index = 14
                        att_out = att_out.view(-1,windowSize)
                        if len(value_tmp) == 0 :
                            value_tmp.append(att_out[10:15,])
                            for i in range(5):
                                att_out = att_out[torch.arange(att_out.size(0))!= index ] 
                                index = index - 1
                            
                            final = [[0]*int(feature.shape[0]-(end)) for i in range(10)] 
                            final = torch.tensor(final) 
                            final = final.to(torch.device('cpu'))
        
                            att_out = att_out.to(torch.device('cpu'))
                            att_out = torch.cat((att_out,final),1)
                            att_out = att_out[:,:] 
                            
                            att_out = att_out.detach().numpy()                                             
                            all_link_np[0:10,:] = att_out[:,:]
                            all_link_np = torch.tensor(all_link_np) 
                            all_link_np = all_link_np.to(torch.device('cpu'))
                            
                        else:  
                            value_tmp.append(att_out[0:5,]) 
                            # new_value is a 5 * 15 matrix
                            for i in range(5):
                                for j in range(15):                                        
                                    if value_tmp[0][i][j] > value_tmp[1][i][j]:
                                        new_value[i].append(value_tmp[0][i][j].item()) 
                                    elif value_tmp[0][i][j] <= value_tmp[1][i][j]:                                
                                        new_value[i].append(value_tmp[1][i][j].item())            
                            del value_tmp[1]
                            value_tmp.append(att_out[10:15,]) 
                            value_tmp[0] = value_tmp[1]
                            del value_tmp[1]
                            for i in range(5):
                                new_value[i] = new_value[i][0:15]
                            for i in range(5):###刪除10~14
                                att_out = att_out[torch.arange(att_out.size(0))!= index ] 
                                index = index - 1
                            for i in range(5):########刪除 0~4
                                att_out = att_out[torch.arange(att_out.size(0))!= i ]    
                            new_value = torch.tensor(new_value)                                   
                            new_value = new_value.to(torch.device('cpu'))                                  
                            att_out = att_out.to(torch.device('cpu'))
                            att_out = torch.cat((new_value,att_out))

                            begin = [[0]*start for i in range(10)]
                            final = [[0]*int(feature.shape[0]-(end)) for i in range(10)]
    
                            begin = torch.tensor(begin) 
                            final = torch.tensor(final)
                            att_out = att_out.clone().detach()                                   
                            
                            att_out = att_out.to(torch.device('cpu'))                         
                            att_out = torch.cat((begin,att_out),1)
                            att_out = torch.cat((att_out,final),1)
                            att_out = att_out[:,:]
                            
                            att_out = att_out.detach().numpy()
                            all_link_np = all_link_np.detach().numpy()
                            all_link_np[ii*10:(ii+1)*10,:] = att_out[:,:]
                            all_link_np = torch.tensor(all_link_np) 
                            all_link_np = all_link_np.to(torch.device('cpu'))
                            
                            ii += 1                                                    
                    elif end == feature.shape[0]:

                        att_out = att_out.view(-1,(feature.shape[0]-start))
                        begin = [[0]*(start) for i in range(feature.shape[0]-start)]
                        
                        att_out = att_out.clone().detach()                                   
                        begin = torch.tensor(begin)     
                        
                        att_out = att_out[0:feature.shape[0]-start]
                        att_out = att_out.to(torch.device('cpu'))                         
                        att_out = torch.cat((begin,att_out),1)
                        att_out = att_out.detach().numpy()
                        
                        all_link_np = all_link_np.detach().numpy()
                        all_link_np[start:feature.shape[0],:] = att_out[:,:]
                        all_link_np = torch.tensor(all_link_np) 
                        all_link_np = all_link_np.to(torch.device('cpu'))
                
                gt_window = label[0:feature.shape[0]]
                gt_window = gt_window.to(torch.device('cpu'))
                lossout = lossfun(all_link_np,gt_window)
                batch_loss = lossout.item()
                lossout = Variable(lossout, requires_grad=True)
                ##########################################################################
                
#                 del feature
                lossout.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                batch_loss = batch_loss
                if np.isnan(batch_loss):
                    pass
                else:
                    epoch_loss += batch_loss
            epoch_loss = epoch_loss/video_count
            print("epoch: {}, loss: {}".format(epoch,epoch_loss))
            train_losses.append(epoch_loss)
            
            if epoch % eval_rate == (eval_rate-1):
                tmp, tmp_bound,boundary2,testing_losses = tools.evaluate_window(label_dir, model, video_list, 5, windowSize, dataset_dir, bbc=config['isBBC'])    ## 一段時間後驗證一次模型，更多說明見evaluate_window
                if tmp>f_score:
                    f_score = tmp
                    bsf_bound = tmp_bound
                    bsf_boundary2 = boundary2
                    best_model = model
                print("Epoch: {}, f_score: {}, best f_score: {}".format(epoch,tmp,f_score))
        return best_model
    else:
        print("Leave One Out Training")
        fscore_dict = {}
        bound_dict = {}
        
#         ==============================org 做十五個時候註解掉
        for test_index in range(len(video_list)):
            test_video = video_list[test_index]
            
            print("test_video",test_video)
            
            model = TELNet_model(input_size=feature_dim, windowSize=windowSize).to(device)  ## 這個是簡化過的，現在要以這個為準
            lossfun = nn.CrossEntropyLoss()
            lossL1 = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=config['lr'])
            scheduler = optim.lr_scheduler.StepLR(optimizer, 10)                             ## 每10個epoch就降低學習率一次
            
            training_list = video_list.copy()
            training_list.remove(test_video)
            
            bsf_score = 0
            model.train()
            train_losses = []
            test_losses = []
            for epoch in range(config['epoches']):
                epoch_loss = 0
                for video_name in training_list:
                    feature_dir = os.path.join(dataset_dir,'parse_data',video_name)
                    feature = tools.load_feature(feature_dir).to(device)
                    label = tools.load_keyShot(label_dir,video_name).to(device)
                    batch_loss=0
#                 ##########################################################################yang merge & loss
                    nbatch = int((feature.shape[0]-windowSize)/(windowSize-5))+2              #yang   windowsize = 15
                    # all_link =  torch.tensor([]).to(torch.device('cpu'))
                    all_link_np = np.zeros((feature.shape[0],feature.shape[0])) # 383*383 all zero matrix
                    value_tmp = [] # store temp attention value
                    ii = 1
                    for i in range(nbatch):
                        start = i*windowSize                                                ## 每個batch開始的shot，比如說batch0 從shot0開始， batch1從16開始。這邊可以改成有重疊
                        end = (i+1)*windowSize   ## 每個batch結束的shot
                        if start > 0: # windowsize = 15
                            start = start - 5*i
                            end = end - 5*i
                        end = min(end, feature.shape[0])
                        
                        src = feature[start:end]                                            ## 取這一個batch所有的shot feature
                        gt_window = label[start:end]                                  ## 以及所對應的keyShot
                        att_out = model(src)                                                ## 將這15shot feature丟到model裡，att_out是任一shot對每個shot的分數
                    ####################################################yang loss 
                        
                        # value, tmp = torch.topk(att_out.view(-1,windowSize),15)                   ## 前面與training一樣，topk是指選高的前k個，因為att_out是對每個shot的分數，所以就是把對一個shot而言連到另一個分數高的shot          
                        # tools.fix_pred(tmp, start)                                                 ## tmp給出的是相對的位置，因此實際的連線要加上開始的索引值

                        new_value = [] # attention output percentage
                        for i in range(5):
                            new_value.append([])
                        if end != feature.shape[0]:
                            index = 14
                            att_out = att_out.view(-1,windowSize)
                            if len(value_tmp) == 0 :
                                value_tmp.append(att_out[10:15,])
                                for i in range(5):
                                    att_out = att_out[torch.arange(att_out.size(0))!= index ] 
                                    index = index - 1
                                
                                final = [[0]*int(feature.shape[0]-(end)) for i in range(10)] 
                                final = torch.tensor(final) 
                                final = final.to(torch.device('cpu'))
            
                                att_out = att_out.to(torch.device('cpu'))
                                att_out = torch.cat((att_out,final),1)
                                att_out = att_out[:,:] 
                               
                                att_out = att_out.detach().numpy()                                             
                                all_link_np[0:10,:] = att_out[:,:]
                                all_link_np = torch.tensor(all_link_np) 
                                all_link_np = all_link_np.to(torch.device('cpu'))
                                
                            else:  
                                value_tmp.append(att_out[0:5,]) 
                                # new_value is a 5 * 15 matrix
                                for i in range(5):
                                    for j in range(15):                                        
                                        if value_tmp[0][i][j] > value_tmp[1][i][j]:
                                            new_value[i].append(value_tmp[0][i][j].item()) 
                                        elif value_tmp[0][i][j] <= value_tmp[1][i][j]:                                
                                            new_value[i].append(value_tmp[1][i][j].item())            
                                del value_tmp[1]
                                value_tmp.append(att_out[10:15,]) 
                                value_tmp[0] = value_tmp[1]
                                del value_tmp[1]
                                for i in range(5):
                                    new_value[i] = new_value[i][0:15]
                                for i in range(5):###刪除10~14
                                    att_out = att_out[torch.arange(att_out.size(0))!= index ] 
                                    index = index - 1
                                for i in range(5):########刪除 0~4
                                    att_out = att_out[torch.arange(att_out.size(0))!= i ]    
                                new_value = torch.tensor(new_value)                                   
                                new_value = new_value.to(torch.device('cpu'))                                  
                                att_out = att_out.to(torch.device('cpu'))
                                att_out = torch.cat((new_value,att_out))

                                begin = [[0]*start for i in range(10)]
                                final = [[0]*int(feature.shape[0]-(end)) for i in range(10)]
        
                                begin = torch.tensor(begin) 
                                final = torch.tensor(final)
                                att_out = torch.tensor(att_out)                                   
                             
                                att_out = att_out.to(torch.device('cpu'))                         
                                att_out = torch.cat((begin,att_out),1)
                                att_out = torch.cat((att_out,final),1)
                                att_out = att_out[:,:]
                                
                                att_out = att_out.detach().numpy()
                                all_link_np = all_link_np.detach().numpy()
                                all_link_np[ii*10:(ii+1)*10,:] = att_out[:,:]
                                all_link_np = torch.tensor(all_link_np) 
                                all_link_np = all_link_np.to(torch.device('cpu'))
                                
                                ii += 1                                                    
                        elif end == feature.shape[0]:

                            att_out = att_out.view(-1,(feature.shape[0]-start))
                            begin = [[0]*(start) for i in range(feature.shape[0]-start)]
                            
                            att_out = torch.tensor(att_out)                                   
                            begin = torch.tensor(begin)     
                            
                            att_out = att_out[0:feature.shape[0]-start]
                            att_out = att_out.to(torch.device('cpu'))                         
                            att_out = torch.cat((begin,att_out),1)
                            att_out = att_out.detach().numpy()
                            
                            all_link_np = all_link_np.detach().numpy()
                            all_link_np[start:feature.shape[0],:] = att_out[:,:]
                            all_link_np = torch.tensor(all_link_np) 
                            all_link_np = all_link_np.to(torch.device('cpu'))

                    gt_window = label[0:feature.shape[0]]
                    gt_window = gt_window.to(torch.device('cpu'))
                    lossout = lossfun(all_link_np,gt_window)
                    batch_loss = lossout.item()
                    lossout = Variable(lossout, requires_grad=True)
                    lossout.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    batch_loss = batch_loss
                    if np.isnan(batch_loss):                           #判断是否是空值 空值true
                        pass
                    else:
                        epoch_loss += batch_loss 

                epoch_loss = epoch_loss/len(training_list)
                print("epoch: {}, loss: {}".format(epoch,epoch_loss))
                train_losses.append(epoch_loss)
                
                tmp, tmp_bound,boundary2,testing_losses = tools.evaluate_window(label_dir, model, [test_video], 5, windowSize, dataset_dir, bbc=config['isBBC'])    ## 一段時間後驗證一次模型，更多說明見evaluate_window
                test_losses.append(testing_losses)
                print("epoch: {},testing loss: {}".format(epoch,testing_losses))
                if tmp>bsf_score:
                    bsf_score = tmp
                    bsf_bound = tmp_bound
                    bsf_boundary2 = boundary2
                print("Epoch: {}, f_score: {}, best f_score: {}".format(epoch,tmp,bsf_score))
                print("best boundary: ",bsf_boundary2)
                    ################################################################################
            fscore_dict.update({test_video:bsf_score})
            bound_dict.update({test_video:bsf_bound})
                # Plot loss figure
                #training
            plt.figure()
            plt.plot(train_losses,'g',label= u'training loss')
            plt.plot(test_losses,'r',label= u'testing loss')
            plt.title(f'{test_video} training & testing loss')
            plt.savefig(r'D://bbc_result//loss//'+f'{test_video} windowSize=15 mask=5 yang_keyshot merge2=5 training & testing yang_loss_new label_correct'+'.png')
            # plt.show()
#             =============================
#             #--------------------------- 每15個給一個boundary 還要去更改 coverage overflow
#         print('video_list',video_list)
#         for test_index in range(len(video_list)):
#             print('test_index',test_index)
#             test_video = video_list[test_index]
            
#             a = [test_video]
#             fscore = 0
#             for i in range(len(a)):                                            ## 因為每個Dataset都有很多影片，故要計算每個的平均
#                 video_name = a[i]
#                 boundary = []
#                 score = co.fscore_eval_bbc(boundary, video_name)                       ## 計算Fscore
#                 fscore += score
#             fscore = fscore/len(a)
#             print(fscore)
#         #=--------------------------------
        return fscore_dict, bound_dict

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Save model at: {save_path}")
    return 

    
    


if __name__ == '__main__':
    config_path = './config_clean.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Using Device:{}".format(torch.device(config['device'])))
    print(f"Training_mode:{config['Training_mode']}, dataset: {config['dataset_dir']}")
    
    #BBC Leave one out
#     score, boundary = train(config)
#     save_score(config['save_score_path'],score)
#     save_bound(config['save_bound_path'],boundary)
    
    #Cross dataset evaluation
    best_model = train(config)
    # save_model(best_model, config['save_model_path'])
    
