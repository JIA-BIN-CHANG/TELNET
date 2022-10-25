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
import argparse

# import tools
import tools as tools
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
    with open(path, 'w') as fw:
        for video_name in list(bound_dict.keys()):
            bound_list = bound_dict[video_name]
            fw.write(f'{video_name}: ')
            for bound in bound_list:
                fw.write(f'{bound} ')
            fw.write('\n')
    return

def inference(model, feature, label, windowSize):
    '''yang merge & loss'''
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
    return all_link_np 

def train(config):
    # Read config
    device = torch.device(config['device'])
    feature_dim = config['feature_dim']
    windowSize = config['window_size']
    n_stack = config['n_stack']
    dataset_dir = config['dataset_dir']
    video_list = os.listdir(os.path.join(dataset_dir,'parse_data'))
    label_dir = config['label_dir']
    eval_rate = config['eval_rate']
    print(f'Window Size: {windowSize}, n_stack: {n_stack}')
    if(config['Training_mode'] == 'cross'):
        model = TELNet_model(input_size=feature_dim, windowSize=windowSize).to(device)             ## 這個是簡化過的，現在要以這個為準
        lossfun = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10)                             ## 每10個epoch就降低學習率一次
        f_score = 0
        train_losses = []
        for epoch in range(config['epoches']):
            epoch_loss = 0
            for video_name in video_list:
                batch_loss = 0
                model.train()
                visual_feature_dir = os.path.join(dataset_dir,'parse_data',video_name)   ## 聲明shot feature的路徑
                feature = tools.load_feature(visual_feature_dir).to(device)                 ## 讀取shot feature
                label = tools.load_keyShot(label_dir,video_name).to(device)             ## 讀取BBC的 keyshot

                all_link_np = inference(model, feature, label, windowSize) #merge algorithm
                
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
                if np.isnan(batch_loss):
                    pass
                else:
                    epoch_loss += batch_loss
            epoch_loss = epoch_loss/len(video_list)
            print("epoch: {}, loss: {}".format(epoch,epoch_loss))
            train_losses.append(epoch_loss)
            
            #evaluate and save best model to temp
            if epoch % eval_rate == (eval_rate-1):
                tmp, tmp_bound,boundary_list,testing_losses = tools.evaluate_window(label_dir, model, video_list, 5, windowSize, dataset_dir, bbc=config['isBBC'])    ## 一段時間後驗證一次模型，更多說明見evaluate_window
                if tmp>f_score:
                    f_score = tmp
                    bsf_bound = tmp_bound
                    bsf_boundary_list = boundary_list
                    best_model = model
                print("Epoch: {}, f_score: {}, best f_score: {}".format(epoch,tmp,f_score))
        return best_model
    else:
        print("Leave One Out Training")
        fscore_dict = {}
        bound_dict = {}
        for test_video in video_list:
            print("Test video: ",test_video)
            model = TELNet_model(input_size=feature_dim, windowSize=windowSize).to(device)  ## 這個是簡化過的，現在要以這個為準
            lossfun = nn.CrossEntropyLoss()
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

                    all_link_np = inference(model, feature, label, windowSize) #merge algorithm

                    gt_window = label[0:feature.shape[0]]
                    gt_window = gt_window.to(torch.device('cpu'))
                    lossout = lossfun(all_link_np,gt_window)
                    batch_loss = lossout.item()
                    lossout = Variable(lossout, requires_grad=True)
                    lossout.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    if np.isnan(batch_loss):                           #判断是否是空值 空值true
                        pass
                    else:
                        epoch_loss += batch_loss 

                epoch_loss = epoch_loss/len(training_list)
                # print("epoch: {}, loss: {}".format(epoch,epoch_loss))
                train_losses.append(epoch_loss)
                
                #evaluate every epoch and save best f1-score
                tmp, tmp_bound,boundary_list,testing_losses = tools.evaluate_window(label_dir, model, [test_video], 5, windowSize, dataset_dir, bbc=config['isBBC'])    ## 一段時間後驗證一次模型，更多說明見evaluate_window
                test_losses.append(testing_losses)
                # print("epoch: {},testing loss: {}".format(epoch,testing_losses))
                if tmp>bsf_score:
                    bsf_score = tmp
                    bsf_bound = tmp_bound
                    bsf_boundary_list = boundary_list
                print("Epoch: {}, f_score: {}, best f_score: {}".format(epoch,tmp,bsf_score))
                print("Best boundary: ",bsf_boundary_list)

            fscore_dict.update({test_video:bsf_score})
            bound_dict.update({test_video:bsf_bound})
            plt.figure()
            plt.plot(train_losses,'g',label= u'training loss')
            plt.plot(test_losses,'r',label= u'testing loss')
            plt.title(f'{test_video} training & testing loss')
            loss_fig_dir = config['save_loss_plot']
            plt.savefig(loss_fig_dir+f'{test_video}' +'.png')
        return fscore_dict, bound_dict

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Save model at: {save_path}")
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True,
                        help="Use it to train the model by cross or leave one out")
    parser.add_argument('--dataset', required=True,
                        help="Use it to select config file")
    args = parser.parse_args()

    config_path = 'config_{}.json'.format(args.dataset)
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("Using Device:{}".format(torch.device(config['device'])))
    print(f"Training_mode:{args.type}, dataset: {args.dataset}")

    if (args.type == "cross"):
        '''Cross dataset evaluation'''
        config['Training_mode'] = 'cross'
        best_model = train(config)
        # save_model(best_model, config['save_model_path'])
    if (args.type == "leave_one_out"):
        '''BBC Leave one out'''
        config['Training_mode'] = 'leave_one_out'
        score_dict, boundary_dict = train(config)
        save_score(config['save_score_path'],score_dict)
        save_bound(config['save_bound_path'],boundary_dict)
    

    
