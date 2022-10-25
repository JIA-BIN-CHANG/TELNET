# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:37:11 2022

@author: Yuuki Misaki
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from torch import nn
import coverage_overflow as co

with open('./config_bbc.json','r') as f:
    config = json.load(f)

device = config['device']

def load_boundarykeyShot(boundary):
    key_gt = []
    boundary = boundary[0]
    boundary[0] = 0
    for i in range(len(boundary)):
        key = boundary[i+1]
        length = boundary[i+1]-boundary[i]
        key_gt += [key for i in range(length)]
        print(key_gt)
        if key == boundary[-1]:
            break
    key_gt = np.delete(key_gt, -1)
    key_gt = torch.tensor(key_gt)
    return key_gt

def load_feature(path,feature_dim=4096):
    listShot = os.listdir(path)
    nShot = len(listShot)
    
    features = torch.empty((nShot,feature_dim))
    for i in range(nShot):
        feature_path = os.path.join(path,listShot[i])
        tmp = torch.load(feature_path,map_location=torch.device('cpu'))
        features[i,:] = tmp
    
    return features

def load_keyShot(label_dir, video_name):
    yang_keyshot_dir = "./yang_bbc_keyshot/keyshot"
    tmp = open(os.path.join(label_dir,'{}_shot.txt').format(video_name)).readlines()
    scene_boundary = [int(each) for each in tmp[0].split(',')]
    tmp = open(os.path.join(yang_keyshot_dir,'{}_keyShot.txt').format(video_name)).readlines()
    
    keyShots = [int(each) for each in tmp[0].split(',')]
    key_gt = []
    for i in range(len(keyShots)):
        key = keyShots[i]
        length = scene_boundary[i+1]-scene_boundary[i]
        key_gt += [key for i in range(length)]
    key_gt = torch.tensor(key_gt)
    return key_gt

def clean_gt(gt,start_shot,windowSize):
    for i in range(len(gt)):
        gt[i] = max(0,gt[i]-start_shot)
        gt[i] = min(windowSize-1,gt[i])

def fix_pred(gt,start_shot):
    for i in range(len(gt)):
        gt[i] = gt[i]+start_shot

def pred_scenes(pred,mask=5):
    """
    這邊有點複雜...會逐行說明
    Parameters
    ----------
    pred : torch.tensor
        pred are top 5 shot current shot attention to.
    mask : int, optional
        In pred, only care about the shot in range current index-mask to current index+mask The default is 8.
        換句話說，只有前後8個才會連線

    Returns
    -------
    boundary : list
        return scene boundary represented by shot index. (Not 0 and 1)

    """
    mask = 5
    pred_np = pred.detach().numpy()                                             ## 將pred轉為numpy array形式
    total_shot = len(pred_np) ## 總共的shot數
    #以下是讓每個shot 找到分數最高的shot並且進行連線
    
    links = []
    for i in range(total_shot):
        attention_to = pred_np[i]                                               ## pred_np[i] 是對於第i個shot而言"可能"連線的對象(候選人)    
        lower = max(0,i-mask)                                                   ## 對於第i個shot而言他只在乎i-8~i+8之間的候選人，當然i-8不得小於0，i+8不能超過總數
        upper = min(total_shot,i+mask)
        upper = upper+1
        noLink = True
        for each in attention_to:
            if each in range(lower,upper):                                      ## 如果他個候選人中有在-8~+8間就會與最高分的建立連線
                links.append((i,each))
                noLink = False
                break
        if noLink:
            links.append((i,i))                                                 ## 如果沒有就連到自己

    ## org 以下是找出scene
    scenes = []        
    for link in links:
        start = int(min(link))
        end = int(max(link))
        new_scene = [i for i in range(start,end+1)]                             ## 一個Scene是有連線的全部(? 
        isNew = True
        for i in range(len(scenes)):
            if start in scenes[i]:
                scenes[i] += [s for s in range(scenes[i][-1]+1,end+1)]          ## 如果這個連線是其他Scene的延伸，就加入那個Scene
                isNew = False
                break
        if isNew and len(new_scene)>0:
            scenes.append(new_scene)                                            ## 否則就使一個新的Scene
    del links

    #以下是將scene的最後一個當作boundary 儲存起來
    boundary = [total_shot-1]                                                   ## 規定最後一個Shot一定是Scene boundary
    tmp_point = 0
    for pred_scene in scenes:
        if pred_scene[-1]>tmp_point:
            if(pred_scene[-1] == total_shot):
                continue
            boundary.append(pred_scene[-1])                                     ## 每個Scene的最後一個Shot就是Scene的Boundary
            tmp_point = pred_scene[-1]
    del scenes
    
    if boundary[-1] != (total_shot-1): 
        boundary.append(total_shot-1)
        
    for i in range(len(boundary)-1,0,-1):
        if boundary[i-1] == boundary[i]-1:
            boundary = np.delete(boundary,i)
    
    boundary_list = boundary
    return np.array(boundary),boundary_list

def evaluate_window(label_dir,model,video_list,mask,windowSize,ground_dir,bbc=False):
    """
    驗證模型，看看模型的F-score，逐行說明
    """
    video_name = video_list[0]
    lossfun = nn.CrossEntropyLoss()
    visual_feature_dir = os.path.join(ground_dir,'parse_data')                  ## feature的路徑
    fscore = 0                                                                  ## 紀錄 Fscore            
    for video_name in video_list:                                            ## 因為每個Dataset都有很多影片，故要計算每個的平均
        label = load_keyShot(label_dir,video_name).to(device)
        visual_feature_path = os.path.join(visual_feature_dir,video_name)
        if not os.path.isdir(visual_feature_path):
            print('it is not')
            continue
        feature = load_feature(visual_feature_path).to(device)                ## 讀取 shot feature
        pred = torch.tensor([]).to(torch.device('cpu'))      
        batch_loss=0
        nbatch = int((feature.shape[0]-windowSize)/(windowSize-5)) + 2              #yang   windowsize = 15
        all_link_np = np.zeros((feature.shape[0],feature.shape[0]))
        value_tmp = []
        att_out_value_tmp = []
        tmp_tmp = []
        ii = 1
        for j in range(nbatch): #Devide the whole video into n batches, each batch with window size 15 
            start = j*windowSize
            end = (j+1)*windowSize
            if start > 0: # windowsize = 15, stride = 10
                start = start - 5*j
                end = end - 5*j
            
            end = min(end, feature.shape[0])
            src = feature[start:end]

            att_out = model(src)            
            value, tmp = torch.topk(att_out.view(-1,windowSize),5) ## 前面與training一樣，topk是指選高的前k個，因為att_out是對每個shot的分數，所以就是把對一個shot而言連到另一個分數高的shot
            fix_pred(tmp, start)## tmp給出的是相對的位置，因此實際的連線要加上開始的索引值

            new_value = []
            new_tmp = []
            for i in range(5):
                new_value.append([])
                new_tmp.append([])
            if end != feature.shape[0]: # if the current batch is not the last batch
                att_out = att_out.view(-1,windowSize)
                index = 14
                if len(value_tmp) == 0 :
                    value_tmp.append(value[10:15,:])
                    att_out_value_tmp.append(att_out[10:15,:])
                    tmp_tmp.append(tmp[10:15,])
                    for i in range(5):
                        tmp = tmp[torch.arange(tmp.size(0))!= index ]
                        value = value[torch.arange(value.size(0))!= index ]
                        att_out = att_out[torch.arange(att_out.size(0))!= index ] 
                        index = index - 1
                    final = [[0]*int(feature.shape[0]-(end)) for i in range(10)]
                    final = torch.tensor(final)
                    
                    att_out = att_out.to(torch.device('cpu'))
                    final = final.to(torch.device('cpu'))

                    att_out = torch.cat((att_out,final),1)
                    att_out = att_out[:,:] 
                    att_out = att_out.detach().numpy()  
                    
                    all_link_np[0:10,:] = att_out[:,:]
                    all_link_np = torch.tensor(all_link_np) 
                    all_link_np = all_link_np.to(torch.device('cpu'))
                else:  
                    value_tmp.append(value[0:5,:]) 
                    att_out_value_tmp.append(att_out[0:5,:])
                    tmp_tmp.append(tmp[0:5,])
                    for i in range(5):
                        for j in range(5):
                            for jj in range(5):
                                if value_tmp[0][i][j] > value_tmp[1][i][jj]:
                                    if tmp_tmp[0][i][j] in new_tmp[i]:
                                        pass
                                    else:
                                        new_tmp[i].append(tmp_tmp[0][i][j].item())
                                elif value_tmp[0][i][j] <= value_tmp[1][i][jj]:                                
                                    if tmp_tmp[1][i][jj] in new_tmp[i]:
                                        pass
                                    else:
                                        new_tmp[i].append(tmp_tmp[1][i][jj].item())                  
                                
                    for i in range(5):
                        for j in range(15):                                        
                            if att_out_value_tmp[0][i][j] > att_out_value_tmp[1][i][j]:
                                new_value[i].append(att_out_value_tmp[0][i][j].item())
                                
                            elif att_out_value_tmp[0][i][j] <= att_out_value_tmp[1][i][j]:                                
                                new_value[i].append(att_out_value_tmp[1][i][j].item())
                                             
                    del value_tmp[1]
                    del att_out_value_tmp[1]
                    del tmp_tmp[1]
                    value_tmp.append(value[10:15,])
                    att_out_value_tmp.append(att_out[10:15,]) 
                    tmp_tmp.append(tmp[10:15,])   
                    tmp_tmp[0] = tmp_tmp[1]
                    value_tmp[0] = value_tmp[1]
                    att_out_value_tmp[0] = att_out_value_tmp[1]
                    del value_tmp[1]
                    del att_out_value_tmp[1]
                    del tmp_tmp[1]
                    
                    for i in range(5):
                        new_value[i] = new_value[i][0:15]
                        new_tmp[i] = new_tmp[i][0:5]
                    for i in range(5):# 刪除10~14
                        value = value[torch.arange(value.size(0))!= index ]
                        att_out = att_out[torch.arange(att_out.size(0))!= index ]
                        tmp = tmp[torch.arange(tmp.size(0))!= index ] 
                        index = index - 1
                    for i in range(5):# 刪除 0~4
                        value = value[torch.arange(value.size(0))!= i ]
                        att_out = att_out[torch.arange(att_out.size(0))!= i ]
                        tmp = tmp[torch.arange(tmp.size(0))!= i ] 
                          
                    new_tmp = torch.tensor(new_tmp)
                    tmp = tmp.to(torch.device('cpu'))
                    new_tmp = new_tmp.to(torch.device('cpu'))
                    tmp = torch.cat((new_tmp,tmp))
                    
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
                      
            elif end == feature.shape[0]: # if the current batch is the last batch

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
                tmp = tmp.to(torch.device('cpu'))
                pred = torch.cat((pred,tmp)) 
                break
            tmp = tmp.to(torch.device('cpu'))
            pred = torch.cat((pred,tmp))

        gt_window = label[0:feature.shape[0]]
        gt_window = gt_window.to(torch.device('cpu'))
        try:
            lossout = lossfun(all_link_np,gt_window)
        except:
            print(f'{video_name}size not match, feature shape: {feature.shape[0]}, target size: {label.shape}')
        batch_loss = lossout.item()
        batch_loss = batch_loss
    
        boundary,boundary_list = pred_scenes(pred,mask=mask)## pred_scene會根據連線產生預測的scene boundary，pred_scene有進一步說明
    
        if bbc:
            score = co.fscore_eval_bbc(boundary, video_name) ## 計算Fscore
            fscore += score
        else:
            bgt_path = os.path.join(ground_dir,'label')
            score = co.fscore_eval(boundary, video_name,gt_path=bgt_path)
            fscore += score
        del gt_window
    fscore = fscore/len(video_list)
    return fscore, boundary,boundary_list,batch_loss

if __name__ == '__main__':
    dataset_dir = 'D:/UC_project/OVSD_Dataset/parse_data'
    video_list = os.listdir(dataset_dir)
    label_dir = "D:/UC_project/OVSD_Dataset/label"
    for video_name in video_list:
        video_name = 'Route_66'
        tmp = open(os.path.join(label_dir,'{}_shot.txt').format(video_name)).readlines()
        scene_boundary = [int(each) for each in tmp[0].split(',')]
        tmp = open(os.path.join(label_dir,'{}_keyShot.txt').format(video_name)).readlines()
        keyShots = [int(each) for each in tmp[0].split(',')]
        key_gt = []
        for i in range(len(keyShots)):
            key = keyShots[i]
            length = scene_boundary[i+1]-scene_boundary[i]
            key_gt += [key for i in range(length)]
        feature_dir = os.path.join(dataset_dir, video_name)
        total_shot = len(os.listdir(feature_dir))
        if len(key_gt) != total_shot:
            print(video_name)

        
    
    
