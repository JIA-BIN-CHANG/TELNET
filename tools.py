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

with open('./config_ovsd.json','r') as f:
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
    # yang_keyshot_dir = "./yang_bbc_keyshot/keyshot"
    tmp = open(os.path.join(label_dir,'{}_shot.txt').format(video_name)).readlines()
    scene_boundary = [int(each) for each in tmp[0].split(',')]
    tmp = open(os.path.join(label_dir,'{}_keyShot.txt').format(video_name)).readlines()
    keyShots = [int(each) for each in tmp[0].split(',')]
    if len(keyShots) != len(scene_boundary)-1:
        # print(f'{video_name} key shot and boundary not match')
        return None
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
    Parameters
    ----------
    pred : torch.tensor
        pred are top 5 shot current shot attention to.
    mask : int, optional
        In pred, only care about the shot in range current index-mask to current index+mask The default is 8.
        ???????????????????????????10???????????????

    Returns
    -------
    boundary : list
        return scene boundary represented by shot index. (Not 0 and 1)

    """
    mask = 5
    pred_np = pred.detach().numpy()                                             ## ???pred??????numpy array??????
    total_shot = len(pred_np) ## total shots
    #??????????????????shot ?????????????????????shot??????????????????
    
    links = []
    for i in range(total_shot):
        attention_to = pred_np[i]                                               ## pred_np[i] ????????????i???shot??????"??????"???????????????(?????????)    
        lower = max(0,i-mask)                                                   ## ?????????i???shot??????????????????i-5~i+5???????????????????????????i-5????????????0???i+5??????????????????
        upper = min(total_shot,i+mask)
        upper = upper+1
        noLink = True
        for each in attention_to:
            if each in range(lower,upper):                                      ## ??????????????????????????????-5~+5????????????????????????????????????
                links.append((i,each))
                noLink = False
                break
        if noLink:
            links.append((i,i))                                                 ## ???????????????????????????

    ##???????????????scene
    scenes = []        
    for link in links:
        start = int(min(link))
        end = int(max(link))
        new_scene = [i for i in range(start,end+1)]                             ## ??????Scene?????????????????????(? 
        isNew = True
        for i in range(len(scenes)):
            if start in scenes[i]:
                scenes[i] += [s for s in range(scenes[i][-1]+1,end+1)]          ## ???????????????????????????Scene???????????????????????????Scene
                isNew = False
                break
        if isNew and len(new_scene)>0:
            scenes.append(new_scene)                                            ## ????????????????????????Scene
    del links

    #????????????scene?????????????????????boundary ????????????
    boundary = [total_shot-1]                                                   ## ??????????????????Shot?????????Scene boundary
    tmp_point = 0
    for pred_scene in scenes:
        if pred_scene[-1]>tmp_point:
            if(pred_scene[-1] == total_shot):
                continue
            boundary.append(pred_scene[-1])                                     ## ??????Scene???????????????Shot??????Scene???Boundary
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
    ??????????????????????????????F-score???????????????
    """
    video_name = video_list[0]
    lossfun = nn.CrossEntropyLoss()
    visual_feature_dir = os.path.join(ground_dir,'parse_data')
    fscore = 0
    for video_name in video_list:
        label = load_keyShot(label_dir,video_name)
        if label is None:
            continue
        label = label.to(device)
        visual_feature_path = os.path.join(visual_feature_dir,video_name)
        if not os.path.isdir(visual_feature_path):
            continue
        feature = load_feature(visual_feature_path).to(device)
        pred = torch.tensor([]).to(torch.device('cpu'))      
        batch_loss=0
        #windowsize = 15
        nbatch = int((feature.shape[0]-windowSize)/(windowSize-5)) + 2
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
            value, tmp = torch.topk(att_out.view(-1,windowSize),5) #topk candidates
            fix_pred(tmp, start)#add window offset to tmp
            tmp = tmp.to(torch.device('cpu'))

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
                    att_out = torch.cat((att_out,final),1)
                    att_out = att_out[:,:] 
                    att_out = att_out.detach().numpy()  
                    
                    all_link_np[0:10,:] = att_out[:,:]
                    all_link_np = torch.tensor(all_link_np) 
                else:  
                    value_tmp.append(value[0:5,]) 
                    att_out_value_tmp.append(att_out[0:5,])
                    tmp_tmp.append(tmp[0:5,])

                    for shot_index in range(5):
                        for window_cur_top5 in range(5):
                            for window_next_top5 in range(5):
                                if value_tmp[0][shot_index][window_cur_top5] > value_tmp[1][shot_index][window_next_top5]:
                                    if tmp_tmp[0][shot_index][window_cur_top5] in new_tmp[shot_index]:
                                        pass
                                    else:
                                        new_tmp[shot_index].append(tmp_tmp[0][shot_index][window_cur_top5].item())
                                elif value_tmp[0][shot_index][window_cur_top5] <= value_tmp[1][shot_index][window_next_top5]:                                
                                    if tmp_tmp[1][shot_index][window_next_top5] in new_tmp[shot_index]:
                                        pass
                                    else:
                                        new_tmp[shot_index].append(tmp_tmp[1][shot_index][window_next_top5].item())             
                    #calculate test loss
                    for shot_index in range(5):
                        for shot in range(10):
                            new_value[shot_index].append(att_out_value_tmp[0][shot_index][shot].item())
                        for candidate_index in range(5):
                            if att_out_value_tmp[0][shot_index][10+candidate_index] > att_out_value_tmp[1][shot_index][candidate_index]:
                                new_value[shot_index].append(att_out_value_tmp[0][shot_index][candidate_index].item())
                            elif att_out_value_tmp[0][shot_index][10+candidate_index] <= att_out_value_tmp[1][shot_index][candidate_index]:
                                new_value[shot_index].append(att_out_value_tmp[1][shot_index][candidate_index].item())
                        for shot in range(5,15):
                            new_value[shot_index].append(att_out_value_tmp[1][shot_index][shot].item())     
                                             
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
                    for i in range(5):# ??????10~14
                        value = value[torch.arange(value.size(0))!= index ]
                        att_out = att_out[torch.arange(att_out.size(0))!= index ]
                        tmp = tmp[torch.arange(tmp.size(0))!= index ] 
                        index = index - 1
                    for i in range(5):# ?????? 0~4
                        value = value[torch.arange(value.size(0))!= i ]
                        att_out = att_out[torch.arange(att_out.size(0))!= i ]
                        tmp = tmp[torch.arange(tmp.size(0))!= i ] 
                          
                    new_tmp = torch.tensor(new_tmp)
                    tmp = torch.cat((new_tmp,tmp))         
                    new_value = torch.tensor(new_value)     
                    
                    att_out = att_out.to(torch.device('cpu'))
                    att_out = torch.cat((new_value,att_out))

                    begin = [[0]*start for i in range(10)]
                    final = [[0]*int(feature.shape[0]-(end)) for i in range(10)]               
                    begin = torch.tensor(begin) 
                    final = torch.tensor(final)
                    
                    att_out = att_out.clone().detach()
                    att_out = torch.cat((begin,att_out),1)
                    att_out = torch.cat((att_out,final),1)
                    att_out = att_out[:,:]
                    all_link_np[ii*10:(ii+1)*10,:] = att_out[:,:]

                    ii += 1
                      
            elif end == feature.shape[0]: # if the current batch is the last batch

                att_out = att_out.view(-1,(feature.shape[0]-start))
                begin = [[0]*(start) for i in range(feature.shape[0]-start)]

                att_out = att_out.clone().detach()                                   
                begin = torch.tensor(begin)     

                att_out = att_out[0:feature.shape[0]-start]
                att_out = att_out.to(torch.device('cpu'))                         
                att_out = torch.cat((begin,att_out),1)
                all_link_np[start:feature.shape[0],:] = att_out[:,:]
                pred = torch.cat((pred,tmp))
                break

            pred = torch.cat((pred,tmp))

        gt_window = label[0:feature.shape[0]].to(torch.device('cpu'))

        try:
            lossout = lossfun(all_link_np,gt_window)
        except:
            print(f'{video_name}size not match, feature shape: {feature.shape[0]}, target size: {label.shape}')
        batch_loss = lossout.item()
        batch_loss = batch_loss
    
        boundary,boundary_list = pred_scenes(pred,mask=mask)## pred_scene??????????????????????????????scene boundary???pred_scene??????????????????
    
        if bbc:
            score = co.fscore_eval_bbc(boundary, video_name) ## ??????Fscore
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

        
    
    
