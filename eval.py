# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:25:47 2022
這個是用來測試訓練好的模型

@author: Yuuki Misaki
"""

import os
import json
import torch
import numpy as np

import coverage_overflow as co
import tools
from model.TELNet_Model import TELNet_model
import argparse

def eval_model(model, config):
    eval_list = config['eval_list']
    window_size = config['window_size']
    gt_dir = config['label_dir']
    
    fscore_dict = {}
    for eval_dataset in eval_list:
        fscore = 0
        dataset_dir = os.path.join(eval_dataset, './parse_data')
        if 'BBC' in dataset_dir:
            isBBC = True
        else:
            isBBC = False
            
        video_list = os.listdir(dataset_dir)
        for video_name in video_list:
            feature_dir = os.path.join(dataset_dir, video_name)
            nShot = len(os.listdir(feature_dir))
            nbatch = int(nShot/window_size)+1
            feature = tools.load_feature(feature_dir)
            pred = torch.tensor([])
            value_tmp = []
            att_out_value_tmp = []
            tmp_tmp = []
            all_link_np = np.zeros((feature.shape[0],feature.shape[0]))
            ii = 1
            for j in range(nbatch):
                start = j*window_size
                end = (j+1)*window_size
                if start > 0: # window_size = 15, stride = 10
                    start = start - 5*j
                    end = end - 5*j
                
                end = min(end, feature.shape[0])
                src = feature[start:end]

                att_out = model(src)            
                value, tmp = torch.topk(att_out.view(-1,window_size),5) ## 前面與training一樣，topk是指選高的前k個，因為att_out是對每個shot的分數，所以就是把對一個shot而言連到另一個分數高的shot
                tools.fix_pred(tmp, start)## tmp給出的是相對的位置，因此實際的連線要加上開始的索引值

                new_value = []
                new_tmp = []
                for i in range(5):
                    new_value.append([])
                    new_tmp.append([])
                if end != feature.shape[0]: # if the current batch is not the last batch
                    att_out = att_out.view(-1,window_size)
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
            boundary, _ = tools.pred_scenes(pred)
            if isBBC:
                score = co.fscore_eval_bbc(boundary, video_name)
            else:
                score = co.fscore_eval(boundary, video_name, gt_dir)
            if score is np.nan:
                pass
            else:
                fscore += score
        fscore /= len(video_list)
        fscore_dict.update({eval_dataset:fscore})
    return fscore_dict
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        help="Use it to select config file")
    args = parser.parse_args()

    config_path = 'config_cross_{}.json'.format(args.dataset)
    with open(config_path) as f:
        config = json.load(f)
    model_path = config['trained_model_path']
    model = TELNet_model(config['feature_dim'], config['window_size'])
    model.load_state_dict(torch.load(model_path))
    
    fscore_dict = eval_model(model, config)
    
    for key in list(fscore_dict.keys()):
        print(f'{key} fscore: {fscore_dict[key]}')    

