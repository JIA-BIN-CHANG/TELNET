# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:10:27 2021
這裡的寫法很詭異，在coverOverflow中，src是一堆0跟，1，但tgt是shot index
but anyway現在會過可以用，有空要整理一下
@author: NTUT
"""

from genericpath import isfile
import os
import numpy as np


def load_bgt(gt_dir,video_name,gt_name=None):
    """

    Parameters
    ----------
    video_name : str
        groundtruth file are a series of 0 and 1. 1 is represented for boundary

    Returns
    -------
    gt : np.array

    """
    if gt_name is None:
        gt_name = os.path.join(gt_dir,video_name+'_BGT.txt')
    if not os.path.isfile(gt_name):
        return None
    tmp = open(gt_name,'r').readlines()
    tmp = [each.replace('\n','').replace('[','').replace(']','') for each in tmp]
    gt = []
    for each in tmp[0].split(','):
        gt.append(int(each))
    gt = np.array(gt)
    return gt
  
def convert(gt):
    """
    It will convert a sequence of 0,1 to shot index
    Parameters
    ----------
    gt : np.array
        gt can be either groundtruth ot prediction.

    Returns
    -------
    new_gt: np.array
        Boundary shot index will be presented in new_gt
    """
    
    boundary = []
    for i in range(len(gt)):
        if gt[i] == 1:
            boundary.append(i)
    return np.array(boundary)

def modification(boundary):
    """
    This function is to make sure the last shot be one of scene boundary
    Parameters
    ----------
    boundary : np.array
        prediction boundary

    Returns
    -------
    None.

    """
    boundary[-1] = 1
    
def createScene(src):
    shotInScene = []
    if len(src) <= 1:
        shotInScene = [i for i in range(src[-1]+1)]
        return [shotInScene]
    for i in range(len(src)-1):
        if i == 0:
            start_shot = 0
            end_shot = src[i+1]
        else:
            start_shot = src[i]
            end_shot = src[i+1]
        scene = [shot for shot in range(start_shot,end_shot)]
        shotInScene.append(scene)
    shotInScene[-1].append(src[-1])
    return shotInScene

def scoring(coverage,overflow,printed=True,string=None):
    fscore = []
    for c, o in zip(coverage,overflow):
        if o == 1:
            fscore.append(0)
        elif c == 0:
            c = 1
        else:
            denominator = 1/c+1/(1-o)
            fscore.append(2/denominator)
    fscore = np.array(fscore)
    if printed:
        if string is None:
            raise ValueError("Attribute String are required for print.")
        else:
            print('{}\tF_Score is: {}'.format(string,np.mean(fscore)))
    return np.mean(fscore) 

def coverOverflow(src,tgt):
    """
    

    Parameters
    ----------
    src : np.array
        Groundtruth represented by sequence of 0 and 1. src will first modifiy to make sure last shot be scene boundary
        Then convert to shot index. After give shot index create scene.
    tgt : np.array
        Prediction represented by shot index. The Last shot has be identify as scene boundary.

    Returns
    -------
    coverage : list
        Coverage for each scene.
    overflow : list 
        Overflow for each scene.

    """
    modification(src)
    src = createScene(convert(src))
    tgt = createScene(tgt)
    
    coverage = []
    overflow = []
    for i in range(len(src)):
        gt_scene = src[i]
        start = gt_scene[0]
        nShot = len(gt_scene)
        overlap = 0
        for pred_scene in tgt:
            if pred_scene[-1]<start:
                continue
            else:
                tmp = list(set(gt_scene)&set(pred_scene))
                if len(tmp)>overlap:
                    overlap = max(len(tmp),1)
                    candidate = pred_scene    
        coverage.append(overlap/nShot)
        if i == 0:
            gt_prev = []
            try:
                gt_next = src[1]
            except:
                gt_next = []
        elif i == len(src)-1:
            gt_prev = src[i-1]
            gt_next = []
        else:
            gt_prev = src[i-1]
            gt_next = src[i+1]       
        if (len(gt_prev)+len(gt_next)) != 0:
            tmp = list(set(gt_prev)&set(candidate))+list(set(gt_next)&set(candidate))
            overflow.append(len(tmp)/(len(gt_prev)+len(gt_next)))
        else:
            overflow.append(0)
        
    return coverage, overflow
        
def fscore_eval_bbc(boundary,video_name,printed=True,coverover=False):
    """
    Evaluting fscore for one video. 
    It will calculate will annotator and return the best score for this video.
    Parameters
    ----------
    boundary : np.array
        Prediction boundary, represented in shot index
    video_name : str
        video name.
    printed : bool
        Print f_score if True, Optional, Default is True
    coverover: bool
        return coverage and overflow if True, Optional, Default is False 
    Returns
    -------
    cover : list
        Coverage for each scene, optional, return if coverover is True
    over : list
        Overflow for each scene, optional, return if coverover is True
    score : float
        
    """
    score = 0
    cover = []
    over = []
    for i in range(5):
     
        gt_path = os.path.join('D:/UC_project/BBC_Earth_Dataset','annotations/scenes/annotator_'+str(i))
        bgt = load_bgt(gt_path,video_name)
        
        cover_shot, over_shot = coverOverflow(bgt, boundary)
        tmp = scoring(cover_shot,over_shot,printed=False)
        cover.append(cover_shot)
        over.append(over_shot)
        if tmp > score:
            score = tmp
            cover_score = cover_shot
            over_score = over_shot
            annotator = i
    if printed:
        print('{}\t best fscore is {}\t annotator is {}'.format(video_name,score,annotator))
    if coverover:
        return np.array(cover_score), np.array(over_score), score
    return score

def fscore_eval(boundary,video_name,gt_path,printed=True,coverover=False,bgt_path=None):
    """
    Evaluting fscore for one video. 
    It will calculate will annotator and return the best score for this video.
    Parameters
    ----------
    boundary : np.array
        Prediction boundary, represented in shot index
    video_name : str
        video name.
    gt_path : str
        groundTruth path, gt must be bgt, a sequence of 0 and 1
    printed : bool
        Print f_score if True, Optional, Default is True
    coverover: bool
        return coverage and overflow if True, Optional, Default is False 
    Returns
    -------
    cover : list
        Coverage for each scene, optional, return if coverover is True
    over : list
        Overflow for each scene, optional, return if coverover is True
    score : float
        
    """
    score = 0
    cover = []
    over = []
    if bgt_path is None:
        bgt = load_bgt(gt_path,video_name)
    else:
        bgt = load_bgt(gt_path, video_name, bgt_path)
    if bgt is None:
        return np.nan
    cover_shot, over_shot = coverOverflow(bgt, boundary)
    tmp = scoring(cover_shot,over_shot,printed=False)
    cover.append(cover_shot)
    over.append(over_shot)
    
    if tmp > score:
        score = tmp
        cover_score = cover_shot
        over_score = over_shot
    if printed:
        print('{}\t best fscore is {}'.format(video_name,score))
    if coverover:
        return np.array(cover_score), np.array(over_score), score
    return score

def eval_test_all_boundary():
    video_list = ['01_From_Pole_to_Pole','02_Mountains','03_Ice_Worlds','04_Great_Plains','05_Jungles','06_Seasonal_Forests','07_Fresh_Water',
                  '08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']
    tmp = 0
    for video_name in video_list:
        shot_folder = os.path.join('../bbc_dataset_video',video_name)
        nShots = len(os.listdir(shot_folder))    
        boundary = np.array([i for i in range(nShots)])
        score = fscore_eval_bbc(boundary,video_name,printed=True)
        tmp += score
    tmp = tmp/len(video_list)
    print('Guess all shot are boundary fscore: {}'.format(tmp))
        
        
if __name__ == '__main__'  :
    video_list = ['01_From_Pole_to_Pole','02_Mountains','03_Ice_Worlds','04_Great_Plains','05_Jungles','06_Seasonal_Forests',
                  '08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']
    
    eval_test_all_boundary()

        
            
