from pkgutil import iter_modules
from xml.etree.ElementPath import prepare_self
import matplotlib.pyplot as plt
import os

# mask = 7
predict_file = 'final_result/ovsd_result/boundary/ovsd_train.txt'
# gary_file = 'boundary/BBC_Dataset_UCF101 and HMDB_clip_len=16_bound.txt'

gt_path = 'final_result/ovsd_result/gt'
gt_file_list = os.listdir(gt_path)
print(gt_file_list)
all_gt_boundary=[]
for gt_file in gt_file_list:
    with open(os.path.join(gt_path, gt_file)) as f:
        line = f.readline()
        gt_boundary = line.split(',')
        gt_boundary = [int(item) for item in gt_boundary]
        all_gt_boundary.append(gt_boundary)
    f.close()

all_predict_boundary=[]
with open(predict_file) as f:
    lines = f.readlines()
    for line in lines:
        video = line.split(' ')
        predict_boundary = [int(item) for item in video[2:]]
        all_predict_boundary.append(predict_boundary)
f.close()

for video_index in range(len(all_gt_boundary)):
    f, ax = plt.subplots(figsize=(10,5))
    gt = all_gt_boundary[video_index]
    predict = all_predict_boundary[video_index]
    plt.title('OCSD video '+str(video_index+1))
    # plt.title('BBC video '+str(video_index+1)+', mask = '+str(mask))
    plt.bar(gt, 1)
    plt.bar(predict, -1)
    plt.legend(['gt boundarys: '+str(len(gt)), 'predict boundarys: '+str(len(predict))], loc='upper right')
    plt.xlabel('shot number')
    plt.yticks([])
    plt.savefig('final_result/ovsd_result/boundary_plot/OVSD_video'+str(video_index+1)+'.jpg')
    # plt.savefig('boundary_plot/BBC_video'+str(video_index+1)+'_mask'+str(mask)+'.jpg')