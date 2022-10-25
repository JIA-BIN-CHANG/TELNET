import matplotlib.pyplot as plt

score_file='f1/BBC_Dataset_sports1M epoches 100 mask=4 merge_window=15.txt'

all_content=[]
with open(score_file) as f:
    lines = f.readlines()
    for line in lines:
        video = line.split(' ')
        all_content.append(video)
f.close()

for index in range(len(all_content)):
    all_content[index][1] = round(float(all_content[index][1][:10]), 2)

#define figure and axes
fig, ax = plt.subplots(figsize=(10,10))

#create table
table = ax.table(cellText=all_content, loc='center')

#modify table
table.set_fontsize(14)
table.scale(1,4)
ax.axis('off')

#display table
plt.show()

