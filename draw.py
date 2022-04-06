import pandas as pd
import matplotlib.pyplot as plt
filename='results\\15\log\\ear_lr0.005_lam0.0001.log'
file=open(filename)
# filename_att='results\\15\log\\result_attention_test.log'
# file_att=open(filename_att)
x=[]
y=[]
x_att=[]
y_att=[]
i=0
for line in file.readlines():
    if 'train' in line:
        i=i+1
        lam=line.split("train:")[1].split(',')[0]
        loss=line.split("Loss:")[1].split(',')[0]
    
        x.append(float(lam))
        y.append(float(loss))
# for line in file_att.readlines():
#     i=i+1
#     loss_att=line.split("RMSE:")[1].split(',')[0]
#     lam_att=line.split("proportion used: ")[1].split('%')[0]
#     x_att.append(float(lam_att))
#     y_att.append(float(loss_att))
plt.plot(x,y,marker='o',label='ST-Fixed')
# plt.plot(x_att,y_att,marker='v',label='Att-ST-Earliest')
plt.legend()
plt.xlabel('TIME')
plt.ylabel('RMSE')
plt.show()