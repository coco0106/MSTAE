import os
import matplotlib.pyplot as plt
filePath='results\\90\\log\\'
plt.rcParams['figure.figsize'] = (9, 9)
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for filename in os.listdir(filePath):
    file=open(filePath+filename)
    x=[]
    y=[]
    
    for line in file.readlines():
        if 'train' in line:
            lam=line.split("train:")[1].split(',')[0]
            
            loss=line.split("MAE:")[1].split(',')[0]
            x.append(float(lam))
            y.append(float(loss))
    plt.plot(x,y,label=filename.split('.log')[0],linewidth=2.5)

plt.legend(fontsize=15)

plt.xlabel('Time-step observed percentage',fontsize=25)
plt.ylabel('MAE',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(alpha=0.5)
plt.savefig("all.pdf")
# plt.show()