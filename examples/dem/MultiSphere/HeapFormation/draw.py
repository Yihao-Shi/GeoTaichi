import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams

params = {
             'backend': 'ps',
             'font.size': 26,
             'lines.linewidth': 4.5,
             'lines.markersize': 10,
             'xtick.labelsize': 26,
             'ytick.labelsize': 26,
             'xtick.major.pad': 12,
             'ytick.major.pad': 12,
             "axes.labelpad":   8,
             'legend.fontsize': 26,
             'figure.figsize': [12, 9],
             'font.family': 'serif',
             'text.usetex': True,
             'font.serif': 'Arial',
             'savefig.dpi': 300
         }
rcParams.update(params)

         
color = [(0/255, 0/255, 0/255), 
         (255/255, 0/255, 0/255), 
         (94/255, 114/255, 255/255), 
         (0/255, 128/255, 0/255)]
         
dt=1e-4
time=0.3

def read_file1(path):
    data=[]
    with open(path, 'r') as f:
    	for ann in f.readlines():
            data.append(time/dt/float(ann.strip('\n')))
            #print(float(ann))
    return data
    
data1=read_file1('case1/OutputData/time.txt')
data2=read_file1('case3/OutputData/time.txt')
data3=read_file1('case5/skin02/OutputData/time.txt')
data4=read_file1('case6/OutputData/time.txt')
data5=read_file1('case7/OutputData/time.txt')
t=[0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.3,3.6,3.9,4.2,4.5,4.8,5.1,5.4,5.7,6.0,6.3,6.6,6.9,7.2,7.5,7.8,8.1,8.4,8.7,9.0,
   9.3,9.6,9.9,10.2,10.5,10.8,11.1,11.4,11.7,12.0,12.3,12.6,12.9,13.2,13.5,13.8,14.1,14.4,14.7,15.0] 
plt.plot(t, data1, color=color[0], label="Particle number = 0.1 M")
plt.plot(t, data2, color=color[1], label="Particle number = 0.3 M")
plt.plot(t, data3, color=color[2], label="Particle number = 0.6 M")
plt.plot(t, data4, color=color[3], label="Particle number = 1 M")
plt.plot(t, data5, color='grey', label="Particle number = 1.6 M")
plt.xlim([0, 15.0])
plt.ylim([0., 1600])
plt.xlabel("Time (s)")
plt.ylabel('Speed (steps/s)')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig ("particle.eps")   
plt.close()

data1=read_file1('case5/skin0/OutputData/time.txt')
data2=read_file1('case5/skin01/OutputData/time.txt')
data3=read_file1('case5/skin02/OutputData/time.txt')
data4=read_file1('case5/skin03/OutputData/time.txt')
data5=read_file1('case5/skin04/OutputData/time.txt')
plt.plot(t, data1, color=color[0], label="Skin factor = 0")
plt.plot(t, data2, color=color[1], label="Skin factor = 0.1")
plt.plot(t, data3, color=color[2], label="Skin factor = 0.2")
plt.plot(t, data4, color=color[3], label="Skin factor = 0.3")
plt.plot(t, data5, color='grey', label="Skin factor = 0.4")
plt.xlim([0, 15.0])
plt.ylim([0., 400])
plt.xlabel("Time (s)")
plt.ylabel('Speed (steps/s)')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig ("skin.eps")   
plt.close()

def read_file2(path):
    data=0.
    with open(path, 'r') as f:
    	for ann in f.readlines():
            data+=float(ann.strip('\n'))
    return data

data11=read_file2('case1/OutputData/time.txt')
data12=read_file2('case3/OutputData/time.txt')
data13=read_file2('case5/skin02/OutputData/time.txt')
data14=read_file2('case6/OutputData/time.txt')
data15=read_file2('case7/OutputData/time.txt')

data21=read_file2('RTX2070/time1.txt')
data22=read_file2('RTX2070/time2.txt')
data23=read_file2('RTX2070/time3.txt')
data24=read_file2('RTX2070/time4.txt')
data25=read_file2('RTX2070/time5.txt')

data31=read_file2('RTX4080/time1.txt')
data32=read_file2('RTX4080/time2.txt')
data33=read_file2('RTX4080/time3.txt')
data34=read_file2('RTX4080/time4.txt')
data35=read_file2('RTX4080/time5.txt')

xaxis=[1e5, 3e5, 6e5, 1e6, 1.6e6]
y3070=[data11,data12,data13,data14,data15]
y2070=[data21,data22,data23,data24,data25]
y4080=[data31,data32,data33,data34,data35]
y3070=[150000/i for i in y3070]
y2070=[150000/i for i in y2070]
y4080=[150000/i for i in y4080]

plt.plot(xaxis, y2070, marker='o', linestyle='-', color=color[0], label="RTX 2070")
plt.plot(xaxis, y3070, marker='^', linestyle='--', color=color[1], label="RTX 3070")
plt.plot(xaxis, y4080, marker='*', linestyle='--', color=color[2], label="RTX 4080")
plt.ticklabel_format(axis='x',style='sci',scilimits=(0, 0))
plt.xlim([0, 1600000])
plt.ylim([0., 1800])
plt.xlabel("Particle number")
plt.ylabel('Speed (steps/s)')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig ("gpu.eps")   
plt.close()

data1=read_file2('case1/OutputData/time.txt')
data3=read_file2('case3/OutputData/time.txt')
data4=read_file2('case4/OutputData/time.txt')
data5=read_file2('case5/skin02/OutputData/time.txt')
data6=read_file2('case6/OutputData/time.txt')
data7=read_file2('case7/OutputData/time.txt')
xaxis1=[1e5, 2e5, 3e5, 4.8e5, 6e5, 1e6, 1.6e6, 3e6, 5e6, 6.25e6, 6.5e6]
xaxis2_1=[1e5, 2e5, 3e5, 4.8e5, 6e5, 8e5, 1.2e6, 1.6e6, 3.5e6]
xaxis2_2=[1e5, 2e5, 3e5, 4.8e5, 6e5, 8e5]
ygeotaichi=[data1,328,data3,data4,data5,data6,data7, 6912, 11722, 14759, 15890]
ymusen1=[1102., 1209, 1634., 2355., 2910, 4380, 8169, 12359, 32309]
ymusen2=[356., 589, 884., 1316., 1727, 2453]
ygeotaichi=[i/15. for i in ygeotaichi]
ymusen1=[i/15. for i in ymusen1]
ymusen2=[i/15. for i in ymusen2]


plt.plot(xaxis1, ygeotaichi, marker='o', linestyle='-', color=color[0], label="GeoTaichi")
plt.plot(xaxis2_1, ymusen1, marker='^', linestyle='--', color=color[1], label="MUSEN")
#plt.plot(xaxis2_2, ymusen2, marker='*', linestyle='--', color=color[2], label="MUSEN, skin factor=2")
plt.ticklabel_format(axis='x',style='sci',scilimits=(0, 0))
plt.xlim([0, 6250000])
plt.ylim([0., 2500])
plt.xlabel("Particle number")
plt.ylabel('Run time (s/$10^3$ steps)')
plt.legend(frameon=False)
plt.text(3200000, 1100, 'Average speedup: 3.37')
plt.annotate('', xy=(1000000,380), xytext=(1000000,125),arrowprops=dict(facecolor='blue',edgecolor='blue',arrowstyle='<-',linewidth=2))
plt.annotate('', xy=(1600000,776), xytext=(1600000,230),arrowprops=dict(facecolor='blue',edgecolor='blue',arrowstyle='<-',linewidth=2))
plt.annotate('', xy=(3000000,1775), xytext=(3000000,482),arrowprops=dict(facecolor='blue',edgecolor='blue',arrowstyle='<-',linewidth=2))
#plt.annotate("Out of GPU memory", xy=(3550000, 2100), xytext=(3400000, 1800), arrowprops=dict(arrowstyle="->",color='black'))
#plt.xscale("log")
#plt.yscale("log")
plt.tight_layout()
plt.savefig ("software.eps")   
plt.close()

xaxis1=[1e5, 2e5, 3e5, 4.8e5, 6e5, 1e6, 1.6e6, 3e6, 5e6, 6.25e6]
xaxis2_1=[1e5, 2e5, 3e5, 4.8e5, 6e5, 8e5, 1.2e6, 1.6e6, 3.5e6]
xaxis2_2=[1e5, 2e5, 3e5, 4.8e5, 6e5, 8e5]
data1=[292/1000, 389/1000, 488/1000, 657./1000, 780/1000, 1240/1000, 1920/1000, 3540/1000, 5870/1000, 7210/1000]
data2_1=[386./1024, 562/1024, 716./1024, 1052./1024, 1288/1024, 1692/1024, 2486/1024, 3350/1024, 7100/1024]
data2_2=[756./1024, 1664/1024, 2348./1024, 3780./1024, 5177/1024, 7124/1024]
plt.plot(xaxis1, data1, marker='o', linestyle='-', color=color[0], label="GeoTaichi")
plt.plot(xaxis2_1, data2_1, marker='^', linestyle='--', color=color[1], label="MUSEN")
plt.plot(np.arange(0, 6250001, 6250000), np.repeat(7.35, 2), linestyle='-.', color=color[2])
#plt.plot(xaxis2_2, data2_2, marker='*', linestyle='--', color=color[2], label="MUSEN, skin factor=2")
plt.ticklabel_format(axis='x',style='sci',scilimits=(0, 0))
plt.xlim([0, 6250000])
plt.ylim([0, 8])
plt.xlabel("Particle number")
plt.ylabel('Memory usage (GB)')
plt.text(200000, 6.9, 'Upper bound of GPU memory')
plt.legend(frameon=False)
plt.text(3190000, 2.9, 'Average memory saving: 0.38')
plt.annotate('', xy=(1000000,1.96), xytext=(1000000,1.25),arrowprops=dict(facecolor='blue',edgecolor='blue',arrowstyle='<-',linewidth=2))
plt.annotate('', xy=(1600000,3.21), xytext=(1600000,1.92),arrowprops=dict(facecolor='blue',edgecolor='blue',arrowstyle='<-',linewidth=2))
plt.annotate('', xy=(3000000,5.86), xytext=(3000000,3.61),arrowprops=dict(facecolor='blue',edgecolor='blue',arrowstyle='<-',linewidth=2))
#plt.xscale("log")
#plt.yscale("log")
plt.tight_layout()
plt.savefig ("memory.eps")   
plt.close()
