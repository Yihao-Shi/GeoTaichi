import numpy as np

data=[]
with open('OutputData/time.txt', 'r') as f:
    for ann in f.readlines():
        data.append(float(ann.strip('\n')))

print(np.array(data).sum())
