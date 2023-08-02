import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import math
from pathlib import Path
import pandas as pd

folder = "25072023-1610"
shift = 0
divide = 6
level = 1

# depths = []
# for index in range(divide):
#     # depth = np.load("output/{0}/{2}/match_diff/{0}-depth-{1}-{2}.npy".format(folder,index,shift))
#     depth = np.load("output/{0}/{2}/first_depth/{0}-depth-{1}-{2}.npy".format(folder,index,shift))
#     depths.append(depth)
# width = sum(depths[i].shape[1] for i in range(divide))
# print(width)

data=[]
point_distance_file = pd.read_csv('output/'+folder+'/distance.csv')
# print(point_distance_file)
for _,row in point_distance_file.iterrows():
    depth = np.load("output/{0}/one/{1}.npy".format(folder,int(row['Angle'])))
    # xk = round(3.04/2.76*384*3.75/row['Distance']*2+depth.shape[1]/2)
    # yk = round(3.04/2.76*384*0.9/row['Distance']+depth.shape[0]/2)
    # if (int(row['Angle']) > 513 or int(row['Angle']) <38):
    print(row['Angle'],depth[int(row['y']),int(row['x'])],row['Distance']/100)
    data.append([depth[int(row['y']),int(row['x'])],row['Distance']/100])
    
# x = [2.40,1.15,2.50,2.35,2.67,3.06,1.42,2.10,2.03,3.30,1.94,1.28]
# y = [1.50,0.60,1.83,2.03,2.23,2.50,0.50,1.30,1.75,3.30,1.00,0.70]
# print(data)
data = np.array(data)
data = data[data[:,0].argsort()]
coeff = np.polyfit(data[:,0],data[:,1],level)
print(coeff)
y_new = np.polyval(coeff,data[:,0])
dis = abs(y_new - data[:,1])


# print(dis)
# print(data[:,1]-data[:,0])
# print(y_new-data[:,0])
print(dis.mean(),dis.max(),dis.min())
# plt.scatter(data[0],abs(data[1]-data[0]))
# plt.scatter(data[0],abs(y_new-data[0]))
plt.scatter(data[:,0],data[:,1])
plt.plot(data[:,0],y_new)
plt.show()

# Save
# output_directory = Path('output/'+folder+'/'+str(shift)+'/calib_param')
# output_directory.mkdir(parents=True,exist_ok=True)
# for index in range(len(depths)):
#     np.save("output/{0}/{2}/calib_param/{0}-depth-{1}-{2}.npy".format(folder,index,shift),np.polyval(coeff,depths[index]))
#     print("save "+str(index)) 