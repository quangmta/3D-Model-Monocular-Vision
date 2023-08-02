import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import math
from pathlib import Path
import pandas as pd

folder = "27072023-1628"
shift = 250
divide = 6
level = 1

depths = []
for index in range(divide):
    # depth = np.load("output/{0}/{2}/match_diff/{0}-depth-{1}-{2}.npy".format(folder,index,shift))
    depth = np.load("output/{0}/{2}/first_depth/{0}-depth-{1}-{2}.npy".format(folder,index,shift))
    depths.append(depth)
width = sum(depths[i].shape[1] for i in range(divide))
# print(width)

data=[]
point_distance_file = pd.read_csv('output/'+folder+'/distance.csv')
# print(point_distance_file)
for _,row in point_distance_file.iterrows():
    if int(row['Angle']) in {164,736,322}:
        continue
    x_delta = row['xp']-shift if row['xp']-shift>=0 else row['xp']-shift+width
    part_number = int(x_delta/(width/divide))
    col_part = int(x_delta-part_number*width/divide)    
    # if not (int(row['Angle']) > 513 or int(row['Angle']) <38):
    print(row['Angle'],row['xp'],depths[part_number][int(row['yp']),col_part],row['Distance']/100)
    data.append([depths[part_number][int(row['yp']),col_part],row['Distance']/100])
    
    # # draw circle    
    # image_i = cv2.imread('input/'+folder+"/"+str(int(row['Angle']))+"/l.jpg")
    # image_i = cv2.resize(image_i,(511,384),interpolation = cv2.INTER_AREA)
    
    # img_new = depths[part_number].copy()
    # # img_new = cv2.imread("output/{0}/{1}/img/{0}-{2}-{1}.png".format(folder,shift,part_number))
    # # img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
        
    # cv2.circle(img_new, (col_part,int(row['yp'])), 5, (0, 0, 255), thickness=2)
    # cv2.circle(image_i, (int(row['x']),int(row['y'])), 5, (0, 0, 255), thickness=2)
    
    # fig = plt.figure(figsize=(12, 6))
    # ax = fig.add_subplot(1, 2, 2)
    # ax.imshow(cv2.cvtColor(image_i, cv2.COLOR_BGR2RGB))
    # ax.set_title(str(int(row['Angle']))+' original',fontsize=10)
    
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(img_new,cmap='jet_r')
    # ax.set_title(str(part_number)+' panorama',fontsize=10)
    
    # fig.tight_layout()
    # plt.show()
    
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
output_directory = Path('output/'+folder+'/'+str(shift)+'/calib_param')
output_directory.mkdir(parents=True,exist_ok=True)
for index in range(len(depths)):
    np.save("output/{0}/{2}/calib_param/{0}-depth-{1}-{2}.npy".format(folder,index,shift),np.polyval(coeff,depths[index]))
    print("save "+str(index)) 