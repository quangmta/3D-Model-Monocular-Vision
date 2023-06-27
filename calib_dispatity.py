import numpy as np
from matplotlib import pyplot as plt
import cv2

folder = "12052023-1348"
depths = []
images = []
# sift = "229"
coeff = [ 1.19410183, -1.003789  ]
epsilon = 50
#Caculate color duplication
def calib_hoz_left(images,color_threshold):
    dups=[]
    for index in range(len(images)):
        img = cv2.cvtColor(images[index], cv2.COLOR_BGR2GRAY)
        dup = []
        for i in range(len(img[0])):
            count = 0
            for j in range(1,len(img)):
                if abs(img[j,i]-img[i,0])<=color_threshold:
                    count+=1
                else:
                    dup.append(count)
                    break
        print(dup)
        dups.append(dup)
        cv2.imshow(str(index),img)
        cv2.waitKey(0)
    return dups    
def calib_hoz_right(images,color_threshold):
    dups=[]
    for index in range(len(images)):
        img = cv2.cvtColor(images[index], cv2.COLOR_BGR2GRAY)
        dup = []
        for i in range(len(img[0])):
            count = 0
            for j in range(-2,-len(img),-1):
                if abs(img[j,i]-img[i,-1])<=color_threshold:
                    count+=1
                else:
                    dup.append(count)
                    break
        print(dup)
        dups.append(dup)
    return dups        
            
for index in range(6):
    img = cv2.imread("output/match/None/"+folder+"-"+str(index)+".png",cv2.IMREAD_COLOR)
    depth = np.load("output/match/None/"+folder+"-depth-"+str(index)+".npy")
    depth = np.polyval(coeff,depth)
    depths.append(depth)
    images.append(img)
depths_new = [row.copy() for row in depths]
for index in range(6):
    d = depths[index][:,0]-depths[index-1][:,-1]
    m = (depths[index][:,0]+depths[index-1][:,-1])/2
    print(m.shape)
    center = int(len(depths[index][0])/2)
    # delta = m-depths[index][:,0]
    for i in range(0,center):        
        depths_new[index][:,i] += -d/2*(1-i/center)
    center = int(len(depths[index-1][0])/2)
    for i in range(-1,-center,-1):
        depths_new[index-1][:,i] += d/2*(1-abs(i+1)/center)
    # print(depths_new[index][:,0]-depths_new[index-1][:,-1])
    print(d.min(),d.max(),d.mean())
    plt.title(str(index))
    plt.plot(np.arange(0,len(d),1),d,'y')
    dy = [d[i+1]-d[i] for i in range(len(d)-1)]
    plt.plot(np.arange(0,len(d),1),depths[index][:,0],'g')
    plt.plot(np.arange(0,len(d),1),depths[index-1][:,-1],'b')
    plt.plot(np.arange(0,len(d),1),(depths[index][:,0]+depths[index-1][:,-1])/2,'black')
    plt.plot(np.arange(0,len(dy),1),dy,'r')
    plt.show()

for index in range(6):
    np.save("output/match/None/calib/"+folder+"-depth-"+str(index)+".npy",depths_new[index])
    print("save "+str(index))