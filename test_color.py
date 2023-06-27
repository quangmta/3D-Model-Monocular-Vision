import cv2
import numpy as np
from matplotlib import pyplot as plt


folder = "12052023-1348"

# sift = "229"
coeff = [ 1.19410183, -1.003789  ]
epsilon = 50

def calib_hoz_left(images,color_threshold,stepx,stepy):
    dups=[]
    for index in range(len(images)):
        img = cv2.cvtColor(images[index], cv2.COLOR_BGR2GRAY)
        # img_ = cv2.cvtColor(images[index-1], cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        dup = []
        for i in range(0,len(img),stepy):
            bottom = i+stepy if i+stepy<=len(img) else len(img)
            color_l = img[i:bottom,0:stepx:].mean()
            count = 0
            for j in range(stepx,len(img[0]),stepx):
                right = j+stepx if j+stepx<=len(img[0]) else len(img[0])
                color_s = img[i:bottom,j:right].mean()
                if abs(color_s-color_l)<=color_threshold:
                    count+=stepx
                else:
                    # dup.append(count)
                    break
            for j in range(stepy):
                dup.append(count)
        # print("Left "+str(index))
        # for i in range(0,len(dup),stepy):
        #     print(i,dup[i])
        dups.append(dup)
        # plt.title(str(index))
        # plt.imshow(img,cmap='gray')
        # plt.show()
    
    return np.array(dups)

def calib_hoz_right(images,color_threshold,stepx,stepy):
    dups=[]
    for index in range(len(images)):
        img = cv2.cvtColor(images[index], cv2.COLOR_BGR2GRAY)
        # id = index+1 if index+1<len(images) else -1
        # img_ = cv2.cvtColor(images[id], cv2.COLOR_BGR2GRAY)
        dup = []
        for i in range(0,len(img),stepy):
            bottom = i+stepy if i+stepy<=len(img) else len(img)
            color_l = img[i:bottom,-stepx:].mean()
            count = 0
            for j in range(-stepx,-len(img[0]),-stepx):
                right = j-stepx if abs(j-stepx)<=len(img[0]) else -len(img[0])
                color_s = img[i:bottom,right:j].mean()
                if abs(color_s-color_l)<=color_threshold:
                    count+=stepx
                else:
                    break
            for j in range(stepy):
                dup.append(count)
        # print("Right "+str(index))
        # for i in range(0,len(dup),stepy):
        #     print(i,dup[i])
        dups.append(dup)
        # plt.title(str(index))
        # plt.imshow(img,cmap='gray')
        # plt.show()
    return np.array(dups)

depths = []
images = []

# print(dup_left.shape)
# print(dup_right.shape)
def derevative_check(deriv,epsilon):
    flag_h = deriv>epsilon
    flag_l = deriv<-epsilon
    flag = np.abs(deriv)>epsilon
    # print(np.count_nonzero(flag_h))
    # print(np.count_nonzero(flag_l))
    # print(np.count_nonzero(flag))
    try:
        if np.argwhere(flag>0)[0]<10:
            flag[:10] = True
        if np.argwhere(flag>0)[-1]>len(flag)-10:
            flag[-10:] = True
    except:
        pass
    i=10
    while i < len(flag_h)-10:
        if (flag[i] and flag[i-1]==False):
            # print("start: ",i)
            start = i
            flag[i] = True
            flag_check = 1 if flag_h[i] else -1
            check = False
            while(check==False and i<len(flag_h)-1):
                i+=1
                if flag_check == 1:
                    # if flag_h[i] and flag[i-1] ==False:
                    #     start = i
                    if flag_l[i] and flag[i+1]==False:
                        check = True
                if flag_check == -1:
                    # if flag_l[i] and flag[i-1] ==False:
                    #     start = i
                    if flag_h[i] and flag[i+1]==False:
                        check = True
            if check == True:
                for k in range(start,i+1):
                    flag[k] = True
            # print("end ",i)
        i+=1
    return flag 

def delta_update(delta,flag,mask):
    maker = -1
    delta_n = delta.copy()
    for i in range(0,len(delta)):
        if flag[i]==True:
            if flag[i-1] ==False:
                maker = i
            if maker>=10 and maker<=len(delta)-10:
                delta_n[i] = delta_n[maker]
            else:
                delta_n[i] = 0
    return delta_n            

  
def calib_mask(dup_l,dup_r,delta,threshold):
    mask_left=np.zeros((len(dup_l),len(dup_l[0])))
    mask_right=np.zeros((len(dup_r),len(dup_r[0])))
    dy = np.array([delta[:,i+1]-delta[:,i] for i in range(len(delta[0])-1)]).T
    last_colum = dy[:,-1]
    dy = np.append(dy,last_colum[:,np.newaxis],axis=1)
    print(dy.shape)
    # print(dup_l.shape)
    for i in range(len(dup_l)):
        for j in range(len(dup_l[0])):                
            mask_left[i][j] = (dup_l[i,j]>threshold and dup_r[i-1,j]>threshold) 
            right = i+1 if i+1<len(dup_l) else -1
            mask_right[i][j] = (dup_r[i,j]>threshold and dup_l[right,j]>threshold)
    return mask_left,mask_right

for index in range(6):
    img = cv2.imread("output/match/None/"+folder+"-"+str(index)+".png",cv2.IMREAD_COLOR)
    # print(img.shape)
    depth = np.load("output/match/None/"+folder+"-depth-"+str(index)+".npy")
    depth = np.polyval(coeff,depth)
    depths.append(depth)
    images.append(img)

dup_left = calib_hoz_left(images,25,1,1)
dup_right = calib_hoz_right(images,25,1,1)

delta = np.array([depths[index][:,0]-depths[index-1][:,-1] for index in range(6)])
mask_left,mask_right = calib_mask(dup_left,dup_right,delta,10)
print(mask_left)
print(mask_right)

def new_depth(depths,mask_left,mask_right,delta):
    depths_new = [row.copy() for row in depths]
    dy = np.array([delta[:,i+5]-delta[:,i] for i in range(len(delta[0])-5)]).T
    last_colum = dy[:,-1]
    for i in range(5):
        dy = np.append(dy,last_colum[:,np.newaxis],axis=1)
    # delta = m-depths[index][:,0]
    for index in range(len(depths)):
        # print(dy[index])
        flag = derevative_check(dy[index],0.2)
        print(np.count_nonzero(flag))
        new_delta = delta_update(delta[index],flag,mask_left[index])
        center = int(len(depths[index][0])/2)
        for i in range(0,center):        
            depths_new[index][:,i] += -new_delta/2*(1-i/center)
        center = int(len(depths[index-1][0])/2)
        for i in range(-1,-center,-1):
            depths_new[index-1][:,i] += new_delta/2*(1-abs(i+1)/center)
    return depths_new
depths_new = new_depth(depths,mask_left,mask_right,delta)

for index in range(6):
    d = depths[index][:,0]-depths[index-1][:,-1]
    m = (depths[index][:,0]+depths[index-1][:,-1])/2
    # print(depths_new[index][:,0]-depths_new[index-1][:,-1])
    print(d.min(),d.max(),d.mean())
    plt.title(str(index))
    plt.plot(np.arange(0,len(d),1),d,'y')
    dy = [d[i+5]-d[i] for i in range(len(d)-5)]
    plt.plot(np.arange(0,len(d),1),depths_new[index][:,0],'green')
    plt.plot(np.arange(0,len(d),1),depths_new[index-1][:,-1],'blue')
    plt.plot(np.arange(0,len(d),1),depths[index][:,0],'black')
    plt.plot(np.arange(0,len(d),1),depths[index-1][:,-1],'cyan')
    # plt.plot(np.arange(0,len(d),1),(depths[index][:,0]+depths[index-1][:,-1])/2,'black')
    plt.plot(np.arange(0,len(dy),1),dy,'r')
    plt.show()

for index in range(len(depths_new)):
    np.save("output/match/None/calib/"+folder+"-depth-"+str(index)+".npy",depths_new[index])
    print("save "+str(index))
    
    