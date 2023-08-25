import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import argparse


# coeff = [ 1.19410183, -1.003789  ]

def calib_hoz_left(images,color_threshold,stepx,stepy):
    dups=[]
    for img in images:
        img = img.astype(int)
        dup = []
        for i in range(0,len(img),stepy):
            bottom = i+stepy if i+stepy<=len(img) else len(img)
            color_l = img[i:bottom,0:stepx]
            count = 0
            for j in range(stepx,len(img[0]),stepx):
                if j+stepx>len(img[0]):
                    break
                color_s = img[i:bottom,j:j+stepx]
                check = True
                for k  in range(3):
                    if abs(color_s[:,:,k]-color_l[:,:,k]).mean()>color_threshold:
                        check = False
                        break
                if check:
                    count+=stepx
                else:
                    break
            for j in range(bottom-i):
                dup.append(count)
        dups.append(dup[:len(img)])
    
    return np.array(dups)

def calib_hoz_right(images,color_threshold,stepx,stepy):
    dups=[]
    for img in images:
        img = img.astype(int)
        dup = []
        for i in range(0,len(img),stepy):
            bottom = i+stepy if i+stepy<=len(img) else len(img)
            color_r = img[i:bottom,-stepx:]
            count = 0
            for j in range(-stepx,-len(img[0]),-stepx):
                if j-stepx<-len(img[0]):
                    break
                color_s = img[i:bottom,j-stepx:j]
                check = True
                for k  in range(3):
                    if abs(color_s[:,:,k]-color_r[:,:,k]).mean()>color_threshold:
                        check = False
                        break
                if check:
                    count+=stepx
                else:
                    break
            for j in range(stepy):
                dup.append(count)
        dups.append(dup[:len(img)])
    return np.array(dups)

def calib_mask(dup_l,dup_r,threshold):
    mask_left=np.zeros((len(dup_l),len(dup_l[0])))
    mask_right=np.zeros((len(dup_r),len(dup_r[0])))
    for i in range(len(dup_l)):
        for j in range(len(dup_l[0])):
            if (dup_l[i,j]>threshold and dup_r[i-1,j]>threshold)or (dup_l[i,j]<threshold and dup_r[i-1,j]<threshold):
                mask_left[i][j] = 0
            elif dup_l[i,j]>threshold and dup_r[i-1,j]<threshold:
                mask_left[i][j] = 1
            else:
                mask_left[i][j] = -1 
                
            right = i+1 if i+1<len(dup_l) else -1
            
            if (dup_l[right,j]>threshold and dup_r[i,j]>threshold) or (dup_l[right,j]<threshold and dup_r[i,j]<threshold):
                mask_right[i][j] = 0
            elif dup_l[right,j]<threshold and dup_r[i,j]>threshold:
                mask_right[i][j] = 1
            else:
                mask_right[i][j] = -1 
                
    return mask_left,mask_right

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

def derevative_check_s(deriv,epsilon):
    arg = np.argwhere(np.abs(deriv)>epsilon).flatten()
    flag = np.zeros(len(deriv))
    if len(arg) > 1:
        flag[arg[0]:arg[-1]] = True
        if arg[0]<10:
            flag[:10] = True
        if arg[-1]>len(flag)-10:
            flag[-10:] = True
    return flag    

def delta_update(delta,flag):
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

def new_depth(depths,mask_left,mask_right,delta):
    depths_new = [row.copy() for row in depths]
    dy = np.array([(delta[:,i+5]-delta[:,i])/5 for i in range(len(delta[0])-5)]).T
    last_colum = dy[:,-1]
    for i in range(5):
        dy = np.append(dy,last_colum[:,np.newaxis],axis=1)
    for index in range(len(depths)):
        # print(dy[index])
        flag = derevative_check_s(dy[index],0.04)
        print(np.count_nonzero(flag))
        new_delta = delta_update(delta[index],flag)
        center = int(len(depths[index][0])/2)
        for i in range(0,center):        
            depths_new[index][:,i] += -new_delta/2*(1-i/center)
        center = int(len(depths[index-1][0])/2)
        for i in range(-1,-center,-1):
            depths_new[index-1][:,i] += new_delta/2*(1-abs(i+1)/center)
    return depths_new

def calib_depth(depths,dup_left,dup_right,delta,step_diff,threshold):
    depths_calib = [row.copy() for row in depths]
    dy = np.array([(delta[:,i+step_diff]-delta[:,i]) for i in range(len(delta[0])-step_diff)]).T
    last_colum = dy[:,-1]
    for i in range(int(step_diff/2)):
        dy = np.insert(dy,i,last_colum,axis=1)
    for i in range(int(step_diff/2),step_diff):
        dy = np.append(dy,last_colum[:,np.newaxis],axis=1)
    for index in range(len(depths)):
        # print(dy[index])
        arg = np.argwhere(np.abs(dy[index])>0.18).flatten()
        if len(arg) > 1:
            # if arg[0]<10:
            # arg = np.insert(arg,0,0)
            # # if arg[-1]>depths[index].shape[1]-10:
            # arg = np.append(arg,depths[index].shape[1]-2)
            # arg = np.append(arg,depths[index].shape[1]-1)
            new_arg = [0]
            meet=1
            id_meet=arg[0]
            for i in range(1,len(arg)):
                if arg[i]-arg[i-1]>1:
                    if meet:
                        new_arg = np.append(new_arg,int((arg[i-1]+id_meet)/2))
                        meet=0
                    else:
                        id_meet=arg[i]
                        meet=1
            if arg[-1]-arg[-2]==1:
                new_arg = np.append(new_arg,arg[-1])
            new_arg = np.append(new_arg,len(delta[index])-1)
            # new_arg =[x for x in new_arg if x>10 and x<len(dy[index])-10]
            print(index,arg,new_arg)
            check = np.zeros(len(new_arg))
            for i in range(len(new_arg)-1):
                print(dup_left[index][new_arg[i]:new_arg[i+1]])
                print(dup_right[index-1][new_arg[i]:new_arg[i+1]])
                if np.sum((dup_left[index][new_arg[i]:new_arg[i+1]]>=threshold) &\
                    (dup_right[index-1][new_arg[i]:new_arg[i+1]]>=threshold))/(new_arg[i+1]-new_arg[i]) < 0.5\
                    and new_arg[i+1]-new_arg[i]<len(delta[index])/2:
                    check[i] = True
                    if new_arg[i]==0:
                        delta[index][new_arg[i]:new_arg[i+1]] = np.zeros(new_arg[i+1]-new_arg[i])
                    elif i>0 and check[i-1] == True:
                        delta[index][new_arg[i]:new_arg[i+1]] = np.full(new_arg[i+1]-new_arg[i],delta[index][new_arg[i-1]])
                    else:
                        delta[index][new_arg[i]:new_arg[i+1]] = np.full(new_arg[i+1]-new_arg[i],delta[index][new_arg[i]])
    for index in range(len(depths)):
        #Check if left and right egdes are needed to turn one way
        p1 = np.sum(delta[index]>0)/(len(delta[index]))
        p2 = np.sum(delta[index]<0)/(len(delta[index]))
        p3 = np.sum(delta[index-1]>0)/(len(delta[index-1]))
        p4 = np.sum(delta[index-1]<0)/(len(delta[index-1]))
        if (p1>0.9 and p4>0.9) or (p2>0.9 and p3>0.9):
            for i in range(0,len(depths[index-1][0])):
                depths_calib[index-1][:,i] += (-delta[index-1]/2*(len(depths[index-1][0])-1-i)+delta[index]/2*i)/(len(depths[index-1][0])-1)
        else:
            center = int(len(depths[index-1][0])/2)
            for i in range(0,center):        
                depths_calib[index-1][:,i] += -delta[index-1]/2*(1-i/center)
            for i in range(-1,-center,-1):
                depths_calib[index-1][:,i] += delta[index]/2*(1-abs(i+1)/center)
    return depths_calib,dy

def match_diff(args):    
    depths = []
    images = []

    for index in range(int(args.divide)):
        img = cv2.imread("{0}/{1}/{3}/img/{1}-{2}-{3}.png".format(args.inout_directory,args.folder,index,args.shift),cv2.IMREAD_COLOR)
        # print(img.shape)
        depth = np.load("{0}/{1}/{3}/{4}/{1}-depth-{2}-{3}.npy".format(args.inout_directory,args.folder,index,args.shift,args.depth_folder))
        # depth = np.polyval(coeff,depth)
        depths.append(depth)
        images.append(img)

    dup_left = calib_hoz_left(images,25,4,4)
    dup_right = calib_hoz_right(images,25,4,4)
    # print(dup_right[3][58:135])
    # print(dup_left[4][58:135])

    delta = np.array([depths[index][:,0]-depths[index-1][:,-1] for index in range(int(args.divide))])
    # mask_left,mask_right = calib_mask(dup_left,dup_right,35)
    # print(mask_left)
    # print(mask_right)
    # depths_new = new_depth(depths,mask_left,mask_right,delta)
    step_diff = 5
    depths_new,dy = calib_depth(depths,dup_left,dup_right,delta,step_diff,15)
   
    for index in range(int(args.divide)):
        d = depths[index][:,0]-depths[index-1][:,-1]
        m = (depths[index][:,0]+depths[index-1][:,-1])/2
        # print(depths_new[index][:,0]-depths_new[index-1][:,-1])
        print(d.min(),d.max(),d.mean())
        
        fig = plt.figure(figsize=(8, 6))

        index2 = index-1 if index>0 else int(args.divide)-1
        img1 = cv2.imread('{0}/{1}/{2}/img/{1}-{3}-{2}.png'.format(args.inout_directory,args.folder,args.shift,index))
        img2 = cv2.imread('{0}/{1}/{2}/img/{1}-{3}-{2}.png'.format(args.inout_directory,args.folder,args.shift,index2))
        
        # for row,lenght in enumerate(dup_left[index]):
        #     img1[row,lenght]=[0,0,225]
        # for row,lenght in enumerate(dup_right[index]):
        #     img2[row,len(img2[0])-lenght]=[0,0,225]
        
        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        ax.plot(dup_left[index],np.arange(0,len(img1)),'r')
        ax.set_title(str(index),fontsize=10)
        ax.axis('off')  
        
        ax = fig.add_subplot(2, 2, 1)
        ax.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        ax.plot(len(img2[0])-dup_right[index2],np.arange(0,len(img2)),'r')
        ax.axis('off')  
        ax.set_title(str(index2),fontsize=10)
        
        
        ax = fig.add_subplot(2, 2, 4)
        ax.imshow(depths[index],cmap='jet_r')
        # ax.set_title(str(index),fontsize=10)
        ax.axis('off')
        
        ax = fig.add_subplot(2, 2, 3)
        ax.imshow(depths[index2],cmap='jet_r')
        # ax.set_title(str(index),fontsize=10)
        ax.axis('off')  
        
        fig.tight_layout()
        # plt.show()
        
        fig2= plt.figure(2)
        ax = fig2.add_subplot()
        
        
        ax.set_title(str(index))
        ax.plot(np.arange(0,len(d),1),d,'y',label='diffenrence')
        ax.plot(np.arange(0,len(dy[index]),1),dy[index],'r',label='diffirential')
        # dy = [(d[i+step_diff]-d[i]) for i in range(len(d)-step_diff)]
        ax.plot(np.arange(0,len(d),1),depths[index][:,0],'black',label='depth')
        ax.plot(np.arange(0,len(d),1),depths[index-1][:,-1],'cyan',label='depth*')
        ax.plot(np.arange(0,len(d),1),depths_new[index][:,0],'green',label='new depth')
        ax.plot(np.arange(0,len(d),1),depths_new[index-1][:,-1],'blue',label='new depth*')
        # plt.plot(np.arange(0,len(d),1),(depths[index][:,0]+depths[index-1][:,-1])/2,'black')
        ax.legend(loc='lower left')
        plt.show()
        # break


    output_directory = Path(args.inout_directory+'/'+args.folder+'/'+args.shift+'/match_diff')
    output_directory.mkdir(parents=True,exist_ok=True)
    for index in range(len(depths_new)):
        np.save("{0}/{1}/{3}/match_diff/{1}-depth-{2}-{3}.npy".format
                (args.inout_directory,args.folder,index,args.shift),depths_new[index])
        print("save "+str(index))    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-io','--inout_directory', help="directory to images", default="output")
    parser.add_argument('-f','--folder', help="folder of input images", default="27072023-1628")
    parser.add_argument('-df','--depth_folder', help="folder of depth", default="calib_param")
    parser.add_argument('-s','--shift', help="shift of input images", default="170")    
    parser.add_argument('-d','--divide', help="divide coefficent of input images", default="6")
    
    args = parser.parse_args()
    match_diff(args)