import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob 
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as colors

#FOV = 62.2

path = "output/27072023-1628/200"
# path = "output/12052023-1348/"
coeff = [ 1.19410183, -1.003789  ]

def draw_depth(file_stem,depth):
    plt.imsave(path+"/img/"+"gr.png",depth,cmap='gray')
    image_gray = cv2.imread(path+"/img/"+'gr.png',cv2.IMREAD_GRAYSCALE)
       
    # print(np.unravel_index(np.argmin(image_gray),image_gray.shape))
    plt.title(file_stem)   
    plt.imshow(depth,cmap='jet_r')
    norm_color = plt.Normalize(vmin=0,vmax=255) 
    print(depth.min(),depth.max(),depth.min(),depth.max(),image_gray.min(),image_gray.max())
    cmap = colors.ListedColormap(cm.jet(norm_color(np.linspace(image_gray.min(), image_gray.max(), 256))))
    cmap = cmap.reversed()       
    # v = np.linspace(dis.min(), dis.max(), 20, endpoint=True)
    # plt.contourf(depth,v,cmap = cmap)
    norm_range = plt.Normalize(vmin=depth.min(),vmax=depth.max()) 
    sm = cm.ScalarMappable(cmap=cmap,norm=norm_range)
    cbar = plt.colorbar(sm)
    cbar.locator = plt.MaxNLocator(nbins=20)
    cbar.update_ticks()
      
    # plt.savefig(path+"/Distance/"+file_stem+'_d.png')
    plt.show()
    plt.close()
    
depths = []
depth_path = sorted(glob.glob(path+"/first_depth/*.npy", recursive=True))
for path_elem in depth_path:
    depth = np.load(path_elem)
    if depths == []:
        depths = depth
    else:
        depths = np.concatenate((depths,depth),axis=1)
    # depth = np.polyval(coeff,depth)
    try:
        file_stem = path_elem.split('\\')[-1]
    except:
        file_stem = path_elem.split('/')[-1]
        pass
    
    output_directory = Path(path+"/Distance")
    output_directory.mkdir(parents=True,exist_ok=True)
    
    draw_depth(file_stem,depth)

draw_depth("all",depths)

