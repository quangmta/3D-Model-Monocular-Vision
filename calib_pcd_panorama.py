import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import glob
from pathlib import Path
import open3d as o3d
from scipy import ndimage
from ui.gradio_pano_to_3d import pano_depth_to_world_points
import tqdm

devide = 6
shift = 250
folder = "27072023-1628"
output_directory = 'output/'+folder+'/'+str(shift)+'/point-cloud'
Path(output_directory).mkdir(parents=True,exist_ok=True)
# path_in_npy = "output/match/"+folder+"-depth.npy"
# path_in_img = "output/match/"

# Compute edge magnitudes
def edges(d):
    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)

def worldCoords(width, height,fx,fy):
    cx, cy = width / 2, height / 2
    xx, yy = np.tile(range(width),(height,1)), np.tile(range(height),(width,1)).T
    xx = (xx - cx) / fx
    yy = (yy - cy) / fy
    return xx, yy
pcd_all = o3d.geometry.PointCloud()
for index in range(devide):
    # depth = np.load("output/{0}/{2}/{0}-depth-{1}-{2}.npy".format(folder,index,shift))
    # depth = np.load("output/{0}/{2}/calib_param/{0}-depth-{1}-{2}.npy".format(folder,index,shift))
    depth = np.load("output/{0}/{2}/match_diff/{0}-depth-{1}-{2}.npy".format(folder,index,shift))
    # depth = np.polyval(coeff,depth)
    img = cv2.imread("output/{0}/{2}/img/{0}-{1}-{2}.png".format(folder,index,shift))

    height, width = depth.shape[:2]
    focal = 3.04/2.76*384
    
    xx,yy = worldCoords(width,height,focal,focal)
    # print(xx.shape,yy.shape)

    # depth[edges(depth) > 0.3] = 1e6  # Hide depth edges
    points_3D = np.stack((xx * depth, yy * depth, depth),axis=2).reshape(-1,3)
    print(points_3D.shape)

    # Remove INF values from point cloud
    points_3D[points_3D == float('+inf')] = 0
    points_3D[points_3D == float('-inf')] = 0

    # Mask colors and points
    colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1,3)

    delta = 2*math.pi*(index/devide+shift/(6*width))
    Q = [[np.cos(delta),0,np.sin(delta)],
         [0,1,0],
         [-np.sin(delta),0,np.cos(delta)]]
    points_3D = np.dot(Q,points_3D.T).T
    
    # Get rid of points with value 0 (i.e no depth)
    mask_map =depth.flatten() > depth.flatten().min()
    
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map].astype(np.float64)/255
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(output_points.reshape(-1,3))
    pcd.colors = o3d.utility.Vector3dVector(output_colors.reshape(-1,3))
    o3d.io.write_point_cloud(output_directory+'/{0}-{1}-{2}.pcd'.format(folder,index,shift),pcd)
    pcd_all+=pcd

o3d.io.write_point_cloud(output_directory+'/{0}-{1}.pcd'.format(folder,shift), pcd_all)

