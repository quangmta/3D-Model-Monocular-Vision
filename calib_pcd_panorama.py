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

folder = "12052023-1348"
path_out = 'output/match/'
# path_in_npy = "output/match/"+folder+"-depth.npy"
# path_in_img = "output/match/"

def worldCoords(width, height,fx,fy):
    cx, cy = width / 2, height / 2
    xx, yy = np.tile(range(width),(height,1)), np.tile(range(height),(width,1)).T
    xx = (xx - cx) / fx
    yy = (yy - cy) / fy
    return xx, yy

depth = np.load("output/match/"+folder+"-depth"+".npy")
img = cv2.imread("output/match/"+folder+".jpg")

height, width = depth.shape[:2]
focal = 3.04/2.76*height

xx,yy = worldCoords(width,height,focal,focal)
# print(xx.shape,yy.shape)

# depth[edges(depth) > 0.3] = 1e6  # Hide depth edges
points_3D = np.stack((xx * depth, yy * depth, depth),axis=2)
points_3D = points_3D.reshape(height,width,3)
# print(points_3D.shape)

# Remove INF values from point cloud
points_3D[points_3D == float('+inf')] = 0
points_3D[points_3D == float('-inf')] = 0

delta = np.linspace(-np.pi, np.pi, width)
for i in range(height):
    for j in range(width):
        Q = [[np.cos(delta[j]),0,np.sin(delta[j])],
            [0,1,0],
            [-np.sin(delta[j]),0,np.cos(delta[j])]]
        points_3D[i][j] = np.dot(Q,points_3D[i][j].T).T

print(points_3D.shape)

# Mask colors and points
colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1,3)

# Get rid of points with value 0 (i.e no depth)
mask_map =depth.flatten() > depth.flatten().min()

points_3D = points_3D.reshape(-1,3)
output_points = points_3D[mask_map]
output_colors = colors[mask_map].astype(np.float64)/255

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(output_points.reshape(-1,3))
pcd.colors = o3d.utility.Vector3dVector(output_colors.reshape(-1,3))
o3d.io.write_point_cloud(path_out+"panorama/"+folder+"-match"+".pcd", pcd)

