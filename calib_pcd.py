import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import glob
from pathlib import Path
import open3d as o3d
from scipy import ndimage

folder = "12052023-1348"
path_out = 'output/point-cloud/'+folder+"/"
path_in_npy = "output/"+folder+"/"
path_in_img = "input/"+folder+"/"

# angle_scan_cm = 2*math.degrees(math.atan(math.tan(math.radians(62.2/2))*3.04/f))
k=11.25/62.2

output_directory = Path(path_out)
output_directory.mkdir(parents=True,exist_ok=True)

output_directory = Path(path_out+"/i/")
output_directory.mkdir(parents=True,exist_ok=True)

output_directory = Path(path_out+"/o")
output_directory.mkdir(parents=True,exist_ok=True)

depth_path = sorted(glob.glob(path_in_npy+"*.npy", recursive=True))
# depth_path += sorted(glob.glob(path_in_npy+"42.npy", recursive=True))


# dis_ex = np.load(depth_path[0])
# point_num = len(depth_path)*(dis_ex.shape[0]*int(dis_ex.shape[1]*k)-1)

pcd = o3d.geometry.PointCloud()

# Compute edge magnitudes
def depth_edges_mask(depth):
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    # Compute the edge mask.
    mask = depth_grad > 0.05
    return mask

def worldCoords(width, height,fx,fy):
    cx, cy = width / 2, height / 2
    xx, yy = np.tile(range(width),(height,1)), np.tile(range(height),(width,1)).T
    xx = (xx - cx) / fx
    yy = (yy - cy) / fy
    return xx, yy

for path_elem in depth_path:        
    file_stem = path_elem.split('\\')[-1]
    angle = file_stem[:-4]
    if not angle.isnumeric():
        file_stem = path_elem.split('/')[-1]
        angle = file_stem[:-4]
    print(angle)
    depth = np.load(path_elem)
    imgL = cv2.imread(path_in_img+angle+'/l.jpg')
    imgL = cv2.resize(imgL,depth.shape[::-1],interpolation = cv2.INTER_AREA)
    
    height, width = depth.shape[:2]
    focal = 3.04/3.68*width
    
    xx,yy = worldCoords(width,height,focal,focal)
    # print(xx.shape,yy.shape)
    
    # depth[edges(depth) > 0.3] = 1e6  # Hide depth edges
    points_3D = np.stack((xx * depth, yy * depth, depth),axis=2)

    # Remove INF values from point cloud
    points_3D[points_3D == float('+inf')] = 0
    points_3D[points_3D == float('-inf')] = 0

    # Get rid of points with value 0 (i.e no depth)
    mask_map =depth > depth.min()

    # Mask colors and points
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    
    # # find max and min value of first column in matrix
    # arg_min = np.argmin(points_3D[:,:,0])
    # arg_max = np.argmax(points_3D[:,:,0])
    # tem_min = points_3D[arg_min//depth.shape[1],arg_min%depth.shape[1]]
    # tem_max = points_3D[arg_max//depth.shape[1],arg_max%depth.shape[1]]
    # # print(tem_min,tem_max)
    # angle_min = math.atan(tem_min[2]/tem_min[0])
    # angle_max = math.atan(tem_max[2]/tem_max[0])
    # if tem_min[0]<0:
    #     angle_min-=math.pi
    # if tem_max[0]<0:
    #     angle_max-=math.pi
    # angle_scan = abs((angle_max-angle_min)*180/math.pi)
    # print(angle_scan)    
    
    #cal    
    # left = int((k+3/62.2)*disparity.shape[1])
    # right = left+int((k)*disparity.shape[1])
    
    # new_mask_map = mask_map[:,left:right]
    # new_points = points_3D[:,left:right,:]
    # new_colors = colors[:,left:right,:]

    new_mask_map = mask_map
    new_points = points_3D
    new_colors = colors

    # delta = (float(angle))/1024*2*math.pi
    # for i in range(new_points.shape[0]):
    #     for j in range(new_points.shape[1]):
    #         R = math.sqrt(math.pow(new_points[i][j][0],2)+math.pow(new_points[i][j][2],2))
    #         if new_points[i][j][0]==0:
    #             if new_points[i][j][2]:
    #                 alpha = math.pi/2
    #             else:
    #                 alpha = -math.pi/2
    #         else:
    #             alpha = math.atan(new_points[i][j][2]/new_points[i][j][0])                
    #             if(new_points[i][j][0]<0):
    #                 alpha += math.pi
            
    #         new_points[i][j][0] = R*math.cos(alpha+delta)
    #         new_points[i][j][2] = R*math.sin(alpha+delta)
    output_points = new_points[new_mask_map]
    output_colors = new_colors[new_mask_map].astype(np.float64)/255

    pcd_i = o3d.geometry.PointCloud()
    pcd_i.points = o3d.utility.Vector3dVector(output_points.reshape(-1,3))
    pcd_i.colors = o3d.utility.Vector3dVector(output_colors.reshape(-1,3))
    o3d.io.write_point_cloud(path_out+""+angle+".pcd", pcd_i)
    # pcd+=pcd_i
# Save the point cloud as a PCD file
# o3d.io.write_point_cloud(path_out+"i/all.pcd", pcd)
# pcd = o3d.io.read_point_cloud(path_out)
# o3d.visualization.draw_geometries([pcd],window_name="PointCloud")
