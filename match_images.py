import cv2
import numpy
import glob
from pathlib import Path

input_path = "input/"
folder = "27072023-1628"
images_path = sorted(glob.glob(input_path+folder+"/*/l.jpg", recursive=True))

angle_list =[]
for path_elem in images_path:    
    angle = path_elem.split('\\')[-2]
    # angle = file_stem[:-4]
    if not angle.isnumeric():
        angle = path_elem.split('/')[-2]
    angle_list.append(int(angle))
angle_list.sort()

image_list = []
for i in angle_list:
    image = cv2.imread(input_path+folder+"/"+str(i)+"/l.jpg")
    image = cv2.resize(image,(512,384),interpolation = cv2.INTER_AREA)
    image_list.append(image)
print("Starting match "+str(len(angle_list))+" images")
stitcher = cv2.Stitcher.create()
status,result = stitcher.stitch(image_list)
if (status == cv2.STITCHER_OK):
    cv2.imshow("img",result)
    cv2.waitKey(0)
    cv2.imwrite("output/panorama/"+folder+"-panorama.png",result)
else:
    print("False")