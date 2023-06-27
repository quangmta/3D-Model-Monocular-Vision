import cv2
import numpy
import glob
from pathlib import Path

folder = "input/12052023-1348/*/l.jpg"
images_path = sorted(glob.glob(folder, recursive=True))
image_list = []
for image_path in images_path:
    image = cv2.imread(image_path)
    image = cv2.resize(image,(0,0),fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
    image_list.append(image)
stitcher = cv2.Stitcher.create()
status,result = stitcher.stitch(image_list)
if (status == cv2.STITCHER_OK):
    cv2.imshow("img",result)
    cv2.waitKey(0)
    cv2.imwrite("output/12052023-1348-high.png",result)
else:
    print("False")