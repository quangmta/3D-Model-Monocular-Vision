import numpy as np
import cv2
folder = '27072023-1628'
width = int(3280/2464*384/62.2*360)
panorama = cv2.imread("output/panorama/"+folder+"-panorama.png")
panorama = cv2.resize(panorama,(width,384))
cv2.imwrite("output/panorama/"+folder+".png",panorama[20:-20,:-2,:])
cv2.imshow("img",panorama[20:-20,:-2,:])
cv2.waitKey(0)