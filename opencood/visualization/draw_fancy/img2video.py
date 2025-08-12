import cv2 as cv
import numpy as np
import glob
import os

projnames = os.listdir("./result")
projnames = ['scene_overview_Mixed2','location_in_bev2']
print(projnames)

for projname in projnames:
    img_array = []
    for filename in sorted(glob.glob(f'./result/{projname}/*.png')):
        print(filename)
        img = cv.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


    out = cv.VideoWriter(f'./result_video/{projname}.mp4',cv.VideoWriter_fourcc(*'mp4v'), 15, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()