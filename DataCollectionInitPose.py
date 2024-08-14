# collect images
import cv2
import time
import os
import numpy as np
from scipy.spatial.transform import Rotation as spR
from ATI import ATIController
import datetime


class readAruCo():
    # read Aruco pose from image
    def __init__(self) -> None:
        self.camera_matrix = np.array([[302.08997788,   0.        , 325.50850109],
                           [  0.        , 305.26831967, 170.27425144],
                           [  0.        ,   0.        ,   1.        ]])
        self.camera_dist = np.array([-0.3488709,   0.19554033, -0.00161501, -0.00039101, -0.08049281])
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.mark_size = 0.016
        self.flag = True

    def readPose(self,img):
        self.flag = True
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.float32)/25
        gray = cv2.filter2D(gray,-1,kernel)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)
        if ids is None:
            self.flag = False
            return 0,0
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[0], self.mark_size, self.camera_matrix, self.camera_dist)
        rr = spR.from_rotvec(rvec[0])
        # rpy = rr.as_euler('zyx', degrees=True)[0][::-1]
        rpy = rr.as_euler('zyx', degrees=True)
        if rpy[0,2]<0:
            rpy[0,2] = rpy[0,2] + 360
        pose = np.hstack((tvec[0]*1000, rpy))
        color_image_result = cv2.aruco.drawAxis(img, self.camera_matrix, self.camera_dist, rvec[0], tvec[0], self.mark_size)
        return pose, color_image_result

# initialization
cap = cv2.VideoCapture(2)
readPose = readAruCo()
time.sleep(1)

# init pose
# [x,y,z,rx,ry,rz]
data_list = []
ts = time.time()
for i in range(100):
    ret, color_image = cap.read()
    tc = time.time()
    if ret:
        pose, resultImg = readPose(color_image)
        data_list.append(pose)
    else:
        continue


# save data
dirname = './data'
if not os.path.isdir(dirname):
    os.makedirs(dirname)

file_name ='init_pose.txt'

np.savetxt(file_name, data_list)