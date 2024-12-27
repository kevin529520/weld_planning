# collect images
import cv2
import time
import os
import numpy as np
from scipy.spatial.transform import Rotation as spR
from ATI import ATIController
import datetime
import cv2.aruco as aruco

class readAruCo():
    # read Aruco pose from image
    def __init__(self) -> None:
        self.camera_matrix = np.array( [[1.54340965e+03, 0.00000000e+00, 9.69812817e+02],
                                        [0.00000000e+00, 1.54134379e+03, 5.12276013e+02],
                                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self.camera_dist = np.array( [0.0522575014850949, -0.353834933819654, 0.00197535440320868, 0.00128091948470123, 0 ])
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.mark_size = 0.005
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
        color_image_result = cv2.drawFrameAxes(img, self.camera_matrix, self.camera_dist, rvec[0], tvec[0], self.mark_size)
        color_image_result = aruco.drawDetectedMarkers(color_image_result, corners, ids)
        cv2.imshow('result', color_image_result)
        cv2.waitKey(1)  # 添加等待时间  
        return pose, color_image_result
    
    # def readPose(self,img):
    #     self.flag = True
        
    #     # 检查输入图像
    #     if img is None or img.size == 0:
    #         print("Invalid input image")
    #         return 0,0
        
    #     # 显示原始图像
    #     # cv2.imshow('original', img)
        
    #     # 灰度转换
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     # cv2.imshow('gray', gray)
        
    #     # 滤波
    #     kernel = np.ones((3,3),np.float32)/9
    #     gray = cv2.filter2D(gray,-1,kernel)
    #     # cv2.imshow('filtered', gray)
        
    #     # 检测标记
    #     corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)
        
    #     if ids is None:
    #         print("No markers detected")
    #         self.flag = False
    #         return 0,0
            
    #     print("Marker detected, ID:", ids)
        
    #     # 姿态估计
    #     rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[0], self.mark_size, self.camera_matrix, self.camera_dist)
    #     rr = spR.from_rotvec(rvec[0])
    #     rpy = rr.as_euler('zyx', degrees=True)
        
    #     if rpy[0,2]<0:
    #         rpy[0,2] = rpy[0,2] + 360
            
    #     pose = np.hstack((tvec[0]*1000, rpy))
        
    #     # 绘制结果
    #     color_image_result = img.copy()  # 使用复制的图像
    #     color_image_result = cv2.drawFrameAxes(color_image_result, self.camera_matrix, self.camera_dist, rvec[0], tvec[0], self.mark_size)
    #     color_image_result = cv2.aruco.drawDetectedMarkers(color_image_result, corners, ids)
    #     cv2.imshow('result', color_image_result)
    #     cv2.waitKey(1)  # 添加等待时间
        
    #     return pose, color_image_result

# initialization
cap = cv2.VideoCapture(2)
cap.set(3,1920) #设置分辨率
cap.set(4,1080)
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
        pose, resultImg = readPose.readPose(color_image)
        data_list.append(pose)
    else:
        continue


# save data

# 将 data_list 转换为 numpy 数组
data_array = np.array(data_list)
data_reshaped = data_array.reshape(data_array.shape[0], -1)
print(data_array)



dirname = './data/camera1'
if not os.path.isdir(dirname):
    os.makedirs(dirname)

file_name ='init_pose.txt'

print(data_list)
# np.savetxt(file_name, data_list)
np.savetxt(dirname + '/' + file_name, data_reshaped)
np.save(dirname + '/' + file_name, data_array)