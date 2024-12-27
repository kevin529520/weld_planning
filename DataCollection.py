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
        # 加载初始位姿数据
        init_pose_path = os.path.join('data', 'camera1', 'init_pose.txt')
        try:
            pose_data = np.loadtxt(init_pose_path)
            # 计算参考位姿
            self.p0 = np.mean(pose_data, axis=0)
            # 验证结果维度
            if not len(self.p0) in [6, 7]:  # 允许6或7个值
                raise ValueError(f"Unexpected number of values in pose: {len(self.p0)}")
        except Exception as e:
            print(f"Error initializing ArUco reader: {e}")
            raise

        # 加载初始力数据
        init_force_path = os.path.join('data', 'ft', 'init_force.txt')
        try:
            force_data = np.loadtxt(init_force_path)
            # 计算参考位姿
            self.f0 = np.mean(force_data, axis=0)
            # 验证结果维度
            if not len(self.f0) in [6, 7]:  # 允许6或7个值
                raise ValueError(f"Unexpected number of values in pose: {len(self.f0)}")
        except Exception as e:
            print(f"Error initializing ATI reader: {e}")
            raise

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
        if rpy[0,0]<0:
            rpy[0,0] = rpy[0,0] + 360
        pose = np.hstack((tvec[0]*1000, rpy))
        # color_image_result = cv2.aruco.drawAxis(img, self.camera_matrix, self.camera_dist, rvec[0], tvec[0], self.mark_size)
        color_image_result = cv2.drawFrameAxes(img, self.camera_matrix, self.camera_dist, rvec[0], tvec[0], self.mark_size)

        color_image_result = aruco.drawDetectedMarkers(color_image_result, corners, ids)
        cv2.imshow('result', color_image_result)
        cv2.waitKey(1)  # 添加等待时间  

        return pose, color_image_result

# initialization
sensor_nano25 = ATIController()
cap = cv2.VideoCapture(2)
cap.set(3,1920) #设置分辨率
cap.set(4,1080)
readPose = readAruCo()
time.sleep(1)
sensor_nano25.setZero()
time.sleep(1)

# collect data
# [time,fx,fy,fz,tx,ty,tz,x,y,z,rx,ry,rz]
data_list = []
ts = time.time()
for i in range(10000):
    if i % 100 == 0:
        print('i:', i)
    finger_base_ft = sensor_nano25.readDate()
    df = finger_base_ft - readPose.f0   
    ret, color_image = cap.read()
    tc = time.time()
    if ret:
        pose, resultImg = readPose.readPose(color_image)
        time_stamp = tc-ts
        # if isinstance(pose, np.ndarray):
        #     pose = pose.flatten()
        # pose = pose.flatten() 
        # print("time: ", time_stamp, "pose: ", pose, "force: ", finger_base_ft)
        if isinstance(pose, np.ndarray):
            pose = pose.flatten()
            dp = pose - readPose.p0
            temp = np.hstack((time_stamp, df, dp))
            data_list.append(temp)
    else:
        continue

# save data
dirname = './data/camera1'
if not os.path.isdir(dirname):
    os.makedirs(dirname)

current_date = datetime.datetime.now()
file_name =str(current_date.year)+"-"+str(current_date.month)+"-"+str(current_date.day)+"-"+str(current_date.hour)+"-"+str(current_date.minute)+'.txt'

for x in data_list:
    if len(x) != 13:
        print("Error: ", x, len(x))
        data_list.remove(x)

np.savetxt(dirname + "/"+ file_name, data_list)
np.save(dirname + '/' + file_name, data_list)