# 从二维码姿态估计触觉探针末端位置

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as spR


class readImg():
    # read image
    def __init__(self):
        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        

    def run(self):
        # ret, frame = self.cap_right.read()
        ret, frame = self.cap.read()
        return ret, frame


"""read aruco code pose"""
class readAruCo():
    # read Aruco pose from image
    def __init__(self) -> None:
        # camera 0
        self.camera_matrix = np.array( [[ 522.1905267183170736, 0.0000000000000000, 308.8532705794233948  ],
                  [ 0.0000000000000000, 522.8166355728753842, 234.2130798407917496 ],
                  [ 0.0000000000000000, 0.0000000000000000, 1.0000000000000000 ]])
        self.camera_dist = np.array([ 0.1058497356483027 ,-0.1488874492570700 ,0.0000000000000000 ,0.0000000000000000 ,-0.8698561798572152])
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.mark_size = 0.005

    def readPose(self,img):
        # return tuple(A,B): A is data, B is image
        #  A: [[marker_id, x,y,z,rx,ry,rz]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.float32)/25
        gray = cv2.filter2D(gray,-1,kernel)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)

        if ids is None or len(ids) != 1:
            return False,False
        
        indexs = ids.argsort(axis=0)
        indexs = indexs.reshape(-1)

        order_ids = ids[indexs]
        corners = np.array(corners)
        orders_corners = corners[indexs]
        color_image_result = img.copy()
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(orders_corners, self.mark_size, self.camera_matrix, self.camera_dist)

        for i in range(len(ids)):
            
            # color_image_result = cv2.aruco.drawAxis(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
            # color_image_result = cv2.aruco.drawFrameAxes(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], 0.01)
            color_image_result = cv2.drawFrameAxes(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], 0.01)
        return np.hstack((order_ids, tvec.reshape(-1,3), rvec.reshape(-1,3))), color_image_result


# from aruco pose to tip position
class tactileTipPosition():
    def __init__(self):
        self.imgRead = readImg()
        self. markerParse = readAruCo()

        self. marker2tip = np.array([[1,0,0,0],
                                     [0,1,0,0],
                                     [0,0,1,-0.04],
                                     [0,0,0,1]])

    def transfer(self, show_img_flag = False):
        flag=False
        for i in range(20):
            # load image
            ret,img = self.imgRead.run()
            if ret == False:
                continue

            # parse marker pose
            pose, img_with_axis = self.markerParse.readPose(img)
            if type(pose) == bool:
                continue

            if show_img_flag:
                cv2.imshow("img",img_with_axis)
                cv2.waitKey()

            # from marker pose to tip position
            # print('pose:',pose)
            # pose: [[ 0.         -0.08010901 -0.05272539  0.18206681 -3.16545907  0.07335146
#   -0.06071261]]
            temp = pose[0][1:]
            print('二维码姿态向量',temp)
            # print('diaplace:',temp[0:3] - pose)
# temp: [-0.08010901 -0.05272539  0.18206681 -3.16545907  0.07335146 -0.06071261]
            # temp_matrix = self.pose2matrix(temp)
            temp_matrix = pose2matrix(temp)
            print('二维码姿态矩阵',temp_matrix)
            tip_matrix = np.matmul(temp_matrix, self.marker2tip)
            print('探头处的姿态矩阵',tip_matrix)
            flag = True
            break
        # tip position (x,y,z)
        if flag:
            return True, tip_matrix[0:3,3]
        else:
            return False, 0

    # from 6D pose to 4x4 matrix
def pose2matrix(pose_vector):
    pose_matrix = np.eye(4)
    pose_matrix[0:3,3]=pose_vector[0:3]

    temp = spR.from_rotvec(pose_vector[3:6])
    pose_matrix[0:3,0:3] = temp.as_matrix()
    return pose_matrix


# from camera to robot tcp coordinate
def camera2robot():
    # marker pose in camera
    # camera_origin = [0.01, 0.02, 0.03]
    camera_origin = [0.00227122, -0.00269389, 0.02171412]
    camera_e1=[0.9974826,  -0.02152942, -0.0675644]
    camera_e2=[-0.02716051, -0.99613216, -0.08356455]
    camera_e3=[-0.06550398,  0.08518927, -0.99420924]
    # basis vector as column
    E_C = np.vstack((camera_e1,camera_e2,camera_e3)).T

    # marker pose in robot TCP from CAD model
    robot_origin = [0.01, 0.02, 0.03]
    robot_e1=[1,0,0]
    robot_e2=[0,1,0]
    robot_e3=[0,0,1]
    E_R = np.vstack((robot_e1,robot_e2,robot_e3)).T

    # rotation transfer matrix
    camera_to_robot_rotation = np.matmul(E_R, np.linalg.inv(E_C))
    camera_to_robot_translation = camera_origin - robot_origin

    # camera to robot transfer matrix
    eye2hand_matrix = np.eye(4)
    eye2hand_matrix[0:3,3]=camera_to_robot_translation
    eye2hand_matrix[0:3,0:3] = camera_to_robot_rotation

    return eye2hand_matrix


if __name__ == "__main__":
    print('====test====')
    camera_e1=[1,0,1]
    camera_e2=[0,1,0]
    camera_e3=[0,0,1]

    e = np.vstack((camera_e1,camera_e2,camera_e3)).T
    print(e)
    a = tactileTipPosition()
    print(a.transfer(True))


