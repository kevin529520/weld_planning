import cv2
import os

image_dir = './cam0-1280-0131/'
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

camera=cv2.VideoCapture(1)
camera.set(3,2592) #设置分辨率
camera.set(4,1944)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE,3)
# camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # fixed exposure
# camera.set(cv2.CAP_PROP_EXPOSURE,0.05)  # cam1-4:0.05 for 1080, 720

i = 0
while 1:
    (grabbed, img) = camera.read()
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('j'):  # 按j保存一张图片
        i += 1
        u = str(i)
        firename=str(image_dir+u+'.jpg')
        cv2.imwrite(firename, img)
        print('写入：',firename)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("图像采集完毕")
print("开始识别")
import cv2
import numpy as np
import glob

# 找棋盘格角点
# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
#棋盘格模板规格
# w = 4   # 10 - 1
# h = 3   # 7  - 1
w = 4   # 10 - 1
h = 5   # 7  - 1

# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
objp = objp*0.0015  # 18.1 mm  6.3mm 3.2mm

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = [] # 在世界坐标系中的三维点
imgpoints = [] # 在图像平面的二维点
#加载pic文件夹下所有的jpg图像
images = glob.glob(image_dir+'*.jpg')  #   拍摄的十几张棋盘图片所在目录

i=0
for fname in images:
    print('in image')
    img = cv2.imread(fname)
    # 获取画面中心点
    #获取图像的长宽
    h1, w1 = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('findCorners',gray)
    cv2.waitKey(600)
    u, v = img.shape[:2]
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        print("i:", i)
        i = i+1
        # 在原角点的基础上寻找亚像素角点
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #追加进入世界三维点和平面二维点中
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 1280, 720)
        cv2.imshow('findCorners',img)
        cv2.waitKey(600)


cv2.destroyAllWindows()

print('正在计算')
#标定
ret, mtx, dist, rvecs, tvecs = \
    cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


print("ret:",ret  )
print("mtx:\n",mtx)      # 内参数矩阵
print("mtx:\n",np.array(mtx))
print("dist畸变值:\n",dist   )   # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
# print("rvecs旋转（向量）外参:\n",rvecs)   # 旋转向量  # 外参数
# print("tvecs平移（向量）外参:\n",tvecs  )  # 平移向量  # 外参数
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))
print('fine matirx:',newcameramtx)

# camera 0, 1280x720
# mtx:
#  [[695.62418922   0.         624.11765476]
#  [  0.         698.0167323  377.5026324 ]
#  [  0.           0.           1.        ]]
# dist畸变值:
#  [[ 0.08436435 -0.11125308  0.001353   -0.00391756  0.01989086]]



# camera 0, 1920x1080
# mtx:
#  [[1.35498828e+03 0.00000000e+00 9.63582885e+02]
#  [0.00000000e+00 1.35424333e+03 5.70237379e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# dist畸变值:
#  [[ 0.0433853  -0.05228565  0.00079905  0.00208749 -0.01949841]]
# newcameramtx外参 [[1.35827380e+03 0.00000000e+00 9.64499616e+02]
#  [0.00000000e+00 1.35569983e+03 5.67785628e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]



# camera.set(3,2592) #设置分辨率
# camera.set(4,1944)
# ret: 0.7459074963007746
# mtx:
#  [[3.20790520e+03 0.00000000e+00 1.01091874e+03]
#  [0.00000000e+00 3.39787065e+03 6.44844641e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# mtx:
#  [[3.20790520e+03 0.00000000e+00 1.01091874e+03]
#  [0.00000000e+00 3.39787065e+03 6.44844641e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# dist畸变值:
#  [[ 3.40400374e-01 -2.65574509e+00  9.96946780e-03  8.68103031e-03
#   -4.46459652e+01]]
# fine matirx: [[3.21111694e+03 0.00000000e+00 1.01193085e+03]
#  [0.00000000e+00 3.39769067e+03 6.43912013e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]