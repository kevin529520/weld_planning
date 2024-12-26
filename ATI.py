# MIT License.
# Copyright (c) 2021 by BioicDL. All rights reserved.
# Created by LiuXb on 2021/9/16
# -*- coding:utf-8 -*-

"""
@Modified: 
@Description: for ATI nano25 ft sensor
"""
import socket
import numpy as np
import struct
import os

# s_gt = socket.socket()
# s_gt.connect(('192.168.1.1', 49151))
# # read ATI sensor upon command
# read_calibration_info = bytes([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
#                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
# read_force = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
#                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
# reset_force = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
#                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01])
# countsPerForce = 1000000
# countsPerTorque = 1000000
# scaleFactors_force = 15260      # ATI Nano25 of SusTech
# scaleFactors_torque = 92
#
#
# def read_ati(reset=False):
#     if reset == True:
#         s_gt.send(reset_force)
#     s_gt.send(read_force)
#     force_info = s_gt.recv(16)
#     print(force_info)
#     #     print(force_info)
#     header, status, ForceX, ForceY, ForceZ, TorqueX, TorqueY, TorqueZ = struct.unpack('!2H6h', force_info)
#     Fx = ForceX * scaleFactors_force/countsPerForce
#     Fy = ForceY * scaleFactors_force/countsPerForce
#     Fz = ForceZ * scaleFactors_force/countsPerForce
#     Tx = TorqueX * scaleFactors_torque/countsPerTorque
#     Ty = TorqueY * scaleFactors_torque/countsPerTorque
#     Tz = TorqueZ * scaleFactors_torque/countsPerTorque
#     Force_torque = np.array([Fx,Fy,Fz,Tx,Ty,Tz])
#     return Force_torque


class ATIController(object):
    def __init__(self, ip='192.168.1.1'):
        self.__control = socket.socket()
        self.__control.connect((ip, 49151))
        # read ATI sensor upon command
        self.__read_calibration_info = bytes([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self.__read_force = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self.__reset_force = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01])
        self.__countsPerForce = 1000000
        self.__countsPerTorque = 1000000
        self.__scaleFactors_force = 15260
        self.__scaleFactors_torque = 92

    def setZero(self):
        self.__control.send(self.__reset_force)

    def readDate(self):
        self.__control.send(self.__read_force)
        force_info = self.__control.recv(16)
        header, status, ForceX, ForceY, ForceZ, TorqueX, TorqueY, TorqueZ = struct.unpack('!2H6h', force_info)
        Fx = ForceX * self.__scaleFactors_force / self.__countsPerForce
        Fy = ForceY * self.__scaleFactors_force / self.__countsPerForce
        Fz = ForceZ * self.__scaleFactors_force / self.__countsPerForce
        Tx = TorqueX * self.__scaleFactors_torque / self.__countsPerTorque
        Ty = TorqueY * self.__scaleFactors_torque / self.__countsPerTorque
        Tz = TorqueZ * self.__scaleFactors_torque / self.__countsPerTorque
        # N, NM
        Force_torque = np.array([Fx, Fy, Fz, Tx, Ty, Tz])
        return Force_torque


if __name__ == "__main__":
    test = ATIController()
    test.setZero()
    ft = test.readDate()
    print(ft)
    # show ft data
    from matplotlib import pyplot as plt
    import queue
    import  time
    q = queue.Queue(500)
    print('q.qsize:', q.qsize())
    plt.ion()
    print('plt.isinteractive():', plt.isinteractive())
    ax = plt.subplot()
    # while 1:
    i = 0

    data_list = []

    while i < 100:
        print('i:', i)
        i += 1
        ft = test.readDate()
        data_list.append(ft)
        time.sleep(0.01)
        q.put(ft)
        if q.full():
            q.get()
        print(np.array(q.queue)[:, 1])
        plt.plot(np.array(q.queue)[:, 0], label='fx')
        plt.plot(np.array(q.queue)[:, 1], label='fy')
        plt.plot(np.array(q.queue)[:, 2], label='fz')
        plt.legend(loc=3)

        # print(np.array(q.queue)[:, 0])
        # ax2 = ax.twinx()
        # ax2.plot(np.array(q.queue)[:, 3], label='tx')
        # ax2.plot(np.array(q.queue)[:, 4], label='ty')
        # ax2.plot(np.array(q.queue)[:, 5], label='tz')
        # plt.legend()

        plt.pause(0.005)
        plt.clf()

    # save data
    dirname = './data/ft'
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    

    # 将 data_list 转换为 numpy 数组
    data_array = np.array(data_list)
    data_reshaped = data_array.reshape(data_array.shape[0], -1)
    print(data_array)


    file_name ='init_force.txt'

    print(data_list)
    # np.savetxt(file_name, data_list)
    np.savetxt(dirname + '/' + file_name, data_reshaped)
    np.save(dirname + '/' + file_name, data_array)