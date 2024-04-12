#!/usr/bin/env python
from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import cv2  
import numpy as np
import imutils
from collections import deque
import math
from calibkinect import depth2xyzuv, xyz_matrix
import time
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.common import kinematic_kf
from filterpy.kalman import IMMEstimator


# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (24, 45, 67)
greenUpper = (93, 255,255)

buffer = 64
pts = deque(maxlen= buffer)
pos_buf = deque(maxlen= buffer)
vel_buf = deque(maxlen= buffer)
acc_buf = deque(maxlen= buffer)
time_buf = deque(maxlen= buffer)

GRAVITY = 9.81
PIXEL_OFFSET = 2
OUTPUT = 30

RECORD_TIME = 5

OUTPUT_FILE = 'OUTPUT_FILE.txt'

def kinematics(pos, vel, acc):
    #pos_vec = 
    pass
# 
def average_reads(position_list, cur_time):
    if (len(position_list)):
        tot = len(position_list)
        x = 0
        y = 0
        z = 0
        num_invalid = 0

        for xyz in position_list:
            x += xyz[0]
            y += xyz[1]
            if (xyz[2] < 0):
                z += xyz[2]
            else:
                num_invalid +=1
        if num_invalid == tot:
            return([cur_time, x/tot, y/tot, 0])
        else:
            return([cur_time, x/tot, y/tot, z/(tot-num_invalid)])
    else:
        return([cur_time, 0, 0, 0])
    
def max_min_reads(position_list):
    max_x = max(position_list[:][0])
    min_x = min(position_list[:][0])

    max_y = max(position_list[:][1])
    min_y = min(position_list[:][1])

    max_z = max(position_list[:][2])
    min_z = min(position_list[:][2])

    return([[min_x, max_x], [min_y, max_y], [min_z, max_z]])

def get_single_point_xyz(u, v, z, cur_time):
    #C = np.vstack((u, v, z, 0*u+1))
    #X,Y,Z,W = np.dot(xyz_matrix(),C)
    #X,Y,Z = X/W, Y/W, Z/W
    
    minDistance = -10
    scaleFactor = .0021

    Z = 0.1236 * math.tan(z / 2842.5 + 1.1863)

    X = (v - 640 / 2) * (z + minDistance) * scaleFactor
    Y = (u - 480 / 2) * (z + minDistance) * scaleFactor

    return ([cur_time, X,Y,Z])

def doloop(file):
    first = True
    RECORDING = False
    recording_time = 0
    #output data every 30 frames, too sparatic to figure out whats going on otherwise
    counter = 0
    global depth, rgb
    while True:
        # Get a fresh frame
        (depth,_), (rgb,_) = get_depth(), get_video()
        
        #get timestamp
        cur_time = time.time_ns()

        #color tracking
        # resize the frame, blur it, and convert it to the HSV
        # color space
        #frame = imutils.resize(rgb, width=600)
        frame = rgb
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
            c = max(cnts, key=cv2.contourArea)
            #print(c)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            
            #print("area diff: {}, radius area: {}, contour area: {}".format(abs((math.pi * radius**2) - cv2.contourArea(c)), (math.pi * radius**2), cv2.contourArea(c)))
            
            # only proceed if the radius meets a minimum size and the difference in area of the placed circle and contour is less than threshold
            # ball will be tracked within a couple meters of the camera with a contour greater than 100 
            # could add something that encorporates the depth data to scale all contours to a specified z distance to determine the largest contour
            if radius > 10 and cv2.contourArea(c) > 100:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                u = round(y)
                v = round(x)
                #if ball center is inside image
                if (0 <= u and u < 480) and (0 <= v and v < 640):
                    counter += 1
                    #get real world position data from grid of pixels
                    rad = round(radius)
                    u_,v_ = np.mgrid[u-rad if u-rad > 0 else 0:u+rad if u+rad < 480 else 479,v-rad if v-rad>0 else 0:v+rad if v+rad< 640 else 639]
                    xyz,uv = depth2xyzuv(depth[u_,v_], u_, v_)
                    avrg_xyz = average_reads(xyz, cur_time)

                    #get single point real world position data(center point) 
                    point = get_single_point_xyz(u, v, depth[u][v], cur_time)

                    
                    pos_buf.appendleft(avrg_xyz)

                    if (len(pos_buf) > 1 and len(pos_buf[1])):
                        #print('time0: {}, time1: {}'.format(pos_buf[0][0], pos_buf[1][0]))
                        x_vel = ((pos_buf[0][1] - pos_buf[1][1])/(pos_buf[0][0] - pos_buf[1][0]))*1e9
                        y_vel = ((pos_buf[0][2] - pos_buf[1][2])/(pos_buf[0][0] - pos_buf[1][0]))*1e9
                        z_vel = ((pos_buf[0][3] - pos_buf[1][3])/(pos_buf[0][0] - pos_buf[1][0]))*1e9
                        vel_buf.appendleft([x_vel, y_vel, z_vel])
                    else:
                        vel_buf.appendleft(None)

                    if(len(vel_buf) > 1 and vel_buf[1]):
                        x_acc =  ((vel_buf[0][0] - vel_buf[1][0])/(pos_buf[0][0] - pos_buf[2][0]))*1e9
                        y_acc =  ((vel_buf[0][1] - vel_buf[1][1])/(pos_buf[0][0] - pos_buf[2][0]))*1e9
                        z_acc =  ((vel_buf[0][2] - vel_buf[1][2])/(pos_buf[0][0] - pos_buf[2][0]))*1e9
                        acc_buf.appendleft([x_acc, y_acc, z_acc])
                    else:
                        acc_buf.appendleft(None)
                    
                    #Save data if s is hit
                    if (RECORDING):
                        data_out = pos_buf[0] + vel_buf[0] + acc_buf[0]
                        out_str = ''
                        for i in data_out:
                            out_str += str(i) + ', '
                        out_str += '\n'
                        file.write(out_str)

                    if (time.time() - recording_time  > RECORD_TIME and RECORDING):
                        print("DONE RECORDING")
                        RECORDING = False
                        recording_time = 0
                    
                    #ouput data after certain amount of frames
                    if (counter >= OUTPUT):
                        print("\n")
                        #print("Image position values: (x: {}, y: {}, depth: {})".format(x, y, depth[u][v]))
                        print("Real world single point: (x: {}, y: {}, z: {})".format( point[1], point[2], point[3]))
                        print("Average read data, X: {}, y: {}, Z: {}, depth read: {}".format(avrg_xyz[1], avrg_xyz[2], avrg_xyz[3], depth[u,v]))
                        if (len(vel_buf)):
                            print('Velocity data, x: {}, y: {}, z: {}'.format(vel_buf[0][0], vel_buf[0][1], vel_buf[0][2]))
                        if (len(acc_buf)):
                            print('Acceleration data, x: {}, y: {}, z: {}'.format(acc_buf[0][0], acc_buf[0][1], acc_buf[0][2]))
                        counter = 0
                
                #draw the circle and center
                cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # update the points queue
        pts.appendleft(center)

        #update depth data of ball

        '''
        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue
            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        ''' 
        cv2.circle(frame, (314, 239), 3, (255, 0, 0), -1)
        cv2.circle(frame, (319, 239), 3, (255, 0, 0), -1)

        mask_3_channel = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)

        # Build a two panel color image
        d3 = np.dstack((depth,depth,depth)).astype(np.uint8)
        da = np.hstack((d3,frame, mask_3_channel))
        
        if (first):
            print(depth.shape)
            print(frame.shape)
            print(mask_3_channel.shape)
            first = False

        # Simple Downsample
        cv2.imshow('both',np.array(da[:,:,::-1]))
        key = cv2.waitKey(5) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            file.close()
            break
        #save data
        if key == ord("s"):
            print("RECORDING DATA")
            RECORDING = True
            recording_time = time.time()


if __name__ == '__main__':
    file = open(OUTPUT_FILE, "w")
    doloop(file)