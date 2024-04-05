#!/usr/bin/env python
from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import freenect 
import cv2  
import numpy as np
import imutils
from collections import deque
import math
from calibkinect import depth2xyzuv, xyz_matrix

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (24, 45, 67)
greenUpper = (93, 255,255)

buffer = 64
pts = deque(maxlen= buffer)
depth_data = deque(maxlen= buffer)

def kalman_filter():
    pass


def doloop():
    first = True
    global depth, rgb
    while True:
        # Get a fresh frame
        (depth,_), (rgb,_) = get_depth(), get_video()

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

                offset = 2

                #u, v = np.mgrid[int(round(x)-offset):int(round(x)+offset), int(round(y)-offset):int(round(y)+offset)]
                u = round(x)
                v = round(y)

                if (0 <= round(x) and round(x) <= 480) and (0 <= round(y) and round(y) <= 640):
                    #pos, _uv = depth2xyzuv(depth[v, u], u, v)
                    C = np.vstack((u, v, depth[u][v], 0*u+1))
                    X,Y,Z,W = np.dot(xyz_matrix(),C)
                    X,Y,Z = X/W, Y/W, Z/W
                    depth_data.appendleft([X,Y,Z])
                    print("Image position values: (x: {}, y: {}, depth: {})".format(x, y, depth[round(x)][round(y)]))
                    print("Real world: (x: {}, y: {}, z: {})".format( X, Y, Z))
                cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # update the points queue
        pts.appendleft(center)

        #update depth data of ball

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
            # show the frame to our screen
            #cv2.imshow("Frame", frame)

        mask_3_channel = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)

        # Build a two panel color image
        d3 = np.dstack((depth,depth,depth)).astype(np.uint8)
        da = np.hstack((d3,frame, mask_3_channel))
        
        if (first):
            print(depth)
            print(frame.shape)
            print(mask_3_channel.shape)
            first = False

        # Simple Downsample
        #cv.ShowImage('both',np.array(da[::2,::2,::-1]))
        #cv.WaitKey(5)
        cv2.imshow('both',np.array(da[::2,::2,::-1]))
        #cv2.waitKey(5)
        key = cv2.waitKey(5) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break


if __name__ == '__main__':
    
    doloop()