'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
'''

import numpy as np
import cv2

# Start the webcam
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('output.mp4')

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("{0} frames {1}x{2}".format(frames,source_width,source_height))

nframes = 10 # how many frames we want to analyze
target_width = source_width // 4
target_height = source_height // 4
skip_frames = frames / nframes - 1

def convert(frame):
    return cv2.resize(frame,(target_width,target_height))

# Take the first frame and convert it to gray
ret, frame = cap.read()
frm=convert(frame)
gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
# Create the HSV color image
hsvImg = np.zeros_like(frm)
hsvImg[..., 1] = 255

# Play until the user decides to stop
for i in range(frames-1):
    # Save the previous frame data
    previousGray = gray
    ret, frame = cap.read()
    frm = convert(frame)
    gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

    # Calculate the dense optical flow
    flow = cv2.calcOpticalFlowFarneback(previousGray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    vec = np.average(flow,axis=(0,1))
    mask = flow[:,:,0]==0 and flow[:,:,1]==0
    flow = flow-vec
    flow[mask]=0

    # Obtain the flow magnitude and direction angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Update the color image
    hsvImg[..., 0] = 0.5 * ang * 180 / np.pi
    hsvImg[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

    #cv2.imwrite('c:/temp/opt/orig{}.png'.format(i),frm)
    #cv2.imwrite('c:/temp/opt/opt{}.png'.format(i), rgbImg)

    # Display the resulting frame
    #cv2.imshow('dense optical flow', np.hstack((frame, rgbImg)))
    cv2.imshow('dense optical flow',rgbImg)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    #j=skip_frames
    #while ret and (j>0):
    #    ret, _ = cap.read()
    #    j-=1

# When everything is done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()