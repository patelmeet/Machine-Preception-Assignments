import cv2
import numpy as np
#import matplotlib.pyplot as plt

cap = cv2.VideoCapture("road.mp4")  #To capure video - Input stream
ret, frame = cap.read()
h, w, d = frame.shape
fourcc = cv2.cv.CV_FOURCC(*'XVID')	#Video Codec
out = cv2.VideoWriter('output.avi' , fourcc , 30 , (w,h) , True) #Output Stream to save video

while True:
    ret, frame = cap.read()     #read next frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #convert to gray-scale

    # apply gaussion bluring to reduce noice
    gaussion_blurred_image = cv2.GaussianBlur(gray, (7, 7), 0)

    #apply binary threshold to get white area
    _, thresolded_image = cv2.threshold(gaussion_blurred_image, 180, 255, cv2.THRESH_BINARY)
    thresolded_image[0:h/4, 0:w] = 0    #ignore upper part of image to remove sky

    #apply HoughLine transformation to get lines only
    hough_lines = cv2.HoughLinesP(thresolded_image, 1, np.pi / 180, 1, minLineLength=3)
    #draw lines on image which is given by hough transformation
    for x1, y1, x2, y2 in hough_lines[0]:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("road",frame)    #show output
    out.write(frame)    #write output to Output Stream

    if cv2.waitKey(1) & 0xFF == ord('q'):  # wait for key-'q'
        break

cap.release()       #release Input Stream
out.release()       #releas Output Stream and save video
cv2.destroyAllWindows()
