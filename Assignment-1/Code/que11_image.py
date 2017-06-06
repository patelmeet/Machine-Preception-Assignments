import cv2
import numpy as np
import matplotlib.pyplot as plt

color_image = cv2.imread("road23.jpg",cv2.IMREAD_COLOR)     #read color image

#resize image if it is too big
resized_image = np.copy(color_image)        #copy image
h,w,d = resized_image.shape                 #get shapes of image
while(h>1000 or w>1000):
    resized_image = cv2.resize(resized_image,(0,0), fx=0.5, fy=0.5)     #resize image
    h,w,d = resized_image.shape

plt.subplot(1,3,1),plt.imshow(cv2.cvtColor(resized_image,cv2.COLOR_BGR2RGB),'gray'),plt.title('Original Image'),plt.xticks([]), plt.yticks([])

grayimage = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)      #convert color image to gray image

# apply gaussion bluring to reduce noice
gaussion_blurred_image = cv2.GaussianBlur(grayimage,(7,7),0)

#apply binary threshold to get white area
_,thresolded_image = cv2.threshold(gaussion_blurred_image,180,255,cv2.THRESH_BINARY)
thresolded_image[0:h*30/100,0:w] = 0     #ignore upper part of image to remove sky

#apply HoughLine transformation to get lines only
hough_lines = cv2.HoughLinesP(thresolded_image,1,np.pi/180,5,lines=1,minLineLength=15,maxLineGap=5)

#draw lines on image which is given by hough transformation
for x1,y1,x2,y2 in hough_lines[0]:
    cv2.line(resized_image,(x1,y1),(x2,y2),(0,255,0),2)

plt.subplot(1,3,2),plt.imshow(grayimage,'gray'),plt.title(' Gray Image'),plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(gaussion_blurred_image,'gray'),plt.title('Gaussion Blurred Image'),plt.xticks([]), plt.yticks([])
plt.show()
plt.subplot(1,2,1),plt.imshow(thresolded_image,'gray'),plt.title('Thresolded Image'),plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(cv2.cvtColor(resized_image,cv2.COLOR_BGR2RGB),'gray'),plt.title('Output'),plt.xticks([]), plt.yticks([])
plt.show()
