import cv2
import numpy as np

#image-1
binary_image=np.zeros((500,500),np.uint8)
x=32

#draw lines of different widths within same image
for i in range(1,16):
    cv2.line(binary_image,(0,x),(500,x),255,i)
    cv2.line(binary_image,(x,0),(x,500),255,i)
    x = x+32
cv2.imwrite("binary_image1.jpg",binary_image)

#Create binary image of different width of vertical and horizontal strips
binary_image=np.zeros((200,200),np.uint8)
x=5
width = 5
while(x<200):
	cv2.line(binary_image,(0,x),(200,x),255,width)
	cv2.line(binary_image,(x,0),(x,200),255,width)
	x = x + 14

cv2.imwrite("binary_image5.jpg",binary_image)

#image-2
binary_image = np.ones((200,200),np.uint8)

for i in range(0,200):
    for j in range(0, 200):
        binary_image[i][j] = ((abs(i-200/2)+abs(j-200/2))%25)*10;

#cv2.imwrite('binary_image2.jpg',binary_image)
