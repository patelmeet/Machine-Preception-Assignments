import cv2

from matplotlib import pyplot as plt

image = cv2.imread("kohli.png",cv2.IMREAD_COLOR)     #read color image

# Apply gaussian blur using default function of cv2 with various parameters
# kernal size used to calculate mean
# sigmaX and sigmaY used to calculate variance
# when sigmaX=0 and sigmaY=0, it takes kernal size to calculate variance.
blur1 = cv2.GaussianBlur(image,(5,5),0)
blur2 = cv2.GaussianBlur(image,(9,9),0)
blur3 = cv2.GaussianBlur(image,(9,9),sigmaX=32,sigmaY=32)
blur4 = cv2.GaussianBlur(image,(15,15),0)
blur5 = cv2.GaussianBlur(image,(15,15),sigmaX=16,sigmaY=16)

#plot all the images
plt.subplot(2,3,1),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),'gray'),plt.title('Original Image'),plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(cv2.cvtColor(blur1, cv2.COLOR_BGR2RGB),'gray'),plt.title('kernal=5 deviation=0'),plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(cv2.cvtColor(blur2, cv2.COLOR_BGR2RGB),'gray'),plt.title('kernal=9 deviation=0'),plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(cv2.cvtColor(blur3, cv2.COLOR_BGR2RGB),'gray'),plt.title('kernal=9 deviation=32'),plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(cv2.cvtColor(blur4, cv2.COLOR_BGR2RGB),'gray'),plt.title('kernal=15 deviation=0'),plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(cv2.cvtColor(blur5, cv2.COLOR_BGR2RGB),'gray'),plt.title('kernal=15 deviation=16'),plt.xticks([]), plt.yticks([])
plt.show()
