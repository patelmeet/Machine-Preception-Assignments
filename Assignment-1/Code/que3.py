import cv2
import matplotlib.pyplot as plt

image = cv2.imread("2.jpg",cv2.IMREAD_COLOR)   #read image.

Lab_image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)   #convert original BGR image to L*a*b* image

plt.subplot(1,2,1),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),'gray'),plt.title('Original Image'),plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(cv2.cvtColor(Lab_image, cv2.COLOR_BGR2RGB),'gray'),plt.title('L*a*b* Image'),plt.xticks([]), plt.yticks([])
plt.show()


