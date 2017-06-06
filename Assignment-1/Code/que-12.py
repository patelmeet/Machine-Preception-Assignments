import cv2
import numpy as np

#Loop to check for all files together
string = "portrait"     #filename prefix='portrait'
start = 1               #filename start index
end = 5                 #filename end index     "portrait1.jpg" to "portrait16.jpg"
for i in range(start,end+1):

    color_image = cv2.imread(string + str(i) + ".jpg",cv2.IMREAD_COLOR)     #read color image
    #resize image
    resized_image = np.copy(color_image)
    h,w,d = resized_image.shape
    while(h>700 or w>700):
        resized_image = cv2.resize(resized_image,(0,0), fx=0.5, fy=0.5)
        h,w,d = resized_image.shape

    gray_image = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)     #convert color image to gray image

    #define various haarcascade classifiers to detect face and eye
    haarcascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml')
    haarcascade_eye_tree_eyeglasses = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    haarcascade_frontalface_alt = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    haarcascade_frontalface_alt2 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    haarcascade_frontalface_alt_tree = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
    haarcascade_frontalface_default = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    haarcascade_lefteye_2splits = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    haarcascade_profileface = cv2.CascadeClassifier('haarcascade_profileface.xml')
    haarcascade_righteye_2splits = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

    #apply all haarcascade classifiers to gray image using various parameters
    #first parameter is image, second parameter is scale ratio after each stage of classifier
    #third parameter is minimum size of detected object
    result1 = haarcascade_eye.detectMultiScale(gray_image,1.2,4)
    result2 = haarcascade_eye_tree_eyeglasses.detectMultiScale(gray_image)
    result3 = haarcascade_frontalface_alt.detectMultiScale(gray_image,1.3,4)
    result4 = haarcascade_frontalface_alt2.detectMultiScale(gray_image,1.3,4)
    result5 = haarcascade_frontalface_alt_tree.detectMultiScale(gray_image,1.3,4)
    result6 = haarcascade_frontalface_default.detectMultiScale(gray_image,1.3,4)
    result7 = haarcascade_lefteye_2splits.detectMultiScale(gray_image)
    result8 = haarcascade_profileface.detectMultiScale(gray_image,1.3,4)
    result9 = haarcascade_righteye_2splits.detectMultiScale(gray_image,1.2,4)

    #draw rectangle over detected region--for debug purpose
    """
    for (x, y, w, h) in result1:
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    for (x, y, w, h) in result2:
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in result3:
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for (x, y, w, h) in result4:
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    for (x, y, w, h) in result5:
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 0, 255), 2)
    for (x, y, w, h) in result6:
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    for (x, y, w, h) in result7:
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 0), 2)
    for (x, y, w, h) in result8:
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
    for (x, y, w, h) in result9:
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 125, 250), 2)
    """
    #if any classifier doesn't detect anything then it is not portrait
    if(len(result1)==0 and len(result2)==0 and len(result3)==0 and len(result4)==0 and len(result5)==0 and len(result6)==0 and len(result7)==0 and len(result8)==0 and len(result9)==0):
        hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)    #convert resized color image to HSV image
        value = hsv[:,:,2]      #extract Value channel from HSV image

        total_pixel = h * w;    #total number of pixel in resized image

        #apply binary thresholding to get dark pixels, dark pixels have 0 value in thresholded_image,all other are 255
        _, thresholded_image = cv2.threshold(value, 60, 255, cv2.THRESH_BINARY)
        dark_pixels = np.where(thresholded_image == 0)  #get cordinates of dark pixels
        dark_pixel_count = len(dark_pixels[0])          #count number of dark pixels

        if (dark_pixel_count > total_pixel / 2):    #if more than half image is covered by dark pixels then it is night picture
            print string + str(i) +' - night'
        else:
            print string + str(i) +' - landscape'   #otherwise landscape
    else :
        print string + str(i) +' - portrait'
