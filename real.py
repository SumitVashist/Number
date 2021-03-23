import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import imutils
import time


cv2.namedWindow('contour', cv2.WINDOW_NORMAL)
cv2.resizeWindow('contour', 1980, 1080)


# img = cv2.imread("/home/keenu/Downloads/shape.png")
#cam = cv2.VideoCapture(2)
#_, frame = cam.read()


camera = PiCamera()
rawCapture = PiRGBArray(camera)

time.sleep(0.1)
# grab an image from the camera
camera.capture(rawCapture, format="rgb")
image = rawCapture.array

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img = img_gray[300:800, 400:840]

_, threshold = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((5,5), np.uint8) 
img_erosion = cv2.erode(threshold, kernel, iterations=1) 
# img_dilation = cv2.dilate(threshold, kernel, iterations=1) 
diff = cv2.medianBlur(img_erosion, 5)

edge = cv2.Canny(diff, 175, 175)



contours,hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#img=cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR)


for cnt in contours:
    area = cv2.contourArea(cnt)

    if area > 1000 and area <1200:
        img = cv2.drawContours(img, cnt, -1, (255,0, 255), 2)
        print (area)
        cv2.putText(img , (" OK "),(200,250),cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,0,0),1)
        break


    elif area <1000 and area > 700 :
        img = cv2.drawContours(img, cnt, -1, (255,0, 255), 2)
        print (area)
        cv2.putText(img , (" Not OK "),(200,250),cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,0,0),1)
                


cv2.imshow("contour", img)
cv2.waitKey(0)
