import cv2 as cv
import matplotlib.pyplot as plt


img1_bgr=cv.imread("Images/Friends.jpg",1)
img1_rgb= cv.cvtColor(img1_bgr, cv.COLOR_BGR2RGB)

faceCascade=cv.CascadeClassifier("XML files/haarcascade_frontalface_alt.xml")
detectedFaces=faceCascade.detectMultiScale(img1_rgb,1.1,4)
print(detectedFaces)

for (length,breadth,width,height) in detectedFaces:
    cv.rectangle(img1_rgb,(length,breadth),(length+width,breadth+height),(255,0,0),10)
 
plt.imshow(img1_rgb)

