import cv2 as cv
import matplotlib.pyplot as plt


img_bgr=cv.imread("Images/catto.webp",1)
img_rgb= cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

faceCascade=cv.CascadeClassifier("XML files/haarcascade_frontalcatface.xml")
detectedFaces=faceCascade.detectMultiScale(img_rgb,1.1,4)
print(detectedFaces)
for (length,breadth,width,height) in detectedFaces:
    cv.rectangle(img_rgb,(length,breadth),(length+width,breadth+height),(255,0,0),10)
 
plt.imshow(img_rgb)