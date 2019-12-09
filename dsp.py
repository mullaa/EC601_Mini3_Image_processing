import cv2
import matplotlib.pyplot as plt
import sys

image = cv2.imread('./images.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(image, (7,7), 0)


cv2.imshow("Image", image)
cv2.imshow("Gray Image", gray_image)
cv2.imshow("Blurred Image", blurred_image)

canny = cv2.Canny(blurred_image, 10, 30)
cv2.imshow("Canny with low thresholds", canny)
canny2 = cv2.Canny(blurred_image, 50, 150)
cv2.imshow("Canny with high thresholds", canny2)

contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print("Number of objects found = ", len(contours))
cv2.drawContours(image, contours, -1, (0,255,0), 2)
cv2.imshow("objects Found", image)
cv2.waitKey(0)  # waits until a key is pressed

# Face Detection In Python Using OpenCV
def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#load cascade classifier training file for haarcascade
haar_face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
#load test iamge
test1 = cv2.imread('./test1.jpg')
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);
#print the number of faces found
# print('Faces found: ', len(faces))
# Next, let's loop over the list of faces (rectangles) it returned and draw those rectangles using built in
# OpenCV **`rectangle`** function on our original colored image to see if it detected the right faces.
#go over list of faces and draw them as rectangles on original colored img
for (x, y, w, h) in faces:
    cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)
# Display the original image to see rectangles drawn and verify that detected faces are really faces and not false positives.
#conver image to RGB and show image
plt.imshow(convertToRGB(test1))
plt.show()