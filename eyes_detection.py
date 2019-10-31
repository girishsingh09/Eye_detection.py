''' eyes detection '''
# import libraries of python opencv 
import cv2

# To open webcam
camera=cv2.VideoCapture(0)
# capture frame by detect
check,detect=camera.read()
# save detect with picture.jpg name
cv2.imwrite("picture.png",detect)
# to Off webcam
del camera
# Use trained xml classifier for eyes detection in the picture
eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")
# take image 
image=cv2.imread("picture.png")
#Convert the image into gray image without color
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#detect eyes in the gray image
eyes=eye_cascade.detectMultiScale(gray_image,1.3,9)
# draw a rectangle boxe around the eyes
for (x,y,w,h) in eyes:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

# show image window
cv2.imshow("Eye detect",image)
cv2.waitKey(0) # wait key of keyboard
cv2.destroyAllWindows()   #destroy all windows
