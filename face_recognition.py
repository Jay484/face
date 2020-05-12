# Import OpenCV2 for image processing
# Import os for file path
import cv2
import os
import pyttsx
import cv2
# Import numpy for matrices calculations
import numpy as np

#import pandas for reading csv file
import pandas


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

namep=""
# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")

# Load the trained mode
recognizer.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)

head=['id','face']
data= pandas.read_csv('data.csv',names=head)
# Loop
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.1 ,5)

    # For each face in faces
    for(x,y,w,h) in faces:
        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

	head=['id','name']
    	data= pandas.read_csv('data.csv',names=head)
	i=0
       	# Check the ID if exist
	while i< data.shape[0]:
#		print data.iloc[i][0]
#		print data.iloc[i][1]
        	if(Id == int(data.iloc[i]['id'])):
      	   		name = str(data.iloc[i]['name'])
		i+=1

#        Put text describe who is in the picture
       	cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)


	if confidence>=60:
		cv2.putText(im,name, (x,y-40), font, 1, (255,255,255), 3)

	if(namep!=name):
		engine = pyttsx.init()
		voices= engine.getProperty('voices')
		engine.setProperty('rate',150)
		namep=name
		print name
	        #engine.say(name)
		engine.runAndWait()
		del engine
    # Display the video frame with the bounded rectangle
    cv2.imshow('image',im)

    # If 'esc' is pressed, close program
    if cv2.waitKey(10) == 27:
        	break
# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()


