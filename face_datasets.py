# Import OpenCV2 for image processing
# Import os for file path
import cv2
import os

import numpy as np

from PIL import Image
# Open Camera
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, one face id
face_id = input("Enter face id ")
face= raw_input("Enter name of Person ")

# Initialize sample face image
count = 0

if not os.path.exists('dataset/'):
	os.makedirs('dataset/')

# Start looping
while(True):

    # Capture video frame
    boo, image_frame = vid_cam.read()

    # Convert frame to grayscale
    image_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(image_frame, 1.1 , 5)

    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # Increment sample face image
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id)+'.' +face+ '.' + str(count) + ".jpg", image_frame[y:y+h,x:x+w])

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms8
    if cv2.waitKey(100) == 27:
        break

    # If image taken reach 100, stop taking video
    elif count>100:
        break
f= open("data.csv","a")
ID= str(face_id)
string= ID+','+face+'\n'
f.write(string)
f.close()

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()


# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Using prebuilt frontal face training model, for face detection
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Create method to get the images and label data
def getImagesAndLabels(path):

    # Get all file path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    # Initialize empty face sample
    faceSamples=[]
    
    # Initialize empty id
    ids = []

    # Loop all the file path
    for imagePath in imagePaths:

        # Get the image and convert it to grayscale
        PIL_img = Image.open(imagePath)

        # PIL image to numpy array
        img_numpy = np.array(PIL_img,'uint8')

        # Get the image id
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Get the face from the training images
        faces = detector.detectMultiScale(img_numpy)

        # Loop for each face, append to their respective ID
        for (x,y,w,h) in faces:

            # Add the image to face samples
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            # Add the ID to IDs
            ids.append(id)

    # Pass the face array and IDs array
    return faceSamples,ids

# Get the faces and IDs
faces,ids = getImagesAndLabels('dataset')

# Train the model using the faces and IDs
recognizer.train(faces, np.array(ids))

# Save the model into trainer.yml
dir = 'trainer/'
if not os.path.exists(dir):
        os.makedirs(dir)
recognizer.save('trainer/trainer.yml')
