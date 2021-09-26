import pyttsx3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
#from imutils.video import VideoStream
import numpy as np
import argparse
#import imutils
import time
import cv2
import os
#import sys
#import cv2
import playsound
from threading import Thread
TEMP_TUNER = 2.25
TEMP_TOLERENCE = 67
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def process_face(frame):
    
    frame = ~frame
    heatmap = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
    
    heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    ret, binary_thresh = cv2.threshold(heatmap_gray, 200, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    image_erosion = cv2.erode(binary_thresh, kernel, iterations=1)
    image_opening = cv2.dilate(image_erosion, kernel, iterations=1)
    

    image_with_rectangles = np.copy(heatmap)
    
    return image_with_rectangles



def convert_to_temperature(pixel_avg):
    """
    Converts pixel value (mean) to temperature depending upon the camera hardware
    """
    f = pixel_avg / TEMP_TUNER
    c = (f - 32) * 5/9
    
    return f


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		preds = maskNet.predict(faces)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


# Parse the arguments from command line
arg = argparse.ArgumentParser(description='Social distance detection')

arg.add_argument("-a", "--alarm", type=str, default="alarm.wav",	help="path alarm .WAV file")

arg.add_argument('-c', '--confidence', type = float, default = 0.2, help='Set confidence for detecting objects')

arg.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")

arg.add_argument("-m1", "--model1", type=str, default="model/mask_detector.model", help="path to trained face mask detector model")

#arg.add_argument("-c", "--confidence", type=float, default=0.5,	help="minimum probability to filter weak detections")

args = vars(arg.parse_args())

ALARM_ON = False

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

print("[INFO] loading face detector model...")

prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])

weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model1"])

# Load model
#print("\nLoading model...\n")
#network = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("\nStreaming video using device...\n")


# Capture video from file or through device
cap = cv2.VideoCapture(0)


frame_no = 0

while True:

    frame_no = frame_no+1

    # Capture one frame after another
    ret, frame = cap.read()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    frame_for_thermal = np.copy(frame)
    #frame = imutils.resize(frame, width=1000)

	# detect faces in the frame and determine if they are wearing a

    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    #PROCESSING FOR FRAME FOR THERMAL
    frame_for_thermal = cv2.flip(frame_for_thermal, 180)
    gray = cv2.cvtColor(frame_for_thermal, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    output = process_face(frame_for_thermal)

    faceNet.setInput(blob)
    detections = faceNet.forward()
                
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

   	# loop over the detected face locations and their corresponding
   # locations
    for (box, pred) in zip(locs, preds):
   		# unpack the bounding box and predictions
       (startX_mask, startY_mask, endX_mask, endY_mask) = box
       (mask, withoutMask) = pred
   
   		# determine the class label and color we'll use to draw
   		# the bounding box and text
       if mask > withoutMask:
           label_mask = "Mask" 
           color = (0, 255, 0) 
           ALARM_ON = False
       else:
           label_mask="No Mask"
           color = (0, 0, 255)
           if not ALARM_ON:
               ALARM_ON = True
               if args["alarm"] != "":
                   t = Thread(target=sound_alarm, args=(args["alarm"],))
                   t.deamon = True
                   t.start()
                   time.sleep(2)
                   speak("Wear Your Mask Immediately")

   #label = "{}: {:.2f}%".format('person', confidence * 100)
       label_mask = "{}: {:.2f}%".format(label_mask, max(mask, withoutMask) * 100)
   #print("{}".format(label))
       cv2.putText(frame, label_mask, (startX_mask, startY_mask - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
       cv2.rectangle(frame, (startX_mask, startY_mask), (endX_mask, endY_mask), color, 2)

    for (x,y,w,h) in faces:

        roi = output[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

         # Mask is boolean type of matrix.
        mask = np.zeros_like(roi_gray)

        # Mean of only those pixels which are in blocks and not the whole rectangle selected
        mean = convert_to_temperature(np.mean(roi_gray))

        # Colors for rectangles and textmin_area
        temperature = round(mean, 2)
    
        if temperature < TEMP_TOLERENCE:
            color = (0, 255, 0) 
            ALARM_ON = False
        else:
            color = (0, 0, 255)
            if not ALARM_ON:
               ALARM_ON = True
               if args["alarm"] != "":
                   t = Thread(target=sound_alarm, args=(args["alarm"],))
                   t.deamon = True
                   t.start()
                   time.sleep(2)
                   speak("Consult your doctor immediately")

        # Draw rectangles for visualisation
        frame = cv2.rectangle(frame, (x+10, y+10), (x+w, y+h), color, 10)
        cv2.putText(frame, "{} F".format(temperature), (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)

    # Show frame
    cv2.imshow('Frame', frame)
    cv2.resizeWindow('Frame',800,600)

    key = cv2.waitKey(1) & 0xFF

    # Press `q` to exit
    if key == ord("q"):
        break

# Clean
cap.release()
cv2.destroyAllWindows()
