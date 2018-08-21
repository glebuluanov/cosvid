from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
 
min_area = 250

vs = cv2.VideoCapture('case 6 (one after one).MOV')

 
firstFrame = None

while True:
	
	frame = vs.read()
	frame = frame if vs is None else frame[1]
	
	if frame is None:
		break
 
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	if firstFrame is None:
		firstFrame = gray
		continue
	
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 
	for c in cnts:
		if cv2.contourArea(c) < min_area:
			continue

		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
	cv2.imshow("Main", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Delta", frameDelta)
	key = cv2.waitKey(10) & 0xFF
	if key == ord('q'):
		break
		
vs.stop() if vs is None else vs.release()
cv2.destroyAllWindows()
  
