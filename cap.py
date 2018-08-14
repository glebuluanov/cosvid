import cv2
    
vid = cv2.VideoCapture('case 1(left-right).MOV')
    
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
   
while(1):
	ret, img = vid.read()
	if ret == True:
			img = fgbg.apply(img)
			img = cv2.resize(img, (0,0), fx=0.6, fy=0.6)
			cv2.imshow("vid", img)
			k = cv2.waitKey(50) & 0xff
			if k == 27:
				break
	else:
		break
cv2.destroyAllWindows()
  
