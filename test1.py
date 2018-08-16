#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import cv2
import numpy as np
count = 0

#импортируем видео из папки, где находится название проекта.py или указываем аргументом путь до файла
vid = cv2.VideoCapture('Image from iOS.MOV')
#запускаем ОРБ детектор и декодер и БФ сопоставитель
ret, img1 = vid.read()
bf = cv2.BFMatcher() 
orb = cv2.ORB_create()

while(1):
	
	ret, img2 = vid.read()
	
	#ищем особые точки и их описание
	
	if ret == True:
		
		kp1 = orb.detect(img1)
		kp1, des1 = orb.compute(img1, kp1)
		kp2 = orb.detect(img2)
		kp2, des2 = orb.compute(img2, kp2)
	#находим соответствия и выбираем хорошие
		matches = bf.knnMatch(des1,des2, k=2)

		good = []
		for m,n in matches:
			if m.distance < 0.3*n.distance:
				good.append([m])
	#рисуем соответствия
		img = img2
		img = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img,flags=2)
	#выводим изображение в удобном размере
		img1 = img2
		img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
		cv2.imwrite("frame%d.jpg" % count, img)
		cv2.imshow("vid", img)
		count = count + 1
		if cv2.waitKey(10) & 0xff == ord('q'):
			break
	
	else:
		break
cv2.destroyAllWindows()
#
