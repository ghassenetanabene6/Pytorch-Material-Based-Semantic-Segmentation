import cv2 as cv
import numpy as np


cap = cv.VideoCapture('video3.mp4')
i = 0
frame_rgb, frame_gray, histogram_old = None, None, None
next_frame = False

corr = []
while True:
	ret, frame_rgb = cap.read()
	if not ret:
		print("Stream ended !")
		break
	name = 'frame_'+str(i).zfill(6)+'.jpg'
	frame_gray = cv.cvtColor(frame_rgb, cv.COLOR_BGR2GRAY)
	histogram = cv.calcHist(frame_gray, [0], None, [64], [0, 256], accumulate = False)
	if i != 0:
		retval = cv.compareHist(histogram, histogram_old, cv.HISTCMP_CHISQR)
		print(retval)
		if retval > 1000 or next_frame:
			next_frame = False
			cv.imshow('frame', frame_rgb)
			key = cv.waitKey(10000)
			if key == ord('d'):#d
				next_frame = True
			elif key == ord('s'):#s
				cv.imwrite('./test/'+name, frame_rgb)
			elif key == ord('q'):#s
				break
		else:
			continue
	
	histogram_old = histogram
	i+=1
