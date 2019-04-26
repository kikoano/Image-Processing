import numpy as np
import cv2

# Video background subtraction

# Read video
cap = cv2.VideoCapture('surveillance.mpg')
# Define the codec and create VideoWriter object
ret, frame = cap.read()
background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
alpha = 0.95
height, width, channels = frame.shape
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30.0,(width, height),True)

while(cap.isOpened()):
	ret, frame = cap.read()
	if ret==True:
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		background = cv2.addWeighted(background,alpha,frame_gray,1-alpha,0)
	
		height, width = frame_gray.shape
		diffImg = np.zeros((height,width), np.uint8)
		cv2.absdiff(frame_gray,background,diffImg)
	
		ret1,thresh1 = cv2.threshold(diffImg,80,255,cv2.THRESH_BINARY)
	
		if np.sum(thresh1) > 0:
			out.write(frame)
		
		#cv2.imshow('frame',diffImg)
		cv2.imshow('frame',frame)
		if cv2.waitKey(24) & 0xFF == ord('q'):
			break
	else:
		break

cap.release()
out.release()
cv2.destroyAllWindows()