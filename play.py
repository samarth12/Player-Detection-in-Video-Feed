import cv2
import os
import numpy as np


cap = cv2.VideoCapture('match.mp4')
idx =0
count = 0
truth = True

def filtercontours(contours):
    playercontours = list()
    for c in contours:
        rect = cv2.boundingRect(c)
        # if contour is too big or too small its not player
        if (rect[2] < 7 or rect[3] < 20) or (rect[2] > 60 or rect[3] > 100): continue
        playercontours.append(c)
    return playercontours

while True:

	ret, img = cap.read()
	img = np.array(img, dtype=np.uint8)
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

	lower_green = np.array([35,100, 100])
	upper_green = np.array([55, 255, 255])

	lower_blue = np.array([110,50,50])
	upper_blue = np.array([130,255,255])


	lower_red = np.array([165,100,100])
	upper_red = np.array([185,255,255])


	lower_yellow = np.array([20,100,100])
	upper_yellow = np.array([40,255,255])


	mask = cv2.inRange(hsv, lower_green, upper_green)

	result = cv2.bitwise_and(img, img, mask=mask)

	result_bgr = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)
	result_grayscale = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)


	kernel = np.ones((13,13),np.uint8)
	threshold = cv2.threshold(result_grayscale,127,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)



	img_save,contours,hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


	prev = 0
	font = cv2.FONT_HERSHEY_SIMPLEX


	contours = filtercontours(contours)
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		player_img = img[y:y+h,x:x+w]
		player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)

		mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
		player1_result = cv2.bitwise_and(player_img, player_img, mask=mask1)
		player1_result = cv2.cvtColor(player1_result,cv2.COLOR_HSV2BGR)
		player1_result = cv2.cvtColor(player1_result,cv2.COLOR_BGR2GRAY)
		count_chelsea = cv2.countNonZero(player1_result)

		mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
		player2_result = cv2.bitwise_and(player_img, player_img, mask=mask2)
		player2_result = cv2.cvtColor(player2_result,cv2.COLOR_HSV2BGR)
		player2_result = cv2.cvtColor(player2_result,cv2.COLOR_BGR2GRAY)
		count_liverpool = cv2.countNonZero(player2_result)

		if(count_chelsea >= 5):

			cv2.putText(img, 'Chelsea', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)

		if(count_liverpool>=5):

			cv2.putText(img, 'Liverpool', (x-2, y-2), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
	"""
if((h>=1 and w>=1) and (h<=30 and w<=30)):
	ball_img = img[y:y+h,x:x+w]

	ball_hsv = cv2.cvtColor(ball_img,cv2.COLOR_BGR2HSV)

	mask1 = cv2.inRange(ball_hsv, lower_yellow, upper_yellow)
	ball_result = cv2.bitwise_and(ball_img, ball_img, mask=mask1)
	ball_result = cv2.cvtColor(ball_result,cv2.COLOR_HSV2BGR)
	ball_result = cv2.cvtColor(ball_result,cv2.COLOR_BGR2GRAY)
	count_ball = cv2.countNonZero(ball_result)


	if(count_ball >= 10):
		cv2.putText(img, 'football', (x-2, y-2), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

		"""
	cv2.imwrite("./Cropped/frame%d.jpg" % count, result)
	print 'Read a new frame: ', truth
	count += 1
	cv2.imshow('Match Detection',img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	truth,img = cap.read()

vidcap.release()
cv2.destroyAllWindows()
