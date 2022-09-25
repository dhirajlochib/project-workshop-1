import cv2, time, pandas
from datetime import datetime

static_obj = None
mover = [ None, None ]
time = []

df = pandas.DataFrame(columns = ["Start", "End"])
video = cv2.VideoCapture(0)
while True:
	check, frame = video.read()
	motion = 0
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	if static_obj is None:
		static_obj = gray
		continue

	diff_frame = cv2.absdiff(static_obj, gray)
	thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
	thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)

	cnts,_ = cv2.findContours(thresh_frame.copy(),
					cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for contour in cnts:
		if cv2.contourArea(contour) < 10000:
			continue
		motion = 1
		(x, y, w, h) = cv2.boundingRect(contour)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
	mover.append(motion)
	mover = mover[-2:]
	
	if mover[-1] == 1 and mover[-2] == 0:
		time.append(datetime.now())

	
	if mover[-1] == 0 and mover[-2] == 1:
		time.append(datetime.now())

	cv2.imshow("Color Frame", frame)
	key = cv2.waitKey(1)
	if key == ord('q'):
		if motion == 1:
			time.append(datetime.now())
		break

for i in range(0, len(time), 2):
	df = df.append({"Start":time[i], "End":time[i + 1]}, ignore_index = True)

df.to_csv("Time_of_movements.csv")
video.release()

cv2.destroyAllWindows()