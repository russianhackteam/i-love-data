import cv2


vidcap = cv2.VideoCapture('video.mp4')# Находим видео с банками и парсим из него кадры
success,image = vidcap.read()
count = 0
id = 0
while success:
	print('Read a new frame: ', count)
	if count%120 == 0 : # Сохраняем каждый 120 кадр
		resize = cv2.resize(image, (757, 320), interpolation = cv2.INTER_LINEAR) 
		cv2.imwrite("./trainA/frame%d.jpg" % id, resize)
		id += 1
		print('save()')
	success,image = vidcap.read()
	count += 1
		
