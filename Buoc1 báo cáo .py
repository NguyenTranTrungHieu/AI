import cv2
import os

detector = cv2.CascadeClassifier('C:/Users/ACER/Desktop/AI Nhan dang khuon mat de xac minh danh tinh/haarcascade_frontalface_default.xml')
for i in range(1, 6):  # đưa 5 khuôn mặt vào
    for j in range(1, 21):  # mỗi khuôn mặt 20 ảnh tổng 100 ảnh
        file = 'ANHGOC/anh'+ str(i) + '.' + str(j) + '.jpg'
        frame = cv2.imread(file)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fa = detector.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in fa:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if not os.path.exists('C:/Users/ACER/Desktop/AI Nhan dang khuon mat de xac minh danh tinh/anhluulai'):
               os.makedirs('C:/Users/ACER/Desktop/AI Nhan dang khuon mat de xac minh danh tinh/anhluulai')
            
            cv2.imwrite('C:/Users/ACER/Desktop/AI Nhan dang khuon mat de xac minh danh tinh/anhluulai/' + 'anhmoi'  + str(i) + '.' + str(j) + '.jpg', gray[y:y+h, x:x+w])

        
