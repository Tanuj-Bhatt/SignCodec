import cv2 
cap = cv2.VideoCapture(0) 
ret, frame = cap.read() 
print('Webcam OK:', ret, frame.shape if ret else None) 
cap.release() 
