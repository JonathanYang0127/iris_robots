import cv2
import time

# for i in range(20):
#     cam = cv2.VideoCapture(i)
#     if cam.isOpened():
#         print(f'Working camera index: {i}')

cam = cv2.VideoCapture(4)
while True:
    ret, img = cam.read()
    img = img[:,80:560,:]
    cv2.imwrite('x.png', img)
    time.sleep(1)
