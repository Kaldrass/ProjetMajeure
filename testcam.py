import cv2
import matplotlib.pyplot as plt

index = 0
arr = []
while True:
    cap = cv2.VideoCapture(index)
    if not cap.read()[0]:
        break
    else:
        arr.append(index)
        ret,img = cap.read()
        plt.subplot(2,2,index+1)
        plt.imshow(img)
    cap.release()
    index += 1
print(arr)
plt.show()